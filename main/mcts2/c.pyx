# cython: language_level=3, boundscheck=False, nonecheck=False, initializedcheck=False, infer_types=True, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from gameimage.c cimport board_to_image, update_image
from tf_funcs import predict_fn
from actionspace import map_w, map_b
from libc.math cimport log

cdef class Node:
    cdef public int N
    cdef public float P
    cdef public float W
    cdef public float fpu
    cdef public dict children
    cdef public np.ndarray image
    cdef public bint to_play
    cdef public int virtual_losses

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else self.fpu

    def __init__(self, fpu=0.0):
        self.N = 0
        self.P = 0.0
        self.W = 0.0
        self.fpu = fpu
        self.children = {}
        self.image = None
        self.virtual_losses = 0
        self.to_play = False

    def __cinit__(self, float fpu=0.0):
        self.N = 0
        self.P = 0.0
        self.W = 0.0
        self.fpu = fpu
        self.children = {}
        self.image = None
        self.virtual_losses = 0
        self.to_play = False

    def __getitem__(self, str move):
        return self.children[move]

    def __setitem__(self, str move, Node node):
        self.children[move] = node

    cpdef bint is_leaf(self) except *:
        return len(self.children) == 0

    cpdef reset(self):
        self.N = 0
        self.P = 0.0
        self.W = 0.0
        self.fpu = self.fpu
        self.children = {}
        self.virtual_losses = 0
        self.to_play = False

cdef class Network:
    cdef object trt_func
    cdef int batch_size

    def __init__(self, trt_func, batch_size=8):
        self.trt_func = trt_func
        self.batch_size = batch_size

    def __cinit__(self, object trt_func, int batch_size=8):
        self.trt_func = trt_func
        self.batch_size = batch_size

    def __call__(self, np.ndarray images, bint fill_buffer=False):
        cdef np.ndarray dummy_images
        cdef object values
        cdef object policy_logits
        
        images[:, -1] /= 99.0
        if fill_buffer:
            dummy_images = np.zeros((self.batch_size, 109, 8, 8), dtype=np.float32)
            dummy_images[:len(images)] = images
            values, policy_logits = predict_fn(self.trt_func, dummy_images)
        else:
            values, policy_logits = predict_fn(self.trt_func, images)
             
        return values.numpy(), policy_logits.numpy()

cdef class MCTS:
    # Search properties
    cdef public Node root
    cdef public Network network
    cdef object rng
    cdef int num_parallel_reads
    cdef bint engine_play

    # MCTS constants
    cdef float pb_c_base, pb_c_init
    cdef float pb_c_factor_root, pb_c_factor_leaf
    cdef float policy_temp, softmax_temp
    cdef float fpu_root, fpu_leaf
    cdef float dirichlet_alpha, exploration_fraction
    cdef int num_sampling_moves

    def __init__(self, config, trt_func, engine_play=True):
        self.root = Node()
        self.network = Network(trt_func, batch_size=config.num_parallel_reads)
        self.rng = np.random.default_rng()
        self.num_parallel_reads = config.num_parallel_reads
        self.engine_play = engine_play

        self.pb_c_base = config.pb_c_base
        self.pb_c_init = config.pb_c_init
        self.pb_c_factor_root = config.pb_c_factor[0]
        self.pb_c_factor_leaf = config.pb_c_factor[1]
        self.policy_temp = config.policy_temp
        self.softmax_temp = config.softmax_temp
        self.fpu_root = config.fpu_root
        self.fpu_leaf = config.fpu_leaf
        self.dirichlet_alpha = config.root_dirichlet_alpha
        self.exploration_fraction = config.root_exploration_fraction
        self.num_sampling_moves = config.num_mcts_sampling_moves

    def __cinit__(self, object config, object trt_func, engine_play=True):
        self.root = Node()
        self.network = Network(trt_func, batch_size=config.num_parallel_reads)
        self.rng = np.random.default_rng()
        self.num_parallel_reads = config.num_parallel_reads
        self.engine_play = engine_play

        self.pb_c_base = config.pb_c_base
        self.pb_c_init = config.pb_c_init
        self.pb_c_factor_root = config.pb_c_factor[0]
        self.pb_c_factor_leaf = config.pb_c_factor[1]
        self.policy_temp = config.policy_temp
        self.softmax_temp = config.softmax_temp
        self.fpu_root = config.fpu_root
        self.fpu_leaf = config.fpu_leaf
        self.dirichlet_alpha = config.root_dirichlet_alpha
        self.exploration_fraction = config.root_exploration_fraction
        self.num_sampling_moves = config.num_mcts_sampling_moves

    def __call__(self, object game, int num_simulations, float time_limit=0.0, bint minimal_exploration=False, bint return_statistics=False):
        cdef Node node, parent
        cdef object tmp_game
        cdef bint terminal
        cdef object values, policy_logits
        cdef list nodes_to_eval, search_path, search_paths
        cdef int nodes_to_find
        cdef Py_ssize_t i
        cdef float value, policy_temp, pb_c_factor_root, pb_c_factor_leaf, fpu_root, fpu_leaf
        
        if minimal_exploration:
            fpu_root, fpu_leaf = 1.0, 0.0
            pb_c_factor_root, pb_c_factor_leaf = 1.0, 1.0
            policy_temp = 1.0
        else:
            fpu_root, fpu_leaf = self.fpu_root, self.fpu_leaf
            pb_c_factor_root, pb_c_factor_leaf = self.pb_c_factor_root, self.pb_c_factor_leaf
            policy_temp = self.policy_temp

        if self.root.is_leaf():
            if self.root.image is None:
                self.root.image = board_to_image(game.board)
            values, policy_logits = self.network(
                images=np.array([self.root.image], dtype=np.float32),
                fill_buffer=True
            )
            self.expand_node(self.root, game, fpu_root)
            self.evaluate_node(self.root, policy_logits[0], policy_temp)

        if not minimal_exploration and not self.engine_play:
            self.add_exploration_noise(self.root)

        while self.root.N < num_simulations:
            nodes_to_eval = []
            search_paths = []

            nodes_to_find = self.num_parallel_reads if self.root.N + self.num_parallel_reads  <= num_simulations else num_simulations - self.root.N
            while len(nodes_to_eval) < nodes_to_find:
                node = self.root
                search_path = [node]
                tmp_game = game.clone()

                # Traverse tree and find a leaf node
                while not node.is_leaf():
                    pb_c_factor = pb_c_factor_root if len(search_path) == 1 else pb_c_factor_leaf
                    move, node = self.select(node, pb_c_factor)
                    tmp_game.make_move(move)
                    search_path.append(node)
                
                # Check if game ends here
                value, terminal = evaluate_game(tmp_game)
                # If the game is terminal, backup the value and continue with the next simulation
                if terminal:
                    self.backup(search_path, flip_value(value))
                    if self.root.N >= num_simulations:
                        break              
                    continue

                # Expansion
                parent = search_path[-2]
                node.image = update_image(tmp_game.board, parent.image.copy())
                self.expand_node(node, tmp_game, fpu_leaf)
                self.add_virtual_loss(search_path)

                # Save the nodes to evaluate and the search paths
                nodes_to_eval.append(node)
                search_paths.append(search_path)

            if not len(nodes_to_eval) > 0:
                continue        

            values, policy_logits = self.network(
                images=np.array([node.image for node in nodes_to_eval], dtype=np.float32),
                fill_buffer=not nodes_to_find == self.num_parallel_reads
            )

            for i, (node, search_path) in enumerate(zip(nodes_to_eval, search_paths)):
                self.evaluate_node(node, policy_logits[i], policy_temp)
                self.remove_virtual_loss(search_path)
                self.backup(search_path, flip_value(value_to_01(values[i].item())))

        if not self.engine_play:
            move = self.select_move(self.root, self.softmax_temp if game.history_len < self.num_sampling_moves else 0.0)
            child_visits = self.calculate_search_statistics() if return_statistics else None
            return move, child_visits
        else:
            return self.select_move(self.root, 0.0), None

    @cython.wraparound(False)
    cdef void expand_node(self, Node node, object game, float fpu):
        cdef bint to_play = game.to_play()
        cdef object move
        for move in game.board.legal_moves:
            node[move.uci()] = Node(fpu=fpu)

    @cython.wraparound(False)
    cdef void evaluate_node(self, Node node, float[:] policy_logits, float policy_temp):
        cdef Node child
        cdef str move_uci
        cdef float p
        cdef float _max
        cdef float expsum
        cdef np.ndarray[double, ndim=1] policy_np
        cdef list actions = [map_w[move_uci] if node.to_play else map_b[move_uci] for move_uci in node.children.keys()]
        cdef Py_ssize_t action
        cdef list policy_masked = [policy_logits[action] for action in actions]

        policy_np = np.array(policy_masked)

        _max = policy_np.max()
        expsum = np.sum(np.exp(policy_np - _max))
        policy_np = np.exp(policy_np - (_max + np.log(expsum)))
        if policy_temp > 0.0 and policy_temp != 1.0:
            policy_np = policy_np ** (1.0 / policy_temp)
            policy_np /= np.sum(policy_np)

        for p, child in zip(policy_np, node.children.values()):
            child.P = p

    cpdef reset(self):
        self.root = Node()

    @cython.wraparound(False)
    cdef select(self, Node node, float pb_c_factor):
        cdef str move
        cdef str bestmove
        cdef float ucb
        cdef float bestucb = -np.inf
        cdef Node child
        cdef Node bestchild

        for move, child in node.children.items():
            ucb = child.Q + self.UCB(child.P, child.N, node.N, pb_c_factor)
            if ucb > bestucb:
                bestucb = ucb
                bestmove = move
                bestchild = child

        return bestmove, bestchild

    @cython.wraparound(False)
    cdef float UCB(self, float cP, int cN, int pN, float pb_c_factor):
        cdef float cpuct
        cpuct = (
            log((pN + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        ) * pb_c_factor
        return cpuct * cP * (pN**0.5) / (cN + 1)

    @cython.wraparound(False)
    cdef str select_move(self, Node node, float softmax_temp = 0.0):
        cdef list moves = list(node.children.keys())
        cdef list visit_counts = [child.N for child in node.children.values()]
        cdef np.ndarray[double, ndim=1] pi

        if softmax_temp == 0.0:
            # Greedy selection (select the move with the highest visit count) 
            # If more moves have the same visit count, choose one randomly
            return moves[self.rng.choice(np.flatnonzero(visit_counts == np.max(visit_counts)))]
        else:
            # Use the visit counts as a probability distribution to select a move
            pi = np.array(visit_counts) ** (1 / softmax_temp)
            pi /= np.sum(pi)
            return moves[np.where(self.rng.multinomial(1, pi) == 1)[0][0]]

    cdef void add_virtual_loss(self, search_path):
        for node in search_path[::-1]:
            node.virtual_losses += 1
            node.W -= 1

    cdef void remove_virtual_loss(self, search_path):
        for node in search_path[::-1]:
            node.virtual_losses -= 1
            node.W += 1

    cdef void backup(self, list search_path, float value):
        for node in search_path[::-1]:
            node.N += 1
            node.W += value
            value = flip_value(value)

    @cython.wraparound(False)
    cdef void add_exploration_noise(self, Node node):
        noise = self.rng.gamma(self.dirichlet_alpha, 1, len(node.children))
        noise /= noise.sum()
        for n, child in zip(noise, node.children.values()):
            child.P = child.P * (1 - self.exploration_fraction) + n * self.exploration_fraction

    @cython.wraparound(False)
    cdef np.ndarray calculate_search_statistics(self):
        cdef np.ndarray[double, ndim=1] child_visits = np.zeros(4672, dtype=np.float64)
        cdef int sum_visits = 0
        cdef str move
        cdef Node child

        for move, child in self.root.children.items():
            child_visits[map_w[move] if self.root.to_play else map_b[move]] = child.N
            sum_visits += child.N
        return child_visits / sum_visits
        
@cython.wraparound(False)
cdef (float, bint) evaluate_game(object game):
    if game.terminal_with_outcome():
        return value_to_01(game.terminal_value(game.to_play())), True
    else:
        return value_to_01(0.0), False

@cython.wraparound(False)
cdef inline float value_to_01(float value):
    return (value + 1) / 2

@cython.wraparound(False)
cdef inline float flip_value(float value):
    return 1-value