# cython: language_level=3, boundscheck=False, nonecheck=False, initializedcheck=False, infer_types=True, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from gameimage.c cimport board_to_image, update_image, convert_to_model_input
import cppchess as chess
from game import Game
from tf_funcs import predict_fn
from actionspace import map_w, map_b
from libc.math cimport log, exp
import pympler.asizeof

cdef class Network:
    cdef object trt_func
    cdef dict cache
    cdef int cache_size
    cdef int ITEM_SIZE_EST # Pympler a size of value + policy_logits (np.float32)

    def __init__(self, trt_func, use_cache=True, cache_size=2048):
        self.trt_func = trt_func
        self.cache = {} if use_cache else None
        self.cache_size = cache_size if use_cache else 0
        self.ITEM_SIZE_EST = 18904

    def __cinit__(self, trt_func, use_cache=True, cache_size=2048):
        self.trt_func = trt_func
        self.cache = {} if use_cache else None
        self.cache_size = cache_size if use_cache else 0
        self.ITEM_SIZE_EST = 18904

    def __call__(self, np.ndarray image, is_cacheable=True):
        cdef object value
        cdef object policy_logits
        cdef np.ndarray[np.float32_t, ndim=4] image_np
        cdef int image_hash

        if self.cache_size > 0 and is_cacheable:
            image_hash = hash(image.tobytes())
            if image_hash in self.cache:
                value, policy_logits = self.cache[image_hash]
            else:
                image_np = convert_to_model_input(image)
                value, policy_logits = predict_fn(self.trt_func, image_np)
                value = value.numpy().item()
                policy_logits = policy_logits.numpy().flatten()
                if (self.cache_items + 1) * self.ITEM_SIZE_EST < self.cache_size * 1024:
                    self.cache[image_hash] = (value, policy_logits)
            return value_to_01(value), policy_logits
        else:
            image_np = convert_to_model_input(image)
            value, policy_logits = predict_fn(self.trt_func, image_np)
            return value_to_01(value.numpy().item()), policy_logits.numpy().flatten()

    cpdef void clear_cache(self):
        self.cache.clear()

    @property
    def cache_items(self):
        return len(self.cache)

    @property
    def cache_size(self):
        return sizeof(self.cache)


cdef class Node:
    cdef public int N
    cdef public float P
    cdef public float W
    cdef public float fpu
    cdef public dict children
    cdef public np.ndarray image

    def __init__(self, P=0.0, fpu=0.0):
        self.N = 0
        self.P = P
        self.W = 0.0
        self.fpu = fpu
        self.children = {}
        self.image = None

    def __cinit__(self, float P=0.0, float fpu=0.0):
        self.N = 0
        self.P = P
        self.W = 0.0
        self.fpu = fpu
        self.children = {}
        self.image = None

    cdef float Q(self) except *:
        return self.W / (self.N) if self.N > 0 else self.fpu

    cdef bint is_leaf(self) except *:
        return len(self.children) == 0

    cdef void add_child(self, str move, float P, float fpu) except *:
        self.children[move] = Node(P, fpu)
    

cdef class MCTS:
    # Search properties
    cdef public Node root
    cdef public Network network
    cdef public object game
    cdef object rng
    cdef bint selfplay

    # MCTS constants
    cdef float pb_c_base, pb_c_init
    cdef float pb_c_factor_root, pb_c_factor_leaf,
    cdef float policy_temp, softmax_temp
    cdef float fpu_root, fpu_leaf
    cdef float dirichlet_alpha, exploration_fraction
    cdef int num_sampling_moves

    def __init__(self, config, game, trt_func, selfplay=False):
        self.root = Node()
        self.game = game
        self.network = Network(trt_func)
        self.selfplay = selfplay
        self.rng = np.random.default_rng()

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


    def __cinit__(self, config, game, trt_func, selfplay=False):
        self.root = Node()
        self.game = game
        self.network = Network(trt_func)
        self.selfplay = selfplay
        self.rng = np.random.default_rng()

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
            ucb = child.Q() + self.UCB(child.P, child.N, node.N, pb_c_factor)
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
        return cpuct * cP * pN**0.5 / (cN + 1)

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

    @cython.wraparound(False)
    cdef void expand(self, Node node, game, float[:] policy_logits, float fpu):
        cdef list policy = []
        cdef list moves = []
        cdef np.ndarray[double, ndim=1] policy_np
        cdef bint to_play = game.to_play()
        cdef object move
        cdef str move_uci
        cdef int action
        cdef float p
        cdef float _max
        cdef float expsum

        for move in game.board.legal_moves: # Acces via generator for speedup
            move_uci = move.uci()
            action = map_w[move_uci] if to_play else map_b[move_uci]
            moves.append(move_uci)
            policy.append(policy_logits[action])

        policy_np = np.array(policy)
        _max = policy_np.max()
        expsum = np.sum(np.exp(policy_np - _max))
        policy_np = np.exp(policy_np - (_max + np.log(expsum)))
        if self.policy_temp > 0.0 and self.policy_temp != 1.0:
            policy_np = policy_np ** (1.0 / self.policy_temp)
            policy_np /= np.sum(policy_np) 

        for move_uci, p in zip(moves, policy_np):
            node.add_child(move_uci, p, fpu)

    @cython.wraparound(False)
    cdef void backup(self, list search_path, float value):
        for node in search_path:
            node.N += 1
            node.W += value
            value = flip_value(value)

    cpdef str run(self, int num_simulations):
        cdef Node parent
        cdef Node node
        cdef float value
        cdef bint terminal
        cdef np.ndarray[float, ndim=1] policy_logits
        cdef list search_path
        cdef float fb_c_factor
        cdef str move

        if self.root.is_leaf():
            if self.root.image is None:
                self.root.image = board_to_image(self.game.board)
            #_, policy_logits = make_predictions(self.network, self.root.image)
            _, policy_logits = self.network(self.root.image)
            self.expand(self.root, self.game, policy_logits, self.fpu_root)

        if self.selfplay:
            self.add_exploration_noise(self.root)

        for _ in xrange(num_simulations):
            node = self.root
            search_path = [node]
            tmp_game = self.game.clone()
            while not node.is_leaf():
                pb_c_factor = self.pb_c_factor_root if len(search_path) == 1 else self.pb_c_factor_leaf
                move, node = self.select(node, pb_c_factor)
                tmp_game.make_move(move)
                search_path.append(node)

            value, terminal = evaluate(tmp_game)
            if not terminal:
                parent = search_path[-2]
                node.image = update_image(tmp_game.board, parent.image)
                #value, policy_logits = make_predictions(self.network, node.image)
                value, policy_logits = self.network(node.image, is_cacheable=True if tmp_game.history_len < 20 else False)
                self.expand(node, tmp_game, policy_logits, self.fpu_leaf)
            
            self.backup(search_path, flip_value(value))

        return self.select_move(self.root, self.softmax_temp if self.game.history_len < self.num_sampling_moves else 0.0)

    @cython.wraparound(False)
    cdef void add_exploration_noise(self, Node node):
        noise = self.rng.gamma(self.dirichlet_alpha, 1, len(node.children))
        noise /= noise.sum()
        for n, child in zip(noise, node.children.values()):
            child.P = child.P * (1 - self.exploration_fraction) + n * self.exploration_fraction

@cython.wraparound(False)
cdef make_predictions(object network, np.ndarray image):
    cdef object value
    cdef object policy_logits
    cdef float value_f
    cdef np.ndarray policy_logits_np

    value, policy_logits = predict_fn(network, convert_to_model_input(image))
    value_f = value.numpy().item()
    policy_logits_np = policy_logits.numpy().flatten()
    return value_to_01(value_f), policy_logits_np
        
@cython.wraparound(False)
cdef (float, bint) evaluate(game):
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