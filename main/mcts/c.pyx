import numpy as np
cimport numpy as np
from game import Game
from actionspace import map_w, map_b
from tf_funcs import predict_fn
from gameimage.c cimport board_to_image, update_image
from libc.math cimport log
import time
np.import_array()

class DebugNode:
    def __init__(self, move, node):
        self.move = move
        self.N = node.N
        self.W = node.W
        self.Q = node.Q
        self.P = node.P

    def __repr__(self) -> str:
        def to_str(value):
            if value < 0:
                return f"{value:.6f}"
            else:
                return f" {value:.6f}"
        
        return "%-7s %-10s  %-13s  %-13s" %(self.move, f"N: {self.N}", f"(P:{to_str(self.P)})", f"(Q:{to_str(self.Q)})")

class Node:
    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else self.fpu

    def __init__(self, fpu=0.0):
        # Values for tree representation
        self.children = {}
        self.image = None
        self.fpu = fpu # First play urgency 1 for root children, 0 for others
        self.to_play = None
        self.vloss_count = 0
        self.waiting_for_eval = False

        # Values for MCTS
        self.N = 0
        self.W = 0.0
        self.P = 0.0

    def __getitem__(self, str move):
        return self.children[move]

    def __setitem__(self, str move, object child):
        self.children[move] = child

    def is_leaf(self):
        return len(self.children) == 0

    def find_best_variation(self, game):
        best_child = max(self.children.values(), key=lambda child: child.N)
        move = list(self.children.keys())[list(self.children.values()).index(best_child)]
        game.make_move(move)
        if best_child.is_leaf():
            return [move]
        else:
            return [move] + best_child.find_best_variation(game)

    def display_move_statistics(self, game: Game = None):
        debug_nodes = [DebugNode(move, child) for move, child in self.children.items()]
        sorted_children = sorted(debug_nodes, key=lambda x: x.N, reverse=True) # Sort by visit count
        best_child = list(sorted_children)[0]

        print(f"Visits: {self.N}")
        print(f"Value: {best_child.Q:.6f}")
        if game is not None:
            tmp_game = game.clone()
            print(f"Best variation: {' -> '.join(self.find_best_variation(tmp_game))}")

        print("%-7s %-10s  %-13s  %-13s" %("Move", "Visits", "Prior", "Avg. value"))
        print("-" * 130)
        for child in sorted_children:
            print(child)


def run_mcts(
        game: Game, 
        config, 
        trt_func,
        num_simulations: int = 0,  
        time_limit: float = None,
        root: Node = None,
        **kwargs):

    assert num_simulations > 0 or time_limit is not None, "Either num_simulations or time_limit must be set"

    MIN_EXPLORE = kwargs.get("minimal_exploration", False)
    ENGINE_PLAY = kwargs.get("engine_play", False)
    VERBOSE_MOVE = kwargs.get("verbose_move", False)
    RETURN_STATS = kwargs.get("return_statistics", False)

    start_time = time.time()
    if MIN_EXPLORE:
        fpu_root, fpu_leaf = 0.0, 0.0
        pb_c_factor_root, pb_c_factor_leaf = 1.0, 1.0
        policy_temp = 1.0
    else:
        fpu_root, fpu_leaf = config.fpu_root, config.fpu_leaf
        pb_c_factor_root, pb_c_factor_leaf = config.pb_c_factor[0], config.pb_c_factor[1]
        policy_temp = config.policy_temp
    pb_c_base = config.pb_c_base
    pb_c_init = config.pb_c_init
    T = config.history_steps
    R = config.repetition_planes
    F = config.history_perspective_flip
    discourage_draws_value = config.discourage_draws_value

    rng = np.random.default_rng()

    if root is None:
        root = Node()

    if root.is_leaf():
        if root.image is None:
            root.image = board_to_image(game.board, T, R, F)
        values, policy_logits = make_predictions(
                config=config,
                trt_func=trt_func,
                images=np.array([root.image], dtype=np.float32),
                fill_buffer=True
            )
        expand_node(root, game, fpu_root)
        evaluate_node(root, policy_logits[0], policy_temp)

    if len(root.children) == 1:
        # If there is only one legal move, return it immediately
        move = list(root.children.keys())[0] 
        root[move].N += 1
        child_visits = calculate_search_statistics(root, config.num_actions) if RETURN_STATS else None
        if VERBOSE_MOVE:
            print(f"Time for MCTS: {time.time() - start_time:.2f}s")
            root.display_move_statistics(game)
        return move, root, child_visits

    if not MIN_EXPLORE and not ENGINE_PLAY:
        add_exploration_noise(config, root, rng)

    while True:
        if time_limit is not None:
            if time.time() - start_time > time_limit:
                break
            nodes_to_find = config.num_parallel_reads
        else:
            if root.N >= num_simulations:
                break
            nodes_to_find = config.num_parallel_reads if root.N + config.num_parallel_reads <= num_simulations else num_simulations - root.N
            
        nodes_to_eval = []
        search_paths = []
        failsafe = 0
        while len(nodes_to_eval) < nodes_to_find and failsafe < nodes_to_find * 2:
            node = root
            search_path = [node]
            tmp_game = game.clone()

            # Traverse tree and find a leaf node
            try:
                while not node.is_leaf():
                    pb_c_factor = pb_c_factor_root if len(search_path) == 1 else pb_c_factor_leaf
                    move, node = select_leaf(node, pb_c_base=pb_c_base, pb_c_init=pb_c_init, pb_c_factor=pb_c_factor)
                    search_path.append(node)
                    tmp_game.make_move(move)
            except:
                # Selection process encountered a node where each child is waiting for evaluation
                failsafe += 1
                continue
                

            # Check if game ends here
            value, terminal = evaluate_game(tmp_game)
            # If the game is terminal, backup the value and continue with the next simulation
            if terminal:
                if not ENGINE_PLAY and value == 0.5: # Discourage draws
                    value = value + discourage_draws_value
                node.to_play = tmp_game.to_play()
                update(search_path, flip_value(value))
                failsafe += 1
                continue

            # Expansion
            if node.image is None:
                node.image = update_image(tmp_game.board, search_path[-2].image.copy(), T, R, F)
            expand_node(node, tmp_game, fpu_leaf)
            add_vloss(search_path)

            # Save the nodes to evaluate and the search paths
            nodes_to_eval.append(node)
            search_paths.append(search_path)
            node.waiting_for_eval = True

        if not nodes_to_eval:
            failsafe += 1
            continue

        values, policy_logits = make_predictions(
                config=config,
                trt_func=trt_func,
                images=np.array([node.image for node in nodes_to_eval], dtype=np.float32),
                fill_buffer=len(nodes_to_eval) < config.num_parallel_reads
            )

        for i, node in enumerate(nodes_to_eval):
            evaluate_node(node, policy_logits[i], policy_temp)
            remove_vloss(search_paths[i])
            update(search_paths[i], flip_value(value_to_01(values[i].item())))

        del nodes_to_eval, values, policy_logits, search_paths
    
    end_time = time.time()
    if VERBOSE_MOVE:
        print(f"Time for MCTS: {end_time - start_time:.2f}s")
        root.display_move_statistics(game)

    if ENGINE_PLAY:
        move = select_move(root, rng, 0.0)
    else:
        move = select_move(root, rng, config.softmax_temp if game.history_len < config.num_mcts_sampling_moves else 0.0)
        if game.resignable and game.history_len > config.resignation_move_limit and root[move].Q < config.resignation_threshold:
            move = None

    child_visits = calculate_search_statistics(root, config.num_actions) if RETURN_STATS else None
    return move, root, child_visits


def expand_node(node: Node, game: Game, fpu: float):
    node.to_play = game.to_play()
    for move in game.board.legal_moves:
        node[move.uci()] = Node(fpu=fpu)


cdef void evaluate_node(object node, float[:] policy_logits, float policy_temp):
    cdef object child
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
    node.waiting_for_eval = False


def evaluate_game(game: Game):
    if game.terminal_with_outcome():
        return value_to_01(game.terminal_value(game.to_play())), True
    else:
        return value_to_01(0.0), False


def make_predictions(config, trt_func, images, fill_buffer=False):
    images[:, -1] /= 99.0
    if fill_buffer:
        fill = np.zeros((config.num_parallel_reads - len(images), config.image_shape[0], 8, 8), dtype=np.float32)
        batch = np.concatenate((images, fill), axis=0)
        assert len(images) + len(fill) == config.num_parallel_reads, f"Expected {config.num_parallel_reads} images, got {len(images)}"
        assert len(batch) == config.num_parallel_reads, f"Expected {config.num_parallel_reads} images, got {len(batch)}"
        values, policy_logits = predict_fn(trt_func, batch)
    else:
        assert len(images) == config.num_parallel_reads, f"Expected {config.num_parallel_reads} images, got {len(images)}"
        values, policy_logits = predict_fn(trt_func, images)
    return values.numpy(), policy_logits.numpy()


cdef value_to_01(float value):
    return (value + 1.0) / 2.0


cdef flip_value(float value):
    return 1 - value


def update(search_path, value):
    for node in reversed(search_path):
        node.N += 1
        node.W += value
        value = flip_value(value)


def add_vloss(search_path):
    for node in reversed(search_path):
        node.W -= 1
        node.vloss_count += 1


def remove_vloss(search_path):
    for node in reversed(search_path):
        node.W += node.vloss_count
        node.vloss_count = 0


cdef select_leaf(object node, float pb_c_base, float pb_c_init, float pb_c_factor):
    cdef str move
    cdef str bestmove
    cdef float ucb
    cdef float bestucb = -np.inf
    cdef object child
    cdef object bestchild

    for move, child in node.children.items():
        if child.waiting_for_eval:
            continue
        ucb = child.Q + UCB(child.P, child.N, node.N, pb_c_base, pb_c_init, pb_c_factor)
        if ucb > bestucb:
            bestucb = ucb
            bestmove = move
            bestchild = child
    
    return bestmove, bestchild 


cdef UCB(float cP, int cN, int pN, float pb_c_base, float pb_c_init, float pb_c_factor):
    cdef float cpuct
    cpuct = (
        log((pN + pb_c_base + 1) / pb_c_base) + pb_c_init
    ) * pb_c_factor
    return cpuct * cP * pN**0.5 / (cN + 1)    


def select_move(node: Node, rng: np.random.Generator, temp = 0.0):
    moves = [move for move in node.children.keys()]
    visit_counts = [child.N for child in node.children.values()]
    if temp == 0.0:
        # Greedy selection (select the move with the highest visit count) 
        # If more moves have the same visit count, choose one randomly
        return moves[rng.choice(np.flatnonzero(visit_counts == np.max(visit_counts)))]
    else:
        # Use the visit counts as a probability distribution to select a move
        pi = np.array(visit_counts) ** (1 / temp)
        pi /= np.sum(pi)
        return moves[np.where(rng.multinomial(1, pi) == 1)[0][0]]


def add_exploration_noise(config, node: Node, rng: np.random.Generator):
    noise = rng.gamma(config.root_dirichlet_alpha, 1, len(node.children))
    noise /= np.sum(noise)
    frac = config.root_exploration_fraction
    for n, child in zip(noise, node.children.values()):
        child.P = child.P * (1 - frac) + n * frac


cdef calculate_search_statistics(object root, int num_actions):
    cdef np.ndarray[double, ndim=1] child_visits = np.zeros(num_actions, dtype=np.float64)
    cdef int sum_visits = 0
    cdef str move
    cdef object child

    for move, child in root.children.items():
        child_visits[map_w[move] if root.to_play else map_b[move]] = child.N
        sum_visits += child.N
    return child_visits / sum_visits