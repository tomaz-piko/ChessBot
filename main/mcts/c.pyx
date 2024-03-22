import numpy as np
cimport numpy as np
from game import Game
from actionspace import map_w, map_b
from tf_funcs import predict_fn
from gameimage.c cimport board_to_image, update_image
from libc.math cimport log
np.import_array()

class Node:
    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else self.fpu

    def __init__(self, parent=None, fpu=0.0):
        # Values for tree representation
        self.parent = parent
        self.children = {}
        self.image = None
        self.fpu = fpu # First play urgency 1 for root children, 0 for others
        self.to_play = None

        # Values for MCTS
        self.N = 0
        self.W = 0.0
        self.P = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None
    
    def reset(self):
        self.parent = None
        self.children = {}
        self.N = 0
        self.W = 0
        self.P = 0

    def describe(self):
        # Sort children by visit count into new dict
        sorted_children = {k: v for k, v in sorted(self.children.items(), key=lambda item: item[1].N, reverse=True)}
        for move, child in sorted_children.items():
            print(f"{move} -> P: {child.P:.6f}, Q: {child.Q:.6f}, W: {child.W:.6f}, N: {child.N}")


class Network:
    def __init__(self, trt_func, batch_size=8):
        self.trt_func = trt_func
        self.batch_size = batch_size

    def __call__(self, images, fill_buffer=False):
        images[:, -1] /= 99.0
        if fill_buffer:
            batch = np.zeros((self.batch_size, 109, 8, 8), dtype=np.float32)
            batch[:len(images)] = images
            values, policy_logits = predict_fn(self.trt_func, batch)
        else:
            values, policy_logits = predict_fn(self.trt_func, images)
             
        return values.numpy(), policy_logits.numpy()


def run_mcts(
        game: Game, 
        config, 
        trt_func,
        num_simulations: int,  
        minimal_exploration: bool = False, 
        root: Node = None,
        reuse_tree: bool = False,
        **kwargs):

    if minimal_exploration:
        fpu_root, fpu_leaf = 1.0, 0.0
        pb_c_factor_root, pb_c_factor_leaf = 1.0, 1.0
        policy_temp = 1.0
    else:
        fpu_root, fpu_leaf = config.fpu_root, config.fpu_leaf
        pb_c_factor_root, pb_c_factor_leaf = config.pb_c_factor[0], config.pb_c_factor[1]
        policy_temp = config.policy_temp
    pb_c_base = config.pb_c_base
    pb_c_init = config.pb_c_init

    #network = Network(trt_func, config.num_parallel_reads)
    rng = np.random.default_rng()

    if root is None:
        root = Node()
        reuse_tree = False # Can't reuse a tree if there is no tree to reuse
    elif not reuse_tree:
        root.reset()

    if root.is_leaf():
        if root.image is None:
            root.image = board_to_image(game.board)
        values, policy_logits = make_predictions(
                trt_func=trt_func,
                images=np.array([root.image], dtype=np.float32),
                batch_size=config.num_parallel_reads,
                fill_buffer=True
            )
        expand_node(root, game, fpu_root)
        evaluate_node(root, policy_logits[0], policy_temp)

    if not minimal_exploration and not reuse_tree:
        add_exploration_noise(config, root, rng)

    while root.N < num_simulations:
        nodes_to_eval = []
        nodes_to_find = config.num_parallel_reads if root.N + config.num_parallel_reads <= num_simulations else num_simulations - root.N
        while len(nodes_to_eval) < nodes_to_find:
            node = root
            tmp_game = game.clone()

            # Traverse tree and find a leaf node
            while not node.is_leaf():
                pb_c_factor = pb_c_factor_root if node.is_root() else pb_c_factor_leaf
                move, node = select_leaf(node, pb_c_base=pb_c_base, pb_c_init=pb_c_init, pb_c_factor=pb_c_factor)
                tmp_game.make_move(move)

            # Check if game ends here
            value, terminal = evaluate_game(tmp_game)
            # If the game is terminal, backup the value and continue with the next simulation
            if terminal:
                update(node, flip_value(value))
                if root.N == num_simulations:
                    break
                continue

            # Expansion
            node.image = update_image(tmp_game.board, node.parent.image.copy())
            expand_node(node, tmp_game, fpu_leaf)
            add_vloss(node)

            # Save the nodes to evaluate and the search paths
            nodes_to_eval.append(node)

        if not len(nodes_to_eval) > 0:
            continue

        values, policy_logits = make_predictions(
                trt_func=trt_func,
                images=np.array([node.image for node in nodes_to_eval], dtype=np.float32),
                batch_size=config.num_parallel_reads,
                fill_buffer=not nodes_to_find == config.num_parallel_reads
            )
        for i, (node) in enumerate(nodes_to_eval):
            evaluate_node(node, policy_logits[i], policy_temp)
            remove_vloss(node)
            update(node, flip_value(value_to_01(values[i].item())))
    
    move = select_move(root, rng, config.softmax_temp if game.history_len < config.num_mcts_sampling_moves else 0.0)
    if kwargs.get("return_statistics"):
        child_visits = calculate_search_statistics(root)
        return move, root, child_visits
    return move, root, None


def expand_node(node: Node, game: Game, fpu: float):
    node.to_play = game.to_play()
    for move in game.board.legal_moves:
        node.children[move.uci()] = Node(parent=node, fpu=fpu)


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


def evaluate_game(game: Game):
    if game.terminal_with_outcome():
        return value_to_01(game.terminal_value(game.to_play())), True
    else:
        return value_to_01(0.0), False


def make_predictions(trt_func, images, batch_size, fill_buffer=False):
    images[:, -1] /= 99.0
    if fill_buffer:
        batch = np.zeros((batch_size, 109, 8, 8), dtype=np.float32)
        batch[:len(images)] = images
        values, policy_logits = predict_fn(trt_func, batch)
    else:
        values, policy_logits = predict_fn(trt_func, images)
            
    return values.numpy(), policy_logits.numpy()


cdef value_to_01(float value):
    return (value + 1.0) / 2.0


cdef flip_value(float value):
    return 1 - value

 
def update(node: Node, value: float):
    node.N += 1
    node.W += value
    if not node.parent is None:
        update(node.parent, flip_value(value))


def add_vloss(node: Node):
    node.W -= 1
    if not node.parent is None:
        add_vloss(node.parent)


def remove_vloss(node: Node):
    node.W += 1
    if not node.parent is None:
        remove_vloss(node.parent)


cdef select_leaf(object node, float pb_c_base, float pb_c_init, float pb_c_factor):
    cdef str move
    cdef str bestmove
    cdef float ucb
    cdef float bestucb = -np.inf
    cdef object child
    cdef object bestchild

    for move, child in node.children.items():
        if pb_c_factor > 0.0:
            ucb = child.Q + UCB(child.P, child.N, node.N, pb_c_base, pb_c_init, pb_c_factor)
        else:
            ucb = child.Q
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


def select_move(node: Node, rng: np.random.Generator, temp = 0):
    moves = [move for move in node.children.keys()]
    visit_counts = [child.N for child in node.children.values()]
    if temp == 0:
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


cdef calculate_search_statistics(object root):
    cdef np.ndarray[double, ndim=1] child_visits = np.zeros(4672, dtype=np.float64)
    cdef int sum_visits = 0
    cdef str move
    cdef object child

    for move, child in root.children.items():
        child_visits[map_w[move] if root.to_play else map_b[move]] = child.N
        sum_visits += child.N
    return child_visits / sum_visits