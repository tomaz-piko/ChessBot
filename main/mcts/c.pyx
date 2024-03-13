import numpy as np
from game import Game
from actionspace import map_w, map_b
from tf_funcs import predict_fn
from time import time
cimport numpy as np
from gameimage.c cimport board_to_image, update_image, convert_to_model_input
from libc.math cimport log, exp, pow
np.import_array()

class Node:
    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else self.fpu

    def __init__(self, parent=None, P=0, fpu=0):
        # Values for tree representation
        self.parent = parent
        self.children = {}
        self.image = None
        self.fpu = fpu # First play urgency 1 for root children, 0 for others

        # Values for MCTS
        self.N = 0
        self.W = 0
        self.P = P

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

def run_mcts(config,
            game: Game, 
            network, 
            root: Node = None, 
            reuse_tree: bool = False, 
            num_simulations: int = 100, 
            num_sampling_moves: int = 30, 
            add_noise: bool = True,
            pb_c_factor = None,
            ):

    rng = np.random.default_rng()
    pb_c_base = config.pb_c_base
    pb_c_init = config.pb_c_init
    policy_temp = config.policy_temp
    softmax_temp = config.softmax_temp
    if pb_c_factor is None:
        pb_c_factor = config.pb_c_factor
    pb_c_factor_root = config.pb_c_factor[0]
    pb_c_factor_leaf = config.pb_c_factor[1]
    fpu_root = config.fpu_root
    fpu_leaf = config.fpu_leaf

    # Initialize the root node
    # Allows us to reuse the image of the previous root node but also allows us to reuse the tree or make clean start
    if root is None:
        root = Node()
        _, policy_logits = make_prediction(root, game, network)
        expand(root, game, policy_logits, policy_temp, fpu_root)
    else:
        if reuse_tree:
            num_simulations = num_simulations - root.N + 1
        else:
            root.reset()
            _, policy_logits = make_prediction(root, game, network)
            expand(root, game, policy_logits, policy_temp, fpu_root)

    # Exploration noise used for full searches
    if add_noise:
        add_exploration_noise(config, root, rng)

    # Run the simulations
    for _ in xrange(num_simulations):
        # Select the best leaf node
        node = root
        tmp_game = game.clone()
        while node.is_leaf() is False:
            pb_c_fact = pb_c_factor_root if node.is_root() else pb_c_factor_leaf
            move, node = select_leaf(node, pb_c_base, pb_c_init, pb_c_fact)  # Select best with ucb
            tmp_game.make_move(move)

        # Evaluate the leaf node
        value, is_terminal = evaluate(tmp_game)
        if not is_terminal:           
            value, policy_logits = make_prediction(root, game, network)
            expand(node, tmp_game, policy_logits, policy_temp, fpu_leaf)            

        # Backpropagate the value
        update(node, flip_value(value))

    # return best_move, root
    temp = softmax_temp if game.history_len < num_sampling_moves else 0
    return select_move(root, rng, temp), root


def get_image(node: Node, game: Game):
    if node.parent is None: # Root node
        if node.image is None:
            node.image = board_to_image(game.board)
        return convert_to_model_input(node.image)
    else:
        if node.parent.image is None:
            node.image = board_to_image(game.board)
        else:
            node.image = update_image(game.board, node.parent.image.copy())
        return convert_to_model_input(node.image)


def make_prediction(node: Node, game: Game, network):
    value, policy_logits = predict_fn(network, get_image(node, game))
    value = value.numpy().item()
    policy_logits = policy_logits.numpy().flatten()
    return value_to_01(value), policy_logits


cdef expand(object node, object game, float[:] policy_logits, float policy_temp = 1.0, float fpu = 1.0):
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
    if policy_temp > 0.0 and policy_temp != 1.0:
        policy_np = policy_np ** (1.0 / policy_temp)
        policy_np /= np.sum(policy_np) 

    for move_uci, p in zip(moves, policy_np):
        child = Node(parent=node, P=p, fpu=fpu)
        node.children[move_uci] = child


def evaluate(game: Game):
    if game.terminal_with_outcome():
        return value_to_01(game.terminal_value(game.to_play())), True
    else:
        return value_to_01(0.0), False


cdef value_to_01(float value):
    return (value + 1.0) / 2.0


cdef flip_value(float value):
    return 1 - value

 
def update(node: Node, value: float):
    node.N += 1
    node.W += value
    if not node.parent is None:
        update(node.parent, flip_value(value))


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