from game import Game
import chess
import numpy as np
from actionspace import map_w, map_b
import main.gameimage.c as gic
from funcs import predict_fn
cimport numpy as np
from libc.math cimport log, exp
np.import_array()

class Node:
    def __init__(self):
        # Values for tree representation
        self.parent = None
        self.children = {}
        self.image = None

        # Values for MCTS
        self.N = 0
        self.W = 0
        self.P = 0
        self.Q = 0

    def is_leaf(self):
        return len(self.children) == 0
    
    def reset(self):
        self.parent = None
        self.children = {}
        self.N = 0
        self.W = 0
        self.P = 0
        self.Q = 0


def run_mcts(config, game: Game, network, root: Node = None, reuse_tree: bool = False, num_simulations: int = 100, num_sampling_moves: int = 30, add_noise: bool = True, pb_c_factor: float = 2.0):
    rng = np.random.default_rng()
    pb_c_base = config.pb_c_base
    pb_c_init = config.pb_c_init

    # Initialize the root node
    # Allows us to reuse the image of the previous root node but also allows us to reuse the tree or make clean start
    if root is None:
        root = Node()
        root.N = 1
        _, policy_logits = make_predictions(root, game, network)
        expand(root, game, policy_logits)
    else:
        if reuse_tree:
            num_simulations = num_simulations - root.N + 1
        else:
            root.reset()
            root.N = 1
            _, policy_logits = make_predictions(root, game, network)
            expand(root, game, policy_logits)

    # Exploration noise used for full searches
    if add_noise:
        add_exploration_noise(config, root, rng)

    # Run the simulations
    for _ in range(num_simulations):
        node = root
        tmp_game = game.clone()

        # Select the best leaf node
        while node.is_leaf() is False:
            move, node = select_leaf(node, pb_c_base, pb_c_init, pb_c_factor)  # Select best with ucb
            tmp_game.make_move(move)

        # Evaluate the leaf node
        value, is_terminal = evaluate(tmp_game)

        # Expand if possible
        if is_terminal is False:
            value, policy_logits = make_predictions(node, tmp_game, network)
            expand(node, tmp_game, policy_logits)
        
        # Backpropagate the value
        update(node, -value)
    # return best_move, root
    temp = config.temp if game.history_len < num_sampling_moves else 0
    return select_move(root, rng, temp), root

def get_image(node: Node, game: Game):
    if node.parent is None: # Root node
        if node.image is None:
            node.image = gic.board_to_image(game.board)
        return node.image
    else:
        if node.parent.image is None:
            node.image = gic.board_to_image(game.board)
        else:
            node.image = gic.update_image(game.board, node.parent.image)
        return node.image

def make_predictions(node: Node, game: Game, network):
    image = get_image(node, game)
    value, policy_logits = predict_fn(network, image.astype(np.float32))
    value = np.array(value)
    policy_logits = np.array(policy_logits)
    return value, policy_logits

cdef expand(object node, object game, float[:] policy_logits):
    cdef dict policy = {}
    cdef bint to_play = game.to_play()
    cdef object move
    cdef str move_uci
    cdef int action
    cdef double p
    cdef double policy_sum = 0

    for move in game.board.legal_moves: # Acces via generator for speedup
        move_uci = chess.Move.uci(move)
        action = map_w[move_uci] if to_play else map_b[move_uci]
        p = exp(policy_logits[action])
        policy[move_uci] = p 
        policy_sum += p

    for move_uci, p in policy.items():
        child = Node()
        child.parent = node
        child.P = p / policy_sum
        node.children[move_uci] = child

def evaluate(game: Game):
    if game.terminal():
        return game.terminal_value(game.to_play()), True
    else:
        return 0, False
    
def update(node: Node, value: float):
    node.N += 1
    node.W += value
    node.Q = node.W / node.N
    if not node.parent.parent is None:
        update(node.parent, -value)

cdef select_leaf(object node, float pb_c_base, float pb_c_init, float pb_c_factor):
    cdef str move
    cdef str bestmove
    cdef float ucb
    cdef float bestucb = -np.inf
    cdef object child
    cdef object bestchild

    for move, child in node.children.items():
        if pb_c_factor > 0.0:
            ucb = UCB(child.Q, child.P, child.N, node.N, pb_c_base, pb_c_init, pb_c_factor)
        else:
            ucb = child.Q
        if ucb > bestucb:
            bestucb = ucb
            bestmove = move
            bestchild = child
    
    return bestmove, bestchild 

cdef UCB(float cQ, float cP, int cN, int pN, float pb_c_base, float pb_c_init, float pb_c_factor):
    cdef float cpuct

    cpuct = (
        log((pN + pb_c_base + 1) / pb_c_base) + pb_c_init
    ) * pb_c_factor
    return cQ + cpuct * cP * pN**0.5 / (cN + 1)    

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
    frac = config.root_exploration_fraction
    for i, (move, _) in enumerate(node.children.items()):
        node.children[move].P = node.children[move].P * (1 - frac) + noise[i] * frac