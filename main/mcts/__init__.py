import numpy as np
from actionspace import map_w, map_b
from game import Game
from gameimage import c as gic
from funcs import predict_fn
from math import log, exp
import chess

# Each state-action pair (s, a) stores a set of statistics:
#     - N(s, a): visit count
#     - W(s, a): total action value
#     - Q(s, a): mean action value
#     - P(s, a): prior probability of selecting action a in state s

# A node stores the state. The action is the move that led to the state.
# For the root node the action is None. For all other nodes the action is the key in the children dictionary of the parent node.

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


def run_mcts(config, game: Game, network, root: Node = None, reuse_tree: bool = False, num_simulations: int = 100, num_sampling_moves: int = 30, add_noise: bool = True, cpuct: float = None):
    """Run Monte Carlo Tree Search algorithm from the current game state.

    Args:
        game (Game): Current game state.
        sim_num (int, optional): Number of simulations to run. Defaults to 100.

    Returns:
        Move, Root: Return the move to play and the root node with an updated tree layout.
    """
    rng = np.random.default_rng()

    # Initialize the root node
    # Allows us to reuse the image of the previous root node but also allows us to reuse the tree or make clean start
    if root is None:
        root = Node()
        root.N = 1
        expand(root, game, network)
    else:
        if reuse_tree:
            num_simulations = num_simulations - root.N + 1
        else:
            root.reset()
            root.N = 1
            expand(root, game, network)

    # Exploration noise used for full searches
    if add_noise:
        add_exploration_noise(config, root, rng)

    # Run the simulations
    for _ in range(num_simulations):
        node = root
        tmp_game = game.clone()

        # Select the best leaf node
        while node.is_leaf() is False:
            move, node = select_leaf(config, node, cpuct)  # Select best with ucb
            tmp_game.make_move(move)

        # Evaluate the leaf node
        value, is_terminal = evaluate(tmp_game)

        # Expand if possible
        if is_terminal is False:
            value = expand(node, tmp_game, network)
        
        # Backpropagate the value
        update(node, -value)
    # return best_move, root
    temp = config.temp if game.history_len < num_sampling_moves else 0
    return select_move(root, rng, temp), root

def get_image(game: Game, node: Node):
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

def expand(node: Node, game: Game, network=None):
    """Evaluate the given node using the neural network.

    Args:
        node (Node): Node to evaluate.z
    """

    image = get_image(game, node)
    # Get the policy and value from the neural network
    value, policy_logits = predict_fn(network, image.astype(np.float32))
    value = np.array(value)
    policy_logits = np.array(policy_logits)

    policy = {}
    to_play = game.to_play()
    for move in game.board.legal_moves: # Acces via generator for speedup
        move_uci = chess.Move.uci(move)
        action = map_w[move_uci] if to_play else map_b[move_uci]
        policy[move_uci] = exp(policy_logits[action])

    policy_sum = sum(list(policy.values()))
    for move, p in policy.items():
        child = Node()
        child.parent = node
        child.P = p / policy_sum
        node.children[move] = child

    return value

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

def select_leaf(config, node: Node, cpuct = None):
    _, move, child = max(
        (ucb(config, node, child, cpuct), move, child)
        for move, child in node.children.items()
    )
    return move, child

def ucb(config, parent: Node, child: Node, cpuct = None):
    # ucb = Q(s, a) + U(s, a)
    # U(s, a) = C(s)*P(s, a)*sqrt(N(s))/(1 + N(s, a))
        # Cs -> exploration constant
        # Psa -> child prior value
        # Ns -> parent visits count
        # Nsa -> child visits count
    # Q(s, a)
        # Q of child
    if cpuct is None: # AlphaZero exploration value, 1 for no exploration
        cpuct = (
            log((parent.N + config.pb_c_base + 1) / config.pb_c_base)
            + config.pb_c_init
        )
    return child.Q + cpuct * child.P * parent.N**0.5 / (child.N + 1)

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
