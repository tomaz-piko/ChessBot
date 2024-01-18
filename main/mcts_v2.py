from config import Config
import numpy as np
import actionspace as asp
from game import Game
from model_v2 import predict_fn, predict_model
import math


config = Config()

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
        self.state = None

        # Values for MCTS
        self.N = 0
        self.W = 0
        self.P = 0
        self.Q = 0

    def is_leaf(self):
        return len(self.children) == 0


def run_mcts(game: Game, num_simulations: int = 100, network=None, trt: bool = True):
    """Run Monte Carlo Tree Search algorithm from the current game state.

    Args:
        game (Game): Current game state.
        sim_num (int, optional): Number of simulations to run. Defaults to 100.

    Returns:
        Move, Root: Return the move to play and the root node with an updated tree layout.
    """
    # Initialize the root node
    root = Node()
    root.N = 1
    expand(root, game, network, trt)
    add_exploration_noise(root)

    # Run the simulations
    for _ in range(num_simulations):
        node = root
        tmp_game = game.clone()

        # Select the best leaf node
        while not node.is_leaf():
            move, node = select_leaf(node)  # Select best with ucb
            tmp_game.make_move(move)

        # Evaluate the leaf node
        value, is_terminal = evaluate(tmp_game)

        # Expand if possible
        if not is_terminal:
            value = expand(node, tmp_game, network, trt)
        
        # Backpropagate the value
        update(node, -value)
    # return best_move, root
    temp = config.temp if game.history_len < config.num_sampling_moves else 0
    return select_move(root, temp), root


def expand(node: Node, game: Game, network=None, trt: bool = True):
    """Evaluate the given node using the neural network.

    Args:
        node (Node): Node to evaluate.z
    """
    # Get the policy and value from the neural network
    if node.state is None:
        node.state = game.make_image(-1)
    value, policy_logits = predict_fn(network, node.state.astype(np.float32)) if trt else predict_model(network, node.state.astype(np.float32))
    value = np.array(value)
    policy_logits = np.array(policy_logits)

    policy = {}
    for move in game.legal_moves():
        action = asp.uci_to_action(move, game.to_play())
        policy[move] = np.exp(np.float128(policy_logits[action]), dtype=np.float128)

    policy_sum = np.sum(list(policy.values()), dtype=np.float128)
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
    if node.parent:
        update(node.parent, -value)

def select_leaf(node: Node):
    _, move, child = max(
        (ucb(node, child), move, child)
        for move, child in node.children.items()
    )
    return move, child

def ucb(parent: Node, child: Node):
    # ucb = Q(s, a) + U(s, a)
    # U(s, a) = C(s)*P(s, a)*sqrt(N(s))/(1 + N(s, a))
        # Cs -> exploration constant
        # Psa -> child prior value
        # Ns -> parent visits count
        # Nsa -> child visits count
    # Q(s, a)
        # Q of child
    #C = 1 # AZ uses its own C
    C = (
        math.log((parent.N + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    return child.Q + C * child.P * parent.N**0.5 / (child.N + 1)

def select_move(node: Node, temp = 0):
    moves = [move for move in node.children.keys()]
    visit_counts = [child.N for child in node.children.values()]
    if temp == 0:
        # Greedy selection (select the move with the highest visit count) 
        # If more moves have the same visit count, choose one randomly
        return moves[np.random.default_rng().choice(np.flatnonzero(visit_counts == np.max(visit_counts)))]
    else:
        # Use the visit counts as a probability distribution to select a move
        pi = np.array(visit_counts) ** (1 / temp)
        pi /= np.sum(pi)
        return moves[np.where(np.random.default_rng().multinomial(1, pi) == 1)[0][0]]


def add_exploration_noise(node: Node):
    moves = node.children.keys()
    noise = np.random.default_rng().gamma(config.root_dirichlet_alpha, 1, len(moves))
    frac = config.root_exploration_fraction
    for i, move in enumerate(moves):
        node.children[move].P = node.children[move].P * (1 - frac) + noise[i] * frac
