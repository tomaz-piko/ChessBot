import numpy as np
import actionspace as asp
from game import Game
from model import predict_fn
import math

TEMP = 1.0
PB_C_BASE = 19652
PB_C_INIT = 1.25
ROOT_DIRICHLET_ALPHA = 0.3
ROOT_EXPLORATION_FRACTION = 0.25

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

        # Values for MCTS
        self.N = 0
        self.W = 0
        self.P = 0
        self.Q = 0

    def is_leaf(self):
        return len(self.children) == 0


def run_mcts(game: Game, network, num_simulations: int = 100, num_sampling_moves = 30, add_noise = True, CPUCT = None):
    """Run Monte Carlo Tree Search algorithm from the current game state.

    Args:
        game (Game): Current game state.
        sim_num (int, optional): Number of simulations to run. Defaults to 100.

    Returns:
        Move, Root: Return the move to play and the root node with an updated tree layout.
    """
    rng = np.random.default_rng()
    # Initialize the root node
    root = Node()
    root.N = 1
    expand(root, game, network)
    if add_noise:
        add_exploration_noise(root, rng)

    # Run the simulations
    for _ in range(num_simulations):
        node = root
        tmp_game = game.clone()

        # Select the best leaf node
        while not node.is_leaf():
            move, node = select_leaf(node, CPUCT)  # Select best with ucb
            tmp_game.make_move(move)

        # Evaluate the leaf node
        value, is_terminal = evaluate(tmp_game)

        # Expand if possible
        if not is_terminal:
            value = expand(node, tmp_game, network)
        
        # Backpropagate the value
        update(node, -value)
    # return best_move, root
    temp = TEMP if game.history_len < num_sampling_moves else 0
    return select_move(root, rng, temp), root


def expand(node: Node, game: Game, network=None):
    """Evaluate the given node using the neural network.

    Args:
        node (Node): Node to evaluate.z
    """
    # Get the policy and value from the neural network
    value, policy_logits = predict_fn(network, game.make_image(-1).astype(np.float32))
    value = np.array(value)
    policy_logits = np.array(policy_logits)

    policy = {}
    for move in game.legal_moves():
        action = asp.uci_to_action(move, game.to_play())
        policy[move] = np.exp(policy_logits[action], dtype=np.float32)

    policy_sum = np.sum(list(policy.values()), dtype=np.float32)
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

def select_leaf(node: Node, CPUCT = None):
    _, move, child = max(
        (ucb(node, child, CPUCT), move, child)
        for move, child in node.children.items()
    )
    return move, child

def ucb(parent: Node, child: Node, CPUCT = None):
    # ucb = Q(s, a) + U(s, a)
    # U(s, a) = C(s)*P(s, a)*sqrt(N(s))/(1 + N(s, a))
        # Cs -> exploration constant
        # Psa -> child prior value
        # Ns -> parent visits count
        # Nsa -> child visits count
    # Q(s, a)
        # Q of child
    if CPUCT is None: # AlphaZero exploration value, 1 for no exploration
        CPUCT = (
            math.log((parent.N + PB_C_BASE + 1) / PB_C_BASE)
            + PB_C_INIT
        )
    return child.Q + CPUCT * child.P * parent.N**0.5 / (child.N + 1)

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


def add_exploration_noise(node: Node, rng: np.random.Generator):
    moves = node.children.keys()
    noise = rng.gamma(ROOT_DIRICHLET_ALPHA, 1, len(moves))
    frac = ROOT_EXPLORATION_FRACTION
    for i, move in enumerate(moves):
        node.children[move].P = node.children[move].P * (1 - frac) + noise[i] * frac
