import numpy as np
from actionspace import map_w, map_b
from game import Game
from gameimage import board_to_image, update_image, convert_to_model_input
from tf_funcs import predict_fn
from math import log
import chess
from time import time

# Each state-action pair (s, a) stores a set of statistics:
#     - N(s, a): visit count
#     - W(s, a): total action value
#     - Q(s, a): mean action value
#     - P(s, a): prior probability of selecting action a in state s

# A node stores the state. The action is the move that led to the state.
# For the root node the action is None. For all other nodes the action is the key in the children dictionary of the parent node.
class DebugNode:
    def __init__(self, move, node):
        self.move = move
        self.N = node.N
        self.W = node.W
        self.Q = node.Q
        self.P = node.P
        self.value = node.init_eval
        self.policy = node.init_policy
        self.policy_norm = node.init_policy_norm
        self.puct = node.puct
        self.action = node.action

    def __repr__(self) -> str:
        def to_str(value):
            if value < 0:
                return f"{value:.6f}"
            else:
                return f" {value:.6f}"
        
        def to_percent(value):
            str = f"{value*100:.2f}%"
            for i in range(7-len(str)):
                str = " " + str
            return str
        
        return "%-5s (%-4s)  %-6s  %-26s  %-17s  %-13s  %-13s  %-13s  %-13s" %(self.move, self.action, f"N: {self.N}", f"(v:{to_str(self.value)}, p: {to_str(self.policy)})", f"(p_norm: {to_percent(self.policy_norm)})", f"(P:{to_str(self.P)})", f"(Q:{to_str(self.Q)})", f"(U:{to_str(self.puct)})", f"(Q+U:{to_str(self.Q + self.puct)})")

class Node:
    @property
    def all_children(self):
        return {**self.children, **self.pruned_children}

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else self.fpu

    def __init__(self, parent=None, P=0, fpu=0):
        # Values for tree representation
        self.parent = parent
        self.children = {}
        self.pruned_children = {} # Used to reconstruct root node after pruning
        self.image = None
        self.fpu = fpu

        # Values for MCTS
        self.N = 0
        self.W = 0
        self.P = P

        # For debugging purposes
        self.init_eval = 0.0
        self.init_policy = 0.0
        self.init_policy_norm = 0.0
        self.puct = 0.0
        self.action = None

    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    
    def reset(self):
        self.parent = None
        self.children = {}
        self.pruned_children = {}
        self.N = 0
        self.W = 0
        self.P = 0

    def find_best_variation(self, game):
        best_child = max(self.children.values(), key=lambda child: child.N)
        move = list(self.children.keys())[list(self.children.values()).index(best_child)]
        game.make_move(move)
        if best_child.is_leaf():
            return [move]
        else:
            return [move] + best_child.find_best_variation(game)

    def display_move_statistics(self, game: Game = None):
        print(f"Visits: {self.N}")
        print(f"Summed value: {self.W:.6f}")
        print(f"Evaluation: {self.init_eval:.6f}")
        if game is not None:
            tmp_game = game.clone()
            print(f"Best variation: {' -> '.join(self.find_best_variation(tmp_game))}")

        print("%-13s %-7s %-28s %-18s %-14s %-14s %-14s %-14s" %("Move", "Visits", "NN Output", "Policy", "Prior", "Avg. value", "UCB", "Q+U"))
        print("-" * 130)
        # Todo: display best variation
        debug_nodes = [DebugNode(move, child) for move, child in self.all_children.items()]
        sorted_children = sorted(debug_nodes, key=lambda x: x.N, reverse=True) # Sort by visit count
        for child in sorted_children:
            print(child)


def run_mcts(config, game: Game, network, root: Node = None, reuse_tree: bool = False, num_simulations: int = 100, num_sampling_moves: int = 30, add_noise: bool = True, pb_c_factor = None, **kwargs):
    """Run Monte Carlo Tree Search algorithm from the current game state.

    Args:
        game (Game): Current game state.
        sim_num (int, optional): Number of simulations to run. Defaults to 100.

    Returns:
        Move, Root: Return the move to play and the root node with an updated tree layout.
    """
    rng = np.random.default_rng()
    pb_c_base = config.pb_c_base
    pb_c_init = config.pb_c_init
    policy_temp = config.policy_temp
    softmax_temp = config.softmax_temp
    if pb_c_factor is None:
        pb_c_factor = config.pb_c_factor
    pb_c_factor_root = pb_c_factor[0]
    pb_c_factor_leaf = pb_c_factor[1]
    fpu_root = config.fpu_root
    fpu_leaf = config.fpu_leaf

    time_start = time()
    # Initialize the root node
    # Allows us to reuse the image of the previous root node but also allows us to reuse the tree or make clean start
    if root is None:
        root = Node()
        value = expand(root, game, network, policy_temp, fpu_root)
        root.init_eval = value
    else:
        if reuse_tree:
            num_simulations = num_simulations - root.N
        else:
            root.reset()
            value = expand(root, game, network, policy_temp, fpu_root)
            root.init_eval = value

    # Exploration noise used for full searches
    if add_noise:
        add_exploration_noise(config, root, rng)

    # Run the simulations
    for _ in range(num_simulations):
        # Select the best leaf node
        node = root    
        tmp_game = game.clone()    
        while node.is_leaf() is False:
            pb_c_fact = pb_c_factor_root if node.is_root() else pb_c_factor_leaf
            move, node = select_leaf(node, pb_c_base, pb_c_init, pb_c_fact)  # Select best with ucb
            tmp_game.make_move(move)

        # Check if leaf is a terminal state
        value, is_terminal = evaluate(tmp_game)
        # Expand if possible
        if not is_terminal:
            value = expand(node, tmp_game, network, policy_temp, fpu_leaf)

        # Store info for debugging
        node.init_eval = value

        # Backpropagate the value
        update(node, flip_value(value))
    
    time_end = time()
    # Print Root & it's children
    if kwargs.get("verbose_move") == 1:
        print(f"Time taken: {time_end - time_start:.2f}s")
        root.display_move_statistics(game=game)
    
    # return best_move, root
    temp = softmax_temp if game.history_len < num_sampling_moves else 0
    return select_move(root, rng, temp), root

def get_image(game: Game, node: Node):
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

def expand(node: Node, game: Game, network, policy_temp: float = 1.0, fpu: float = 0.0):
    """Evaluate the given node using the neural network.

    Args:
        node (Node): Node to evaluate.z
    """

    # Get the policy and value from the neural network
    value, policy_logits = predict_fn(network, get_image(game, node))
    value = value.numpy().item()
    policy_logits = policy_logits.numpy().flatten()

    moves = []
    actions = []
    policy = []
    to_play = game.to_play()
    for move in game.board.legal_moves: # Acces via generator for speedup
        move_uci = chess.Move.uci(move)
        action = map_w[move_uci] if to_play else map_b[move_uci]
        moves.append(move_uci)
        actions.append(action)
        policy.append(policy_logits[action])
   
    policy = np.array(policy)

    _max = np.max(policy)
    expsum = np.sum(np.exp(policy - _max))
    policy_norm = np.exp(policy - (_max + np.log(expsum)))
    if policy_temp > 0 and policy_temp != 1.0:
        policy_norm = policy_norm ** (1.0 / policy_temp)
        policy_norm = policy_norm / np.sum(policy_norm)
    assert np.sum(policy_norm) > 0.99 and np.sum(policy_norm) < 1.01, f"Policy norm sum is {np.sum(policy_norm)}"

    for move, action, p, p_norm in zip(moves, actions, policy, policy_norm):
        child = Node(parent=node, P=p_norm, fpu=fpu)
        child.init_policy = p
        child.init_policy_norm = p_norm
        child.action = action
        node.children[move] = child

    return value_to_01(value)


def evaluate(game: Game):
    if game.terminal_with_outcome():
        return value_to_01(game.terminal_value(game.to_play())), True
    else:
        return value_to_01(0.0), False


def value_to_01(value):
    return (value + 1) / 2


def flip_value(value):
    return 1 - value
    

def update(node: Node, value: float):
    node.N += 1
    node.W += value
    if not node.parent is None:
        update(node.parent, flip_value(value))

def select_leaf(node: Node, pb_c_base: float, pb_c_init: float, pb_c_factor: float):
    bestucb = -np.inf
    bestmove = None
    bestchild = None

    for move, child in node.children.items():
        if pb_c_factor > 0.0:
            puct = child.Q + UCB(child.P, child.N, node.N, pb_c_base, pb_c_init, pb_c_factor)
        else:
            puct = child.Q
        if puct > bestucb:
            bestucb = puct
            bestmove = move
            bestchild = child
        child.puct = puct # Store info for debugging
    return bestmove, bestchild

def UCB(cP: float, cN: int, pN: int, pb_c_base: float, pb_c_init: float, pb_c_factor: float):
    cpuct = (
        log((pN + pb_c_base + 1) / pb_c_base)
        + pb_c_init
    ) * pb_c_factor
    return cpuct * cP * (pN**0.5 / (cN + 1))

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
    noise = rng.gamma(config.root_dirichlet_alpha, 1.0, len(node.children))
    noise /= np.sum(noise)
    frac = config.root_exploration_fraction
    for n, child in zip(noise, node.children.values()):
        child.P = child.P * (1 - frac) + n * frac