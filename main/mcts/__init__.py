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

class Node(object):
    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else self.fpu

    def __init__(self, fpu=0):
        # Values for tree representation
        self.parent = None
        self.children = {}
        self.image = None
        self.fpu = fpu
        self.to_play = None

        # Values for MCTS
        self.N = 0
        self.W = 0.0
        self.P = 0.0

        # For debugging purposes
        self.init_eval = 0.0
        self.init_policy = 0.0
        self.init_policy_norm = 0.0
        self.puct = 0.0
        self.action = None

    def __getitem__(self, move: str):
        return self.children[move]
    
    def __setitem__(self, move: str, node):
        node.parent = self
        self.children[move] = node

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
        debug_nodes = [DebugNode(move, child) for move, child in self.children.items()]
        sorted_children = sorted(debug_nodes, key=lambda x: x.N, reverse=True) # Sort by visit count
        for child in sorted_children:
            print(child)

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
    time_start = time()

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

    network = Network(trt_func, config.num_parallel_reads)
    rng = np.random.default_rng()

    if root is None:
        root = Node()
    elif not reuse_tree:
        root.reset()

    if root.is_leaf():
        if root.image is None:
            root.image = board_to_image(game.board)
        values, policy_logits = network(
                    images=np.array([root.image], dtype=np.float32),
                    fill_buffer=True
                )
        expand_node(root, game, fpu_root)
        evaluate_node(root, policy_logits[0], policy_temp)
        root.init_eval = value_to_01(values[0].item())

    if not minimal_exploration:
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
                node.init_eval = flip_value(value)
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

        values, policy_logits = network(
                images=np.array([node.image for node in nodes_to_eval], dtype=np.float32), 
                fill_buffer=not nodes_to_find == config.num_parallel_reads
            )

        for i, (node) in enumerate(nodes_to_eval):
            evaluate_node(node, policy_logits[i], policy_temp)
            remove_vloss(node)
            update(node, flip_value(value_to_01(values[i].item())))
            node.init_eval = flip_value(value_to_01(values[i].item()))

    end_time = time()
    if kwargs.get("verbose_move"):
        print(f"Time for MCTS: {end_time - time_start}")
        root.display_move_statistics(game)
    
    move = select_move(root, rng, config.softmax_temp if game.history_len < config.num_mcts_sampling_moves else 0.0)
    if kwargs.get("return_statistics"):
        child_visits = calculate_search_statistics(root)
        return move, root, child_visits
    return move, root
    
def expand_node(node: Node, game: Game, fpu: float):
    node.to_play = game.to_play()
    for move in game.board.legal_moves:
        node[move.uci()] = Node(fpu)


def evaluate_node(node: Node, policy_logits, policy_temp):
    # Extract only the policy logits for the legal moves
    actions = [map_w[move] if node.to_play else map_b[move] for move in node.children.keys()]
    policy_logits = policy_logits[actions]

    # Normalize the policy logits
    _max = policy_logits.max()
    expsum = np.sum(np.exp(policy_logits - _max))
    policy_logits_norm = np.exp(policy_logits - (_max + np.log(expsum)))
    if policy_temp > 0.0 and policy_temp != 1.0:
        policy_logits_norm = policy_logits_norm ** (1.0 / policy_temp)
        policy_logits_norm /= np.sum(policy_logits_norm)
    
    # Assign the policy logits to the children as priors
    for p, p_norm, child in zip(policy_logits, policy_logits_norm, node.children.values()):
        child.P = p_norm
        child.init_policy = p
        child.init_policy_norm = p_norm


def evaluate_game(game: Game):
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


def add_vloss(node: Node):
    node.W -= 1
    if not node.parent is None:
        add_vloss(node.parent)


def remove_vloss(node: Node):
    node.W += 1
    if not node.parent is None:
        remove_vloss(node.parent)


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

def calculate_search_statistics(root: Node):
    child_visits = np.zeros(4672, dtype=np.float32)
    sum_visits = np.sum([child.N for child in root.children.values()])
    for uci_move, child in root.children.items():
        child_visits[map_w[uci_move] if root.to_play else map_b[uci_move]] = child.N / sum_visits
    return child_visits