import numpy as np
from gameimage import board_to_image, update_image
from game import Game
from actionspace import map_w, map_b
from tf_funcs import predict_fn
from math import log
from time import time

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
    def Q(self):
        return self.W / self.N if self.N > 0 else self.fpu

    def __init__(self, fpu=0.0):
        # Values for tree representation
        self.children = {}
        self.image = None
        self.fpu = fpu
        self.virtual_losses = 0
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
        self.children[move] = node

    def is_leaf(self):
        return len(self.children) == 0
    
    def reset(self):
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
    

class MCTS:
    def __init__(self, config, trt_func, engine_play=False):
        self.root = Node()
        self.network = Network(trt_func, batch_size=config.num_parallel_reads)
        self.rng = np.random.default_rng()
        self.num_parallel_reads = config.num_parallel_reads
        self.engine_play = engine_play

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
    
    def __call__(self, game, num_simulations, time_limit=0.0, minimal_exploration=False, return_statistics=False, **kwargs):
        time_start = time()

        if minimal_exploration:
            fpu_root, fpu_leaf = 1.0, 0.0
            pb_c_factor_root, pb_c_factor_leaf = 1.0, 1.0
            policy_temp = 0.0
        else:
            fpu_root, fpu_leaf = self.fpu_root, self.fpu_leaf
            pb_c_factor_root, pb_c_factor_leaf = self.pb_c_factor_root, self.pb_c_factor_leaf
            policy_temp = self.policy_temp

        if self.root.is_leaf():
            if self.root.image is None:
                self.root.image = board_to_image(game.board)
            values, policy_logits = self.network(
                    images=np.array([self.root.image], dtype=np.float32),
                    fill_buffer=True
                )
            self.expand_node(self.root, game, fpu_root)
            self.evaluate_node(self.root, policy_logits[0], policy_temp)
            self.root.init_eval = value_to_01(values[0].item())

        if not minimal_exploration and not self.engine_play:
            self.add_exploration_noise(self.root)

        while self.root.N < num_simulations:
            nodes_to_eval = []
            search_paths = []
            nodes_to_find = self.num_parallel_reads if self.root.N + self.num_parallel_reads <= num_simulations else num_simulations - self.root.N
            while len(nodes_to_eval) < nodes_to_find:
                node = self.root
                search_path = [node]
                tmp_game = game.clone()
                
                # Traverse tree and find a leaf node
                while not node.is_leaf():
                    pb_c_factor = pb_c_factor_root if len(search_path) == 1 else pb_c_factor_leaf
                    move, node = self.select(node, pb_c_factor)
                    search_path.append(node)
                    tmp_game.make_move(move)

                # Check if game ends here
                value, terminal = evaluate_game(tmp_game)
                # If the game is terminal, backup the value and continue with the next simulation
                if terminal:
                    self.backup(search_path, flip_value(value))
                    node.init_eval = flip_value(value)
                    if self.root.N == num_simulations:
                        break
                    continue

                # Expansion
                parent = search_path[-2]
                node.image = update_image(tmp_game.board, parent.image.copy())
                self.expand_node(node, tmp_game, fpu_leaf)
                self.add_virtual_loss(search_path)

                # Save the nodes to evaluate and the search paths
                nodes_to_eval.append(node)
                search_paths.append(search_path)
            
            if not len(nodes_to_eval) > 0:
                continue

            values, policy_logits = self.network(
                    images=np.array([node.image for node in nodes_to_eval], dtype=np.float32), 
                    fill_buffer=not nodes_to_find == self.num_parallel_reads
                )

            for i, (node, search_path) in enumerate(zip(nodes_to_eval, search_paths)):
                self.evaluate_node(node, policy_logits[i], policy_temp)
                self.remove_virtual_loss(search_path)
                self.backup(search_path, flip_value(value_to_01(values[i].item())))
                node.init_eval = flip_value(value_to_01(values[i].item()))

        end_time = time()
        if kwargs.get("verbose_move"):
            print(f"Time for MCTS: {end_time - time_start}")
            self.root.display_move_statistics(game)
        if not self.engine_play:
            move = self.select_move(self.root, self.softmax_temp if game.history_len < self.num_sampling_moves else 0.0)
            child_visits = self.calculate_search_statistics() if return_statistics else None
            return move, child_visits
        else:
            return self.select_move(self.root, 0.0), None
    
    def expand_node(self, node, game, fpu):
        node.to_play = game.to_play()
        for move in game.board.legal_moves:
            node[move.uci()] = Node(fpu=fpu)

    def evaluate_node(self, node, policy_logits, policy_temp):
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

    def reset(self):
        self.root = Node()

    def select(self, node, pb_c_factor):
        bestucb = -np.inf
        bestmove = None
        bestchild = None
        
        for move, child in node.children.items():
            ucb = child.Q + self.UCB(child.P, child.N, node.N, pb_c_factor)
            if ucb > bestucb:
                bestucb = ucb
                bestmove = move
                bestchild = child
            child.puct = ucb

        return bestmove, bestchild

    def UCB(self, cP, cN, pN, pb_c_factor):
        cpuct = (
            log((pN + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        ) * pb_c_factor
        return cpuct * cP * pN**0.5 / (cN + 1)

    def select_move(self, node, softmax_temp = 0.0):
        moves = list(node.children.keys())
        visit_counts = [child.N for child in node.children.values()]

        if softmax_temp == 0.0:
            # Greedy selection (select the move with the highest visit count) 
            # If more moves have the same visit count, choose one randomly
            return moves[self.rng.choice(np.flatnonzero(visit_counts == np.max(visit_counts)))]
        else:
            # Use the visit counts as a probability distribution to select a move
            pi = np.array(visit_counts) ** (1 / softmax_temp)
            pi /= np.sum(pi)
            return moves[np.where(self.rng.multinomial(1, pi) == 1)[0][0]]
    
    def add_virtual_loss(self, search_path):
        for node in search_path[::-1]:
            node.virtual_losses += 1
            node.W -= 1

    def remove_virtual_loss(self, search_path):
        for node in search_path[::-1]:
            node.virtual_losses -= 1
            node.W += 1

    def backup(self, search_path, value):
        for node in search_path[::-1]:
            node.N += 1
            node.W += value
            value = flip_value(value)

    def add_exploration_noise(self, node):
        noise = self.rng.gamma(self.dirichlet_alpha, 1, len(node.children))
        noise /= noise.sum()
        for n, child in zip(noise, node.children.values()):
            child.P = child.P * (1 - self.exploration_fraction) + n * self.exploration_fraction

    def calculate_search_statistics(self):
        child_visits = np.zeros(4672, dtype=np.float32)
        sum_visits = np.sum([child.N for child in self.root.children.values()])
        for uci_move, child in self.root.children.items():
            child_visits[map_w[uci_move] if self.root.to_play else map_b[uci_move]] = child.N / sum_visits
        return child_visits


def evaluate_game(game):
    if game.terminal_with_outcome():
        return value_to_01(game.terminal_value(game.to_play())), True
    else:
        return value_to_01(0.0), False


def value_to_01(value):
    return (value + 1) / 2


def flip_value(value):
    return 1-value