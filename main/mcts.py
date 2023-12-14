import main.config as config
import numpy as np
import math
import chess
from main.game import Game
import main.actionspace as asp
import tensorflow as tf


class Node:
    @property
    def children(self) -> dict:
        """Returns the children of the node.

        Returns:
            dict: Dictionary of children. Key is the move string, value is Node.
        """
        return self._children

    """Node class for MCTS.
    """

    def __init__(self, priority: float = 0):
        # Tree layout specifics
        self._children = {}  # Dict of children. Key is the move string, value is Node

        # Game specifics
        self.to_play = None  # Current player

        # MCTS specifics
        self.priority = priority  # Priority of the node
        self.value_sum = 0
        self.visits_count = 0

    def add_child(self, move: str, child: "Node"):
        """Add a child to the current node. No need to

        Args:
            move (str): Move string.
            child (Node): New child node.
        """
        self.children[move] = child

    def expanded(self) -> bool:
        """Checks if the node is expanded. Node is expanded if it has children.

        Returns:
            bool: True if node is expanded, False otherwise.
        """
        return len(self.children) > 0

    def value(self) -> float:
        """Returns the calculated value of the node.

        Returns:
            float: Value of the node. 0 if node has not been visited.
        """
        return 0 if self.visits_count == 0 else self.value_sum / self.visits_count


def run_mcts(game: "Game", network: "MyModel", num_simulations: int = 100):
    """Run Monte Carlo Tree Search algorithm from the current game state.

    Args:
        game (Game): Current game state.
        sim_num (int, optional): Number of simulations to run. Defaults to 100.

    Returns:
        Move, Root: Return the move to play and the root node with an updated tree layout.
    """
    root = Node()  # Priority default is 0
    _evaluate(root, game, network)
    _add_exploration_noise(root)

    for _ in range(num_simulations):
        node = root
        tmp_game = game.clone()
        search_path = [node]

        while node.expanded():
            move, node = _select_child(node)
            tmp_game.make_move(move)
            search_path.append(node)

        value = _evaluate(node, tmp_game, network)
        _backpropagate(search_path, value, tmp_game.to_play())
    return _select_move(root, game), root


def _evaluate(node: Node, game: Game, network) -> float:
    """Evaluates the current node.
        Expands the node and sets priorities for children.

    Args:
        node (Node): Node to evaluate.
        game (Game): Current game state of the node.

    Returns:
        float: Evaluation value of the node.
    """
    policy_logits, value = network.predict(game.make_image(-1))
    node.to_play = game.to_play()
    policy = {}
    for move in game.legal_moves():
        _, action = asp.uci_to_action(move)
        policy[move] = np.exp(policy_logits[action])

    policy_sum = sum(policy.values())
    for move, p in policy.items():
        node.add_child(move, Node(priority=p / policy_sum))

    return value


def _select_child(node: Node) -> (str, Node):
    """Selects a child node to explore next.

    Args:
        node (Node): Node to select child from.

    Returns:
        str, Node: Move string and child node.
    """
    _, action, child = max(
        (_ucb_score(node, child), action, child)
        for action, child in node.children.items()
    )
    return action, child


def _backpropagate(search_path: list[Node], value: float, to_play: bool):
    """Backpropagates the value of the node to the root.

    Args:
        search_path (list[Node]): Path from the root to the node.
        value (float): Value of the node that was evaluated and expanded in current loop.
        to_play (bool): _description_
    """
    for node in search_path:
        node.value_sum += value if node.to_play == chess.WHITE else (1 - value)
        node.visits_count += 1


def _select_move(node: Node, game: Game) -> str:
    """Select the next move to play based on the visit counts of the children.

    Args:
        node (Node): Node to select next move from.
        game (Game): Current game state.
        config (Config): Config object.

    Returns:
        str: Move string.
    """
    visit_counts = [(child.visits_count, move) for move, child in node.children.items()]
    if game.history_len < 30:
        _, action = _softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)
    return action


def _softmax_sample(visit_counts: list) -> (int, str):
    """Creates a softmax distribution from the visit counts and uses it as a probability distribution to select the next move.

    Args:
        visit_counts (list): List of (visits_count, move).

    Returns:
        int: Visits count of the selected move.
        str: Move string.
    """

    def softmax(x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    p = softmax(np.asarray(visit_counts).T[0].astype(int))
    v = np.random.choice(len(visit_counts), p=p)
    return visit_counts[v][0], visit_counts[v][1]


def _ucb_score(parent: Node, child: Node) -> float:
    """Calculates the UCB score of the child node.

    Args:
        parent (Node): Parent node of child.
        child (Node): Node to calculate UCB score for.

    Returns:
        float: Calculated UCB score of the child node.
    """
    pb_c = (
        math.log((parent.visits_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visits_count) / (child.visits_count + 1)
    prior_score = pb_c * child.priority
    value_score = child.value()
    return prior_score + value_score


def _add_exploration_noise(node: Node) -> None:
    """Add dirichlet noise to the root node to encourage exploration on each iteration.

    Args:
        node (Node): Current root node.
    """
    moves = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(moves))
    frac = config.root_exploration_fraction
    for a, n in zip(moves, noise):
        node.children[a].priority = node.children[a].priority * (1 - frac) + n * frac
