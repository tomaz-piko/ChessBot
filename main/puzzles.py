import chess
import pandas as pd
import random
import numpy as np


class Puzzle:
    @property
    def puzzle_id(self):
        return self._puzzle_id

    @property
    def rating(self):
        return self._rating

    @property
    def moves(self):
        return self._moves

    @property
    def moves_str(self):
        return " ".join(self._moves)

    @property
    def moves_played(self):
        return self._moves_played

    @property
    def moves_played_str(self):
        return " ".join(self._moves_played)

    @property
    def completion(self):
        self._calc_completion()
        return self._completion

    @property
    def status(self):
        return self._status

    def __init__(self, puzzle_id, rating, fen, moves):
        self._puzzle_id = puzzle_id
        self._rating = rating
        self._starting_fen = fen
        self._completion = 0
        self._status = "IN_PROGRESS"  # IN_PROGRESS, FAILED, COMPLETED
        self._moves = moves.split(" ")
        self._current_move = 0
        self._moves_played = []
        self.board = chess.Board(fen)
        self._next_move()

    def _next_move(self):
        self.board.push_uci(self._moves[self._current_move])
        self._moves_played.append(self._moves[self._current_move])
        self._current_move += 1

    def _update_status(self):
        # 0 = IN_PROGRESS, -1 = FAILED, 1 = COMPLETED
        if len(self._moves_played) == len(self._moves):
            if self._moves_played == self._moves:
                self._status = "COMPLETED"
                return
            else:
                self._status = "FAILED"
                return
        for i, move in enumerate(self._moves_played):
            if move != self._moves[i]:
                self._status = "FAILED"
                return

    def _calc_completion(self):
        counter = 0
        for i, move in enumerate(self._moves_played):
            if i % 2 == 0:
                continue
            if move == self._moves[i]:
                counter += 1
            else:
                break
        self._completion = round(counter / (len(self._moves)), 2) * 100

    def solve(self, bot):
        while self._status == "IN_PROGRESS":
            move = bot.get_move(self.board)
            self.move_uci(move)

    def restart(self):
        self._completion = 0
        self._status = "IN_PROGRESS"
        self._current_move = 0
        self._moves_played = []
        self.board = chess.Board(self._starting_fen)
        self._next_move()

    def legal_moves(self):
        moves = []
        for move in self.board.legal_moves:
            moves.append(chess.Move.uci(move))
        return moves

    def move_uci(self, move):
        self.board.push_uci(move)
        self._moves_played.append(move)
        self._current_move += 1
        self._update_status()
        if self._status == "IN_PROGRESS":
            self._next_move()


class PuzzleSet:
    history_columns = [
        "PuzzleId",
        "Rating",
        "Moves",
        "MovesPlayed",
        "Completion",
        "Status",
    ]
    history_columns_types = {
        "PuzzleId": "object",
        "Rating": "int32",
        "Moves": "object",
        "MovesPlayed": "object",
        "Completion": "float32",
        "Status": "object",
    }

    @property
    def history(self):
        return self._history

    def __init__(self, filename: str, range: tuple = None, sort: bool = True):
        self._puzzles = self._load_puzzles(filename, range, sort)
        self._history = pd.DataFrame(columns=self.history_columns)
        self._history = self._history.astype(self.history_columns_types)

    def _load_puzzles(self, filename, range=None, sorted=True):
        df = pd.read_csv(filename)
        if sorted:
            df = df.sort_values(by=["Rating"])
        if type(range) == int:
            _from = 0
            _to = range
        elif type(range) == tuple:
            _from = range[0] if range[0] else 0
            _to = range[1] if range[1] else len(df)
        else:
            _from = 0
            _to = len(df)

        puzzles = []
        for item in df[_from:_to].itertuples():
            puzzle = Puzzle(item.PuzzleId, item.Rating, item.FEN, item.Moves)
            puzzles.append(puzzle)
        return puzzles

    def get_all(self):
        return self._puzzles

    def get(self, idx):
        return self._puzzles[idx]

    def restart_all(self):
        for puzzle in self._puzzles:
            puzzle.restart()

    def restart(self, idx):
        self._puzzles[idx].restart()

    def get_statistics(self):
        stats = {}
        h = self._history
        stats["total_attempted"] = h.shape[0]
        stats["solved_count"] = h.loc[h["Status"] == "COMPLETED"].shape[0]
        stats["solved_percentage"] = round(stats["solved_count"] / h.shape[0] * 100, 2)
        stats["rating_avg"] = round(h["Rating"].mean(), 2)
        stats["solved_rating_avg"] = round(
            h.loc[h["Status"] == "COMPLETED"]["Rating"].mean(), 2
        )
        stats["best_success"] = h.loc[h["Status"] == "COMPLETED"]["Rating"].max()
        stats["worst_fail"] = h.loc[h["Status"] == "FAILED"]["Rating"].min()
        return stats

    def get_rating_scatter(self):
        h = self._history
        h.sort_values(by=["Status"], ascending=False, inplace=True)
        color = np.where(h["Status"] == "COMPLETED", "green", "red")
        x = np.arange(0, 1000)
        return h.reset_index().plot.scatter(
            x="index", y="Rating", use_index=True, color=color
        )

    def solve(self, bot, indices=None):
        if indices == None:
            indices = [*range(len(self._puzzles))]
        if type(indices) == int:
            indices = [indices]
        for ind in indices:
            pz = self._puzzles[ind]
            pz.solve(bot)
            history_row = pd.DataFrame(
                [
                    [
                        pz.puzzle_id,
                        pz.rating,
                        pz.moves_str,
                        pz.moves_played_str,
                        pz.completion,
                        pz.status,
                    ]
                ],
                columns=self.history_columns,
            )
            self._history = pd.concat([self._history, history_row], ignore_index=True)