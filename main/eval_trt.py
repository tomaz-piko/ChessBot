from config import Config, minimal_model_config
from game import Game
from mcts import run_mcts
import numpy as np
import os
from puzzles import Puzzle
import pandas as pd
import time
import tensorflow as tf
import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Load puzzles from csv
def load_puzzles():
    df = pd.read_csv("puzzles/sample_1000.csv")
    df.sort_values(by=["Rating"], inplace=True)
    puzzles = []
    for item in df.itertuples():
        puzzle = Puzzle(item.PuzzleId, item.Rating, item.FEN, item.Moves)
        puzzles.append(puzzle)
    return puzzles

class RandomBot:
    def __init__(self, config):
        self.config = config

    def get_move(self, board):
        game = Game(self.config, board=board)
        return np.random.choice(game.legal_moves())

class Bot:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def get_move(self, board):
        game = Game(self.config, board=board)
        move, _ = run_mcts(game, self.model, self.config)
        return move


def solve_puzzles(puzzles, results, config):    
    model = tf.saved_model.load("models/2023-12-22_20:19:10/saved_model")
    predict = model.signatures['serving_default']

    bot = Bot(predict, config)
    for puzzle in puzzles:
        t1 = time.time()
        puzzle.solve(bot)
        t2 = time.time()
        print(f"Solved puzzle {puzzle.puzzle_id} in {t2-t1} seconds.")
        results[puzzle.puzzle_id] = puzzle


if __name__ == "__main__":
    _from = 0
    _to = 50
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    config = minimal_model_config()
    #config.num_mcts_sims = 500
    config.num_sampling_moves = 0
    puzzles = load_puzzles()[_from:_to]
    results = {}

    conversion_params = trt.TrtConversionParams(
         precision_mode=trt.TrtPrecisionMode.FP32,
    )       
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir="checkpoints/trt/saved_model",
        conversion_params=conversion_params,
        use_calibration=False,
    )
    trt_func = converter.convert()

    bot = Bot(trt_func, config)
    for puzzle in puzzles:
        puzzle.solve(bot)
        results[puzzle.puzzle_id] = puzzle

    successes = np.zeros(len(puzzles))
    ratings = np.zeros(len(puzzles))
    for i, (id, puzzle) in enumerate(results.items()):
        successes[i] = 1 if puzzle.status == "COMPLETED" else 0
        ratings[i] = puzzle.rating if puzzle.status == "COMPLETED" else 0
    print("PikoBot results.")
    print(f"Success count: {np.sum(successes)}")
    print(f"Fails count: {len(puzzles) - np.sum(successes)}")

    if np.sum(successes) > 0:
        idcs = np.nonzero(ratings)
        ratings = ratings[idcs]
        print(f"Average rating: {np.mean(ratings)}")
        print(f"Lowest solved rating: {np.min(ratings)}")
        print(f"Highest solved rating: {np.max(ratings)}")

    bot = RandomBot(config=config)
    puzzles = load_puzzles()[_from:_to]
    results = {}
    for puzzle in puzzles:
        puzzle.solve(bot)
        results[puzzle.puzzle_id] = puzzle
    successes = np.zeros(len(puzzles))
    ratings = np.zeros(len(puzzles))
    for i, (id, puzzle) in enumerate(results.items()):
        successes[i] = 1 if puzzle.status == "COMPLETED" else 0
        ratings[i] = puzzle.rating if puzzle.status == "COMPLETED" else 0

    print("RandomBot results.")
    print(f"Success count: {np.sum(successes)}")
    print(f"Fails count: {len(puzzles) - np.sum(successes)}")

    if np.sum(successes) > 0:
        idcs = np.nonzero(ratings)
        ratings = ratings[idcs]
        print(f"Average rating: {np.mean(ratings)}")
        print(f"Lowest solved rating: {np.min(ratings)}")
        print(f"Highest solved rating: {np.max(ratings)}")
