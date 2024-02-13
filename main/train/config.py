from tensorflow.python.compiler.tensorrt.trt_convert import TrtPrecisionMode
import os
current_dir = os.path.dirname(__file__)

class TrainingConfig:
    def __init__(self):
        self.image_shape = (8, 8, 119)
        self.num_actions = 4672

        # Model info
        self.conv_filters = 48
        self.num_residual_blocks = 4
        self.output_dims = (self.num_actions,)

        # Model compilation info
        self.l2_reg = 1e-5
        self.optimizer = "Adam"
        self.adam_args = {
            "learning_rate": 0.003125
        }
        self.sgd_args = {
            "learning_rate": {}, # {step: learning_rate}
            "momentum": 0.9,
            "nesterov": True
        }

        # Self-play info
        self.max_game_length = 512
        self.num_mcts_sims = (100, 600)
        self.num_mcts_sampling_moves = 30
        self.pb_c_factor = (0.0, 2.0) # 0 on minimal search
        self.playout_cap_random_p = 0.25

        # MCTS constants
        self.temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # Training info
        self.num_actors = 4
        self.num_steps = 5
        self.games_per_step = 100 # if avg length is 100, 50 games is 5000 samples * 0.25 (playout cap randomization) = 1250 samples
        self.batch_size = 64
        self.min_samples_for_training = self.batch_size * 20 # 1280
        self.max_samples_per_step = self.min_samples_for_training * 5 # 6400
        self.keep_records_num = 10
        self.max_trains_per_sample = 8 # To prevent overfitting
        self.checkpoint_interval = 5 # Every 5 steps generate new model

        # Save directories
        self.self_play_positions_dir = f"{current_dir}/data/positions"
        self.tensorboard_log_dir = f"{current_dir}/data/logs/fit"
        self.training_records_dir = f"{current_dir}/data/records"
        self.keras_checkpoint_dir = f"{current_dir}/checkpoints/keras"
        self.trt_checkpoint_dir = f"{current_dir}/checkpoints/trt"
        self.positions_usage_stats = f"{current_dir}/data/positions_usage_counts.pkl"

        # TRT and GPU info
        self.allow_gpu_growth = True
        self.trt_precision_mode = TrtPrecisionMode.FP32