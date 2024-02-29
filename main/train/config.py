from tensorflow.python.compiler.tensorrt.trt_convert import TrtPrecisionMode
import os
current_dir = os.path.dirname(__file__)

class TrainingConfig:
    def __init__(self):
        self.image_shape = (8, 8, 119)
        self.num_actions = 4672

        # Model info
        self.conv_filters = 64
        self.num_residual_blocks = 6
        self.output_dims = (self.num_actions,)

        # Model compilation info
        self.l2_reg = 1e-4
        self.optimizer = "Adam"
        self.adam_args = {
            "learning_rate": 0.003125
        }
        self.sgd_args = {
            "learning_rate": {0: 3.125e-3, 350: 3.125e-4, 850: 3.125e-5}, # {epoch: learning_rate} epoch != step
            "momentum": 0.9,
            "nesterov": False
        }
        self.model_mixed_precision = False

        # Self-play info
        self.max_game_length = 512
        self.num_mcts_sims = (100, 600)
        self.num_mcts_sampling_moves = 30
        self.pb_c_factor = (0.0, 1.5) # 0 on minimal search
        self.playout_cap_random_p = 0.25

        # MCTS constants
        self.temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # Training info
        self.num_actors = 2
        self.num_steps = 100
        self.games_per_step = 100 #50 # if avg length is 60, 100 games is 6000 samples * 0.25 (playout cap randomization) = 1500 samples * 6 (max trains per sample) = 9000 training samples
        self.batch_size = 64
        self.min_samples_for_training = self.batch_size * 12 # 768
        self.max_samples_per_step = self.min_samples_for_training * 10 # 7680
        self.keep_records_num = 5
        self.max_trains_per_sample = 6 # To prevent overfitting
        self.delete_sample_after_num_trains = 4 # Delete sample after 4 trains
        self.checkpoint_interval = 2 # Every 2 steps generate new model

        # Save directories
        self.self_play_positions_dir = f"{current_dir}/data/positions"
        self.tensorboard_log_dir = f"{current_dir}/data/logs/fit"
        self.training_records_dir = f"{current_dir}/data/records"
        self.keras_checkpoint_dir = f"{current_dir}/checkpoints/keras"
        self.trt_checkpoint_dir = f"{current_dir}/checkpoints/trt"
        self.positions_usage_stats = f"{current_dir}/data/positions_usage_counts.pkl"
        self.training_info_stats = f"{current_dir}/data/training_info.json"

        # TRT and GPU info
        self.allow_gpu_growth = True
        self.trt_precision_mode = TrtPrecisionMode.FP32