from tensorflow.python.compiler.tensorrt.trt_convert import TrtPrecisionMode
import os
current_dir = os.path.dirname(__file__)

class TrainingConfig:
    def __init__(self):
        self.image_shape = (110, 8, 8)
        self.num_actions = 4672

        # Model info
        self.conv_filters = 64
        self.num_residual_blocks = 5
        self.output_dims = (self.num_actions,)
        self.conv_kernel_initializer = None # None reverts to TF default setting
        self.use_bias_on_outputs = False
        self.value_head_filters = 32
        self.value_head_dense = 128
        self.value_head_loss_weight = 1.0
        self.policy_head_loss_weight = 1.0       

        # Model compilation info
        self.l2_reg = 1e-4
        self.optimizer = "SGD"
        self.sgd_momentum = 0.9
        self.sgd_nesterov = False
        self.learning_rate_scheduler = "Cyclic" # Static / Descreasing
        self.learning_rate = {
            "Cyclic": {
                "base_lr": 3.125e-5,
                "max_lr": 3.125e-3,
                "step_size": 8,
                "mode": "exp_range",
                "gamma": 0.99994
            },
            "Static": {
                "lr": 0.001
            },
            "Decreasing": {
                "lr": {
                    0: 0.1,
                    1000: 0.01,
                    2000: 0.001
                }
            }
        }

        # Self-play info
        self.max_game_length = 512
        self.num_mcts_sims = (100, 600)
        self.num_mcts_sampling_moves = 30
        # pb_c_factor = (root_pb_c, leaf_pb_c)
        self.pb_c_factor_min = (1.0, 1.0) # Used for minimal searches
        self.pb_c_factor = (2.5, 1.5) # Used for full searches
        self.fpu_root = 1.0
        self.fpu_leaf = 0.3
        self.playout_cap_random_p = 0.25

        # MCTS constants
        self.softmax_temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.policy_temp = 1.15

        # Training info
        self.num_actors = 2
        # Training can be paused & continued. Set for preffered steps or leave high enough it doesnt stop until canceled
        self.num_cycles = 1000
        self.checkpoint_interval = 10 # Every n cycles 
        # Cycles are used for a cyclic style of learning, train games -> prepare data -> train -> repeat
        # Projects with more computer power do all these steps asynchonosly 
        self.games_per_cycle = 100 #50 # if avg length is 60, 100 games is 6000 samples * 0.25 (playout cap randomization) = 1500 samples * 6 (max trains per sample) = 9000 training samples
        self.epochs_per_cycle = 200
        self.batch_size = 64
        # 1 batch per epoch mimics AZ and other research when measuring in steps
        self.batches_per_epoch = 1 # 12

        self.min_samples_for_training = self.batch_size * self.batches_per_epoch
        self.max_samples_per_step = self.min_samples_for_training * self.epochs_per_cycle

        self.keep_records_num = 5
        self.max_trains_per_sample = 8 # To prevent overfitting
        self.delete_sample_after_num_trains = 6 # Delete sample after 6 trains

        self.warmup_steps = 100
        self.warmup_learning_rate = 1.0e-4
        self.warmup_samples_per_step = self.min_samples_for_training
        self.warmup_redo_records = True
        self.use_seed_games = True

        # Save directories
        self.seed_positions_dir = f"{current_dir}/data/seed_positions"
        self.seed_records_dir = f"{current_dir}/data/seed_records"
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
