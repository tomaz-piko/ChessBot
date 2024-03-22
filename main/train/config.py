from tensorflow.python.compiler.tensorrt.trt_convert import TrtPrecisionMode
import os
current_dir = os.path.dirname(__file__)

class TrainingConfig:
    def __init__(self):
        self.image_shape = (109, 8, 8)
        self.num_actions = 4672

        # Model info
        self.conv_filters = 96
        self.num_residual_blocks = 6
        self.output_dims = (self.num_actions,)
        self.conv_kernel_initializer = None #"he_normal" # None reverts to TF default setting
        self.use_bias_on_outputs = True
        self.value_head_filters = 32
        self.value_head_dense = 128
        self.value_head_loss_weight = 0.5
        self.policy_head_loss_weight = 1.0       

        # Model compilation info
        self.l2_reg = 1e-4
        self.optimizer = "SGD"
        self.sgd_momentum = 0.9
        self.sgd_nesterov = False
        self.learning_rate_scheduler = "Static" # Cyclic / Static / Descreasing
        self.learning_rate = {
            "Cyclic": {
                "base_lr": 3.125e-5,
                "max_lr": 3.125e-4,
                "step_size": 100,
                "mode": "exp_range",
                "gamma": 0.99994
            },
            "Static": {
                "lr": 3.125e-3
            },
            "Decreasing": {
                "lr": {
                    0: 3.125e-3,
                    1000: 3.125e-4,
                    2000: 3.125e-5
                }
            }
        }

        # Self-play info
        self.max_game_length = 512
        self.num_mcts_sims = (100, 800)
        self.num_mcts_sampling_moves = 30
        # pb_c_factor = (root_pb_c, leaf_pb_c)
        self.pb_c_factor_min = (1.0, 1.0) # Used for minimal searches
        self.pb_c_factor = (2.0, 2.0) # Used for full searches
        self.playout_cap_random_p = 0.25

        # MCTS constants
        self.num_parallel_reads = 16
        self.fpu_root = 1.0
        self.fpu_leaf = 0.3
        self.softmax_temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 2.5
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.policy_temp = 1.125

        # Training info
        self.num_actors = 2
        # Training can be paused & continued. Set for preffered steps or leave high enough it doesnt stop until canceled
        self.num_cycles = 1000
        self.checkpoint_interval = 50 # Every n cycles 
        # Cycles are used for a cyclic style of learning, train games -> prepare data -> train -> repeat
        # Projects with more computer power do all these steps asynchonosly 
        self.games_per_cycle = 100 #50 # if avg length is 60, 100 games is 6000 samples * 0.25 (playout cap randomization) = 1500 samples * 6 (max trains per sample) = 9000 training samples
        self.epochs_per_cycle = 25 # 64 * 25 = 1600 samples per cycle at max
        self.batch_size = 64
        # 1 batch per epoch mimics AZ and other research when measuring in steps
        self.batches_per_epoch = 2 # 12

        self.min_samples_for_training = self.batch_size * self.batches_per_epoch
        self.max_samples_per_step = self.min_samples_for_training * self.epochs_per_cycle

        self.max_trains_per_sample = 2 # To prevent overfitting
        self.keep_positions_num = 50000 # When reached, oldest positions are deleted
        self.keep_records_num = 10 # When reached, oldest records are deleted

        # Define a name for model to diferentiante between different models on tensorboard
        lr = self.learning_rate[self.learning_rate_scheduler]
        bias = "_bias" if self.use_bias_on_outputs else ""
        if self.learning_rate_scheduler == "Cyclic":
            self.model_name = f"b{self.num_residual_blocks}c{self.conv_filters}_SGD_Clr{lr['base_lr']}-{lr['max_lr']}_s{lr['step_size']}_bs{self.batch_size}{bias}"
        elif self.learning_rate_scheduler == "Static":
            self.model_name = f"b{self.num_residual_blocks}c{self.conv_filters}_SGD_Static{lr['lr']}_bs{self.batch_size}{bias}"

        # Save directories
        self.seed_positions_dir = f"{current_dir}/data/seed_positions"
        self.seed_records_dir = f"{current_dir}/data/seed_records"
        self.self_play_positions_dir = f"{current_dir}/data/positions"
        self.tensorboard_log_dir = f"{current_dir}/data/logs/fit/{self.model_name}"
        self.training_records_dir = f"{current_dir}/data/records"
        self.keras_checkpoint_dir = f"{current_dir}/checkpoints/keras"
        self.trt_checkpoint_dir = f"{current_dir}/checkpoints/trt"
        self.positions_usage_stats = f"{current_dir}/data/positions_usage_counts.pkl"
        self.training_info_stats = f"{current_dir}/data/training_info.json"

        # TRT and GPU info
        self.allow_gpu_growth = True
        self.trt_precision_mode = TrtPrecisionMode.INT8