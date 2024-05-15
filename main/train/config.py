from tensorflow.python.compiler.tensorrt.trt_convert import TrtPrecisionMode
import os
current_dir = os.path.dirname(__file__)

class TrainingConfig:
    def __init__(self):
        self.history_steps = 6 # 8 for AZ
        self.repetition_planes = 0 # 2 for AZ
        num_planes = (6 * 2 + self.repetition_planes) * self.history_steps + 5

        self.image_shape = (num_planes, 8, 8)
        self.num_actions = 1858 # 4672 for AZ

        # Self-play info
        self.playout_cap_random_p = 0.35 #0.25        
        self.num_mcts_sims = (80, 600)
        self.num_mcts_sampling_moves = 30
        self.pb_c_factor = (1.0, 1.0) # Used for full searches

        # MCTS constants
        self.num_parallel_reads = 16
        self.fpu_root = 1.0
        self.fpu_leaf = 0.3
        self.softmax_temp = 1.15 # more randomness in the beginning
        self.policy_temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25 # 2.5 -> Lc0
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # Model info
        self.conv_filters = 96 # 128
        self.num_residual_blocks = 10 # 8
        self.output_dims = (self.num_actions,)
        self.conv_kernel_initializer = None #"he_normal" # None reverts to TF default setting
        self.use_bias_on_outputs = True
        self.value_head_filters = 1 # 32
        self.value_head_dense = 256 # 128
        self.policy_head_filters = 2 # 128
        self.value_head_loss_weight = 1.0
        self.policy_head_loss_weight = 1.0       

        # Model compilation info
        self.l2_reg = 1e-4
        self.optimizer = "SGD"
        self.sgd_momentum = 0.9
        self.sgd_nesterov = False
        self.learning_rate_scheduler = "Static" # Cyclic / Static / Descreasing
        self.learning_rate = {
            "Cyclic": {
                "base_lr": 0.001,
                "max_lr": 0.1,
                "step_size": 4 * (self.epochs_per_cycle // self.batch_size),
                "mode": "triangular",
            },
            "Static": {
                "lr": 0.025 #3.125e-3 #0.025
            },
            "Decreasing": {
                "lr": {
                    0: 3.125e-3,
                    1000: 3.125e-4,
                    2000: 3.125e-5
                }
            }
        }

        # STS Testing info
        self.sts_test_interval = 10
        self.sts_num_agents = 6
        self.sts_time_limit = 1.5

        # Training info
        self.num_actors = 2
        self.num_cycles = 1000


        self.checkpoint_interval = 10 # Every n cycles 
        # Cycles are used for a cyclic style of learning, train games -> prepare data -> train -> repeat
        # Projects with more computer power do all these steps asynchonosly 
        self.games_per_cycle = 250 #50 # if avg length is 60, 100 games is 6000 samples * 0.25 (playout cap randomization) = 1500 samples * 6 (max trains per sample) = 9000 training samples
        self.batch_size = 512
        # 1 batch per epoch mimics AZ and other research when measuring in steps
        self.batches_per_epoch = 1 # 12

        # New implementation
        self.min_epochs_per_cycle = 10
        self.max_epochs_per_cycle = 20
        self.min_samples_for_training = self.batch_size * self.batches_per_epoch * self.min_epochs_per_cycle
        self.max_samples_per_step = self.batch_size * self.batches_per_epoch * self.max_epochs_per_cycle

        # Data storage info

        self.max_trains_per_sample = 2 # To prevent overfitting
        self.keep_positions_num = 50000 # When reached, oldest positions are deleted
        self.keep_records_num = 25 # When reached, oldest records are deleted

        # Define a name for model to diferentiante between different models on tensorboard
        bias = "_bias" if self.use_bias_on_outputs else ""
        self.model_name = f"b{self.num_residual_blocks}c{self.conv_filters}_SGD_bs{self.batch_size}{bias}_2"

        # Save directories
        self.self_play_positions_dir = f"{current_dir}/data/positions"
        self.training_records_dir = f"{current_dir}/data/records"
        self.sts_results_dir = f"{current_dir}/data/sts_results"
        self.positions_usage_stats = f"{current_dir}/data/positions_usage_counts.pkl"
        self.training_info_stats = f"{current_dir}/data/training_info.json"
        
        #self.tensorboard_log_dir = f"{current_dir}/logs/fit/{self.model_name}"
        self.tensorboard_log_dir = f"{current_dir}/logs/"
        self.test_suites_dir = f"{current_dir}/test_suites"

        self.keras_checkpoint_dir = f"{current_dir}/checkpoints/keras"
        self.trt_checkpoint_dir = f"{current_dir}/checkpoints/trt"


        # TRT and GPU info
        self.allow_gpu_growth = True
        self.trt_precision_mode = TrtPrecisionMode.INT8