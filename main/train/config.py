import os
current_dir = os.path.dirname(__file__)

class TrainingConfig:
    def __init__(self):
        self.history_steps = 6 # 8 for AZ
        self.history_perspective_flip = False # True: t0 -> us as p1, t1 -> us as p2
        self.repetition_planes = 2 # 2 for AZ
        num_planes = (6 * 2 + self.repetition_planes) * self.history_steps + 5

        self.image_shape = (num_planes, 8, 8)
        self.num_actions = 1858 # 4672 for AZ

        # Self-play info
        self.playout_cap_random_p = 1 #0.35 #0.25        
        self.num_mcts_sims = (600, 600) #(100, 800)
        self.num_mcts_sampling_moves = 30
        self.pb_c_factor = (1.0, 1.0) # Used for full searches
        self.resignation_move_limit = 50 # Can not resign before this move
        self.resignation_threshold = 0.08 # If best move is below this threshold, resign
        self.resignable_games_perc = 0.0 # 80% of games are resignable, the rest are played out even possibly until move limit 512
        self.wdl_termination_move_limit = 100 # 0 -> Disabled
        self.discourage_draws_value = 0.0 # 0.0 -> Disabled

        # MCTS constants
        self.num_parallel_reads = 16
        self.fpu_root = 1.0
        self.fpu_leaf = 0.3
        self.softmax_temp = 1.15 # more randomness in the beginning
        self.policy_temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 2.5
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # Model info
        self.conv_filters = 64 # 96
        self.num_residual_blocks = 8 # 6
        self.output_dims = (self.num_actions,)
        self.conv_kernel_initializer = None #"he_normal" # None reverts to TF default setting
        self.use_bias_on_outputs = True
        self.value_head_filters = 32 # 32
        self.value_head_dense = 128
        self.policy_head_filters = 73 # 73
        self.value_head_loss_weight = 1.0
        self.policy_head_loss_weight = 1.0       

        # Model compilation info
        self.l2_reg = 1e-4
        self.optimizer = "SGD"
        self.sgd_momentum = 0.9
        self.sgd_nesterov = False
        self.learning_rate = 0.025

        # Training info
        self.num_actors = 2
        self.checkpoint_interval = 25 # Every n cycles 
        self.games_per_cycle = 50 
        self.batch_size = 512
        self.sampling_ratio = 0.80 # 0.0 - 1.0 -> Percentage of samples used for training

        # STS Testing info
        self.sts_test_interval = 5 # Every n cycles
        self.sts_num_agents = 8
        self.sts_time_limit = 1.25

        # OBSOLETE
        # parameters removed with the introduction of sampling_ratio
        #self.batches_per_epoch = 1 # Should always be 1
        #self.epochs_per_cycle = 80 # OBSOLETE -> Need to be calculated based on batches and sampling ratio
        #self.max_trains_per_sample = 1 # Also obsolete
        #self.keep_positions_num = 100000 # When reached, oldest positions are deleted
        #self.min_samples_per_cycle = self.batch_size * self.batches_per_epoch * self.epochs_per_cycle
        #self.keep_records_num = 3 # When reached, oldest records are deleted
        
        # Define a name for model to diferentiante between different models on tensorboard
        bias = "_bias" if self.use_bias_on_outputs else ""
        self.model_name = f"b{self.num_residual_blocks}c{self.conv_filters}_SGD_bs{self.batch_size}{bias}_3"

        # Save directories
        self.self_play_positions_dir = f"{current_dir}/data/positions"
        self.training_records_dir = f"{current_dir}/data/records"
        self.sts_results_dir = f"{current_dir}/data/sts_results"
        self.positions_usage_stats = f"{current_dir}/data/positions_usage_counts.pkl"
        self.training_info_stats = f"{current_dir}/data/training_info.json"        
        self.tensorboard_log_dir = f"{current_dir}/logs/"
        self.test_suites_dir = f"{current_dir}/test_suites"
        self.keras_checkpoint_dir = f"{current_dir}/checkpoints/keras"
        self.trt_checkpoint_dir = f"{current_dir}/checkpoints/trt"
        self.tmp_trt_checkpoint_dir = f"{current_dir}/checkpoints/tmp_trt"

        self.syzygy_tb_dir = "/home/tomaz/syzygy"
        self.self_play_positions_backup_dir = "/home/tomaz/positions_backup"

        # TRT and GPU info
        self.allow_gpu_growth = True
        self.trt_precision_mode = 'INT8'