import os
current_dir = os.path.dirname(__file__)

class TrainingConfig:
    def __init__(self):
        self.history_steps = 6 # 8 for AZ
        self.history_perspective_flip = True # True: t0 -> us as p1, t1 -> us as p2
        self.repetition_planes = 2 # 2 for AZ
        num_planes = (6 * 2 + self.repetition_planes) * self.history_steps + 5

        self.image_shape = (num_planes, 8, 8)
        self.num_actions = 1858 # 4672 for AZ

        # Self-play info
        self.playout_cap_random_p = 1 #0.35 #0.25        
        self.num_mcts_sims = (200, 200) #(100, 800)
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
        self.fpu_leaf = 0.0
        self.policy_temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # Model info
        self.conv_filters = 64 # 96
        self.num_residual_blocks = 6 # 6
        self.output_dims = (self.num_actions,)
        self.conv_kernel_initializer = None # "he_normal" # None reverts to TF default setting
        self.use_bias_on_outputs = True
        self.value_head_filters = 32 # 32
        self.value_head_dense = 128
        self.policy_head_filters = 73 # 73
        self.value_head_loss_weight = 1.0
        self.policy_head_loss_weight = 1.0       

        # Model compilation info
        self.l2_reg = 1e-4
        self.sgd_momentum = 0.9
        self.sgd_nesterov = False

        # Tunable hyperparameters
        self.batch_size = 1024
        self.learning_rate = 0.05
        self.sampling_ratio = 0.8 # 0.0 - 1.0 -> Percentage of samples used for training
        self.softmax_temp = 1.0 # More move selection randomness in mcts


        # Training info
        self.num_actors = 2
        self.num_cycles = 15
        self.checkpoint_interval = 50 # Every n cycles
        self.games_per_cycle = 100 

        # STS Testing info
        self.sts_test_interval = 5 # Every n cycles
        self.sts_num_agents = 8
        self.sts_time_limit = 1.25

        # Save directories
        self.self_play_positions_dir = f"{current_dir}/data/positions"
        self.training_records_dir = f"{current_dir}/data/records"
        self.sts_results_dir = f"{current_dir}/data/sts_results"
        self.training_info_stats = f"{current_dir}/data/training_info.json"        
        self.tensorboard_log_dir = f"{current_dir}/logs/"
        self.test_suites_dir = f"{current_dir}/test_suites"
        self.keras_checkpoint_dir = f"{current_dir}/checkpoints/keras"
        self.trt_checkpoint_dir = f"{current_dir}/checkpoints/trt"
        self.tmp_trt_checkpoint_dir = f"{current_dir}/checkpoints/tmp_trt"

        self.syzygy_tb_dir = "/home/tomaz/syzygy"
        self.self_play_positions_backup_dir = "/home/tomaz/positions_backup"
        is_flipped = "_flipped" if self.history_perspective_flip else ""
        self.conversion_data_dir = f"{current_dir}/conversion_data/{self.history_steps}hist_{self.repetition_planes}reps{is_flipped}"

        # Define a name for model to diferentiante between different models on tensorboard
        self.model_name = f"b{self.num_residual_blocks}c{self.conv_filters}_bs{self.batch_size}{is_flipped}_"

        # TRT and GPU info
        self.allow_gpu_growth = True
        self.trt_precision_mode = 'INT8'