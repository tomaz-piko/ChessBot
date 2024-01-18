class Config:
    def __init__(self):       
        self.keras_checkpoint_dir = "checkpoints/keras/"
        self.trt_checkpoint = "checkpoints/trt/saved_model"

        self.trt_precision_mode = "FP32"

        # Game information
        self._N = 8 # board size
        self._M = 6 + 6 + 2 # Number of M feature planes (6 for each player's pieces, 2 for repetitions)
        self._L = 7 # Number of L feature planes (1 for color, 1 for movecount, 2 for P1 castling rights, 2 for P2 castling rights, 1 for halfmovecount)
        self.T = 8
        self.num_actions = 4672

        # Model information
        self.conv_filters = 32
        self.num_residual_blocks = 3 # 19 -> AZ default
        self.input_dims = (self._N, self._N, self._M * self.T + self._L)
        self.output_dims = (self.num_actions,)
        self.batch_norm_momentum = 0.6
        self.l2_reg = 1e-4
        self.learning_rate = {0: 2e-1, 15: 2e-2, 45: 2e-3, 75: 2e-4}
        self.momentum = 0.9

        # MCTS info
        self.num_mcts_sims = (35, 150)
        self.num_mcts_sims_p = 0.25
        self.num_sampling_moves = 30
        self.temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # Training info
        self.max_game_length = 512
        self.buffer_size = 16384
        self.batch_size = 512
        self.training_steps = 100
        self.checkpoint_interval = 5
        self.epochs = 1
        self.verbose = 1
        self.use_trt = True
        self.allow_gpu_growth = True
