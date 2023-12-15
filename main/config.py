class Config:
    def __init__(self, T=8, conv_filters=256, num_residual_blocks=19, num_mcts_sims=25, num_sampling_moves=10, num_actors=6, max_game_length=128, buffer_window_size=1000, buffer_batch_size=32, training_steps=10, checkpoint_interval=2, epochs=10, verbose=1):
        # Game information
        self._N = 8 # board size
        self._M = 6 + 6 + 2 # Number of M feature planes (6 for each player's pieces, 2 for repetitions)
        self._L = 7 # Number of L feature planes (1 for color, 1 for movecount, 2 for P1 castling rights, 2 for P2 castling rights, 1 for halfmovecount)
        self.T = T
        self.num_actions = 4672

        # Model information
        self.conv_filters = conv_filters
        self.num_residual_blocks = num_residual_blocks # 19 -> AZ default
        self.input_dims = (self._N, self._N, self._M * self.T + self._L)
        self.output_dims = (self.num_actions,)

        # MCTS info
        self.num_mcts_sims = num_mcts_sims
        self.num_sampling_moves = num_sampling_moves
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # Training info
        self.num_actors = num_actors
        self.max_game_length = max_game_length
        self.buffer_window_size = buffer_window_size
        self.buffer_batch_size = buffer_batch_size
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval
        self.epochs = epochs
        self.verbose = verbose
