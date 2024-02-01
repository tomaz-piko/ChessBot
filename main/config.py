from tensorflow.python.compiler.tensorrt.trt_convert import TrtPrecisionMode

class Config:
    def __init__(self):       
        self.keras_checkpoint_dir = "checkpoints/keras"
        self.trt_checkpoint_dir = "checkpoints/trt"
        self.tensorboard_log_dir = "logs/fit"

        self.trt_precision_mode = TrtPrecisionMode.FP32

        # Game information
        self._N = 8 # board size
        self._M = 6 + 6 + 2 # Number of M feature planes (6 for each player's pieces, 2 for repetitions)
        self._L = 7 # Number of L feature planes (1 for color, 1 for movecount, 2 for P1 castling rights, 2 for P2 castling rights, 1 for halfmovecount)
        self.T = 8
        self.num_actions = 4672

        # Model information
        self.conv_filters = 48
        self.num_residual_blocks = 4 # 19 -> AZ default
        self.input_dims = (self._N, self._N, self._M * self.T + self._L)
        self.output_dims = (self.num_actions,)
        self.l2_reg = 4e-4
        self.learning_rate = {0: 1e-1, 14000: 1e-2, 28000: 1e-3, 36000: 1e-4}
        self.momentum = 0.9

        # MCTS info
        self.num_mcts_sims = (50, 300)
        self.num_mcts_sims_p = 0.25
        self.num_mcts_sampling_moves = 30

        # Training info
        self.max_game_length = 512
        self.buffer_size = 1024 # Number of positions to store in buffer
        self.minimum_buffer_size = int(self.buffer_size*(3/4)) # Fill three quarters of buffer before starting training
        self.batch_size = 64 # Number of positions to sample from buffer
        self.training_steps = 40000
        self.pause_between_steps = 2
        self.checkpoint_interval = 2000
        self.epochs = 1
        self.verbose = 1
        self.allow_gpu_growth = True
