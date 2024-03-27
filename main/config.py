class Config:
    def __init__(self):  
        # Game information
        self.image_shape = (109, 8, 8)
        self.num_actions = 4672

        # MCTS info
        self.num_parallel_reads = 16
        self.num_mcts_sims = None
        self.time_limit = 15.0 # seconds
        self.num_mcts_sampling_moves = 0
        self.policy_temp = 1.125
        self.softmax_temp = 1.0
        self.pb_c_factor = (2.0, 2.0)

        # PUCT and dirichlet noise parameters
        self.pb_c_base = 19652
        self.pb_c_init = 2.5
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.fpu_root = 1.0
        self.fpu_leaf = 0.3