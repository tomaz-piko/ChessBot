class Config:
    def __init__(self):  
        # Game information
        self.image_shape = (8, 8, 119)
        self.num_actions = 4672

        # MCTS info
        self.num_mcts_sims = 1200
        self.num_mcts_sampling_moves = 0
        self.add_dirichlet_noise = False
        self.pb_c_factor = 1.5

        # PUCT and dirichlet noise parameters
        self.temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
