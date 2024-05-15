class Config:
    def __init__(self):
        self.history_steps = 6 # 8 for AZ
        self.repetition_planes = 0 # 2 for AZ
        num_planes = (6 * 2 + self.repetition_planes) * self.history_steps + 5

        self.image_shape = (num_planes, 8, 8)
        self.num_actions = 1858 # 4672 for AZ

        # Self-play info
        self.num_mcts_sims = 5000
        self.num_mcts_sampling_moves = 0
        self.pb_c_factor = (1.0, 1.0)

        # MCTS constants
        self.num_parallel_reads = 16
        self.fpu_root = 1.0
        self.fpu_leaf = 0.3
        self.softmax_temp = 1.0
        self.policy_temp = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25 # 2.5 -> Lc0
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25