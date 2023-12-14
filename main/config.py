# Model information
num_residual_blocks = 19
input_dims = (8, 8, 119)
output_dims = (4672,)

# MCTS
num_mcts_sims = 10
num_sampling_moves = 30
pb_c_base = 19652
pb_c_init = 1.25
root_dirichlet_alpha = 0.3
root_exploration_fraction = 0.25

# Training
num_actors = 6
max_game_length = 4
buffer_window_size = 1000
buffer_batch_size = 32
training_steps = 10
checkpoint_interval = 2
epochs = 10
verbose = 1
