#%% One way of playing the game

from Network import Network
from dxball import play_game
network = Network([77,3,3])
    
#network.load(path='results/test_run2')
#%%
v_factor = 1
input_fps = 50/v_factor
score, frames_run = play_game(network,use_network=0,course_nbr=666,display_game=1,fps=input_fps,
              max_nbr_frames=1e5, initial_velocity = 5, velocity_exponents = [1.010, 1.025, 1.050],
              stochastic_spawning = True, velocity_factor = v_factor)