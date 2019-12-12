#%% One way of playing the game

from Network import Network
from dxball import play_game
network = Network([77,10,10,3])
    
network.load(path='Networks/Network_77x10x10x3_1')
#%%

score, frames_run = play_game(network,use_network=1,display_game=1,fps=50,
              max_nbr_frames=1e5, stochastic_spawning = True)