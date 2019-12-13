#%% One way of playing the game

from Network import Network
from dxball import play_game
network = Network([77,10,10,3])
    
network.load(path='ResultsGustaf/Stoch=true/inputs=5_n_hidden_layers=1_neurons=5')
#network.load(path='stochastic/77_inputs/20_neurons_1')

score, frames_run = play_game(network,use_network=1,display_game=1,fps=50,
              max_nbr_frames=1e5, stochastic_spawning = True)