#%% One way of playing the game

from Network import Network
from dxball import play_game
network = Network([5,3])  # initiate
 

# Params for loading network
stoch_bool_network = True
inputs = 5
nbrHiddenLayers = 1
nbrNeuronsPerHidden = 5

network.load(path='ResultsGustaf/'
                    + 'Stoch =' + str(stoch_bool_network)
                    + '_inputs=' + str(inputs)
                    + '_n_hidden_layers=' + str(nbrHiddenLayers)
                    + '_neurons=' + str(nbrNeuronsPerHidden))
#network.load(path='ResultsGustaf/Stoch=False/inputs=5_n_hidden_layers=2_neurons=5')

# Params for playing the game
fps = 200
max_playtime = 500000
max_nbr_frames = max_playtime*fps
stoch_bool_game = True
use_network = 0

if use_network == 0:
	fps = 50

score, frames_run = play_game(network, use_network=use_network, display_game=1, fps=fps,
              max_nbr_frames=max_nbr_frames, stochastic_spawning = stoch_bool_game)

print("Score="+str(score))