#%% One way of playing the game

import argparse
from Network import Network
from dxball import play_game
network = Network([5,3])  # initiate
 
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', "--fps", type=int, default=50, help='fps')
	args = parser.parse_args()

	# Params for loading network
	stoch_bool_network = True
	inputs = 77
	nbrHiddenLayers = 2
	nbrNeuronsPerHidden = 10

	network.load(path='ResultsGustaf/'
						+ 'Stoch=' + str(stoch_bool_network)
						+ '_inputs=' + str(inputs)
						+ '_n_hidden_layers=' + str(nbrHiddenLayers)
						+ '_neurons=' + str(nbrNeuronsPerHidden))

	# Params for playing the game
	fps = args.fps
	max_playtime = 500000
	max_nbr_frames = max_playtime*fps
	stoch_bool_game = True
	use_network = 1

	if use_network == 0:
		fps = 50

	score, frames_run = play_game(network, use_network=use_network, display_game=1, fps=fps,
				  max_nbr_frames=max_nbr_frames, stochastic_spawning = (stoch_bool_game==1))

	print("Score="+str(score))

if __name__ == '__main__':
	main()