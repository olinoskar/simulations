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
	inputs = 77
	stoch_bool_network = True
	nbr_hidden_layers = 0
	nbr_neurons_per_hidden = 20  # default = 20
	network_generation = 22

	network.load(path='ResultsGustafFinal/'
						+ 'i=' + str(inputs)
						+ '_s=' + str(stoch_bool_network)
						+ '_n=' + str(nbr_neurons_per_hidden)
						+ '_' + str(nbr_hidden_layers)
						+ '/network_generation_' + str(network_generation))

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