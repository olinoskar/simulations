#%% One way of playing the game

import argparse
from Network import Network
from dxball import play_game
network = Network([5,3])  # initiate
 
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-fps', "--fps", type=int, default=50, help='fps')
	parser.add_argument('-gen', "--network_generation", type=int, default=0, help='gen')
	parser.add_argument('-i', "--inputs", type=int, default=5, help='inputs')
	parser.add_argument('-s',"--stochastic_spawning", type=int, default = 0)
	parser.add_argument('-hl',"--hidden_layers", type=int, default=0)
	args = parser.parse_args()

	# Params for loading network
	inputs = args.inputs
	stoch_bool_network = (args.stochastic_spawning==1)
	nbr_hidden_layers = args.hidden_layers
	nbr_neurons_per_hidden = 20  # default = 20
	network_generation = args.network_generation

	network.load(path='ResultsGustafFinal/'
						+ 'i=' + str(inputs)
						+ '_s=' + str(stoch_bool_network)
						+ '_n=' + str(nbr_neurons_per_hidden)
						+ '_' + str(nbr_hidden_layers)
						+ '/network_generation_' + str(network_generation))

	# Params for playing the game
	fps = args.fps
	max_playtime = 30
	max_nbr_frames = max_playtime*fps
	stoch_bool_game = stoch_bool_network
	use_network = 1

	if use_network == 0:
		fps = 50

	score, frames_run = play_game(network, use_network=use_network, display_game=1, fps=fps,
				  max_nbr_frames=max_nbr_frames, stochastic_spawning = (stoch_bool_game==1),
				  network_generation=network_generation)

	print("Score="+str(score))

def str2bool(v): # for parsing of stochastic spawning argument
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
	main()