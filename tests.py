from Network import Network
from dxball import play_game
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def main():
    #test_network_save_and_load()
    #test_play_game_with_network()
    plot_training_evolution()
    #performance_to_csv()


def test_network_save_and_load():
    print_header('Testing save and load functionality of neural network')

    n1 = Network(shape = [4, 4, 2])

    path = 'results/hejsan'
    n1.save(path = path)

    n2 = Network(shape = [4, 4, 5, 2])
    n2.load(path = path)

    if n1==n2:
        print_success('TEST SUCCESSFUL!')
        return True
    else:
        print_failure('TEST FAILED!')
        print('Reason: saved and loaded networks not equal')
        return False

def test_play_game_with_network():

    path = 'results/oskar12_2/network_generation_16'

    print_header('Testing game with network {}'.format(path))

    try:
        network = Network([77,10,3])
        network.load(path=path)

        score, frames_run = play_game(
            network,
            use_network=1,
            course_nbr=666,
            display_game=1,
            fps=50,
            max_nbr_frames=10000
            )
        print(score)
        print_success('TEST SUCCESSFUL!')
        return True
    except Exception as e:
        print_failure('TEST FAILED!')
        print('Reason: {}'.format(str(e)))
        return False

def plot_training_evolution():

    S = [False, True]
    I = [5, 77]
    neurons = 20
    HL = [0,1,2,3]

    fps = 5000
    max_playtime = 30
    max_nbr_frames = max_playtime*fps

    stoch_fnames, non_stoch_fnames, labels = [],[], []
    for stoch in S:
        for n_inputs in I:

            fig, ax = plt.subplots()
            ax.set_xlabel('Generation', fontsize = 20)
            ax.set_ylabel('Score', fontsize = 20)



            title = "Stochastic with {} inputs".format(n_inputs)
            if not stoch:
                title = "Non-" + title
            ax.set_title(title, fontsize = 25)

            for n_hidden_layers in HL:

                path = 'ResultsGustafFinal/i={}_s={}_n={}_{}'.format(n_inputs, stoch, neurons, n_hidden_layers)
                fname = os.path.join(path, 'train_data.txt')
                label = "Hidden layers: {}".format(n_hidden_layers)

                try:
                    data = np.genfromtxt(fname, delimiter=',')
                    df = pd.DataFrame(data,columns = ['Gens', 'Best ever', 'Score'])
                    df = df.rolling(window = 21, center = True).mean()
                    x = df['Gens'].values
                    y = df['Score'].values
                    ax.plot(x, y, label = label, linewidth=3)
                except ValueError:
                    print(fname)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax.legend(fontsize = 15)
            plt.tight_layout()

            fig.savefig('plots/{}.eps'.format(title.replace(' ', '_')), format='eps')


    #plt.show()


def performance_to_csv():

    S = [False, True]
    I = [5, 77]
    N = [20]
    HL = [0,1,2,3]

    fps = 5000
    max_playtime = 30
    max_nbr_frames = max_playtime*fps

    fname = "final_res_table.csv"

    line = "Stochastic & Inputs & Hidden layers & Mean & $\\sigma$ & $\\sigma$/mean(\\%) \\\\ \\hline"
    print(line)
    with open(fname, 'w') as f:
        f.write(line + '\n')


    res_lines = []
    for stoch in S:
        for n_inputs in I:
            for neurons in N:
                for n_hidden_layers in HL:

                    network = Network([77,10,3])
                    path = 'ResultsGustafFinal/i={}_s={}_n={}_{}'.format(n_inputs, stoch, neurons, n_hidden_layers)
                    network.load(path=path)
                    scores = []
                    n = 100 if stoch else 1
                    for _ in range(n):
                        score, frames_run = play_game(
                            network,
                            use_network=1,
                            display_game=0,
                            fps=fps,
                            max_nbr_frames=max_nbr_frames,
                            stochastic_spawning = stoch,
                            network_generation=200,
                            pause_time=1)
                        #print(score)
                        scores.append(score)

                    mean_score = sum(scores)/len(scores)
                    std = '-' if not stoch else round(statistics.stdev(scores), 1)
                    std_perc = '-' if not stoch else round(100*std/mean_score, 1)
                    line = "{} & {} & {} & {} & {} & {} \\\\".format(
                        stoch, n_inputs, n_hidden_layers,
                        round(mean_score,1), std, std_perc
                        )
                    print(line)
                    with open(fname, 'a') as f:
                        f.write(line + '\n')













############################################################
# Printing functions

def print_header(string):
    print('{}{}{}'.format(HEADER, string, ENDC))
def print_success(string):
    print('{}{}{}'.format(OKGREEN, string, ENDC))
def print_failure(string):
    print('{}{}{}'.format(FAIL, string, ENDC))
def print_warning(string):
    print('{}{}{}'.format(WARNING, string, ENDC))







if __name__ == '__main__':
    main()