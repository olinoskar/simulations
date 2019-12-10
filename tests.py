from Network import Network
from dxball import play_game



HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def main():
    test_network_save_and_load()
    test_play_game_with_network()


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

    path = 'results/oskar3'

    print_header('Testing game with network {}'.format(path))

    try:
        network = Network([5,3,3])
        network.load(path=path)

        score, frames_run, fitness = play_game(
            network,
            use_network=1,
            course_nbr=666,
            display_game=1,
            fps=50,
            max_nbr_frames=10000,
            score_exponent=1, 
            frame_exponent=1
            )
        print_success('TEST SUCCESSFUL!')
        return True
    except Exception as e:
        print_failure('TEST FAILED!')
        print('Reason: {}'.format(str(e)))
        return False




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