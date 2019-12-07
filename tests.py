from Network import Network

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



def test_network_save_and_load():
    print_header('Testing save and load functionality of neural network')

    n1 = Network(shape = [4, 4, 2])

    path = 'results/hejsan'
    n1.save(path = path)

    n2 = Network(shape = [4, 4, 5, 2])
    n2.load(path = path)

    if n1==n2:
        print_success('TEST SUCCESSFUL!')
    else:
        print_success('TEST FAILED!')
        print('Reason: saved and loaded networks not equal')



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