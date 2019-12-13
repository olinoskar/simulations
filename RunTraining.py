# %% Training
from ga import run_ga
from datetime import datetime
import argparse

#now = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")

"""
network_layouts = [[5,10,3],[5,10,10,3],[5,10,10,10,3],
                            [77,10,3],[77,10,10,3],[77,10,10,10,3]]

# %% Sweeping over parameters in lists above
stochastic_spawns = [True, False]
counts = 0
for layout in network_layouts:
    for stoch in stochastic_spawns:    
        counts += 1
        print("counts=",counts)
        run_ga(
            path='Networks/Network'+str(counts),
            network_shape=layout,
            fitness_function = 'score',
            stochastic_spawning = stoch
            )

"""
     
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# %% Testing single layouts 

if __name__ == '__main__': 


    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',
        help='Name of directory to save the network to')
    parser.add_argument('-s', "--stochastic_spawning", type=str2bool, nargs='?',
        const=True, default=False,
        help="Stochastic spwaning or not (true or false)")
    parser.add_argument('-l', "--network_layout",
        help = 'Layout of network. Should be passed as a string, example: -l="77, 10, 10, 3"')
    args = parser.parse_args()

    if not args.path or not args.stochastic_spawning or not args.network_layout:
        print('All arguments must be passed. Exiting')
        import sys
        sys.exit()


    path = args.path
    stoch = args.stochastic_spawning
    layout_str = args.network_layout

    layout = []
    for s in layout_str.split(','):
        layout.append(int(s))

    print('Saving to: ', path)
    print('Stochastic spawning:', stoch)
    print('Network layout:', layout)

    import sys
    sys.exit()


    run_ga(
        path=path,
        network_shape=layout,
        fitness_function = 'score',
        stochastic_spawning = stoch
        ) 
     