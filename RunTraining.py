# %% Training
from ga import run_ga
from datetime import datetime
#now = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")

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
        
# %% Testing single layouts     
layout = [77,10,10,3]
run_ga(
    path='Networks/Network_77x10x10x3_2',
    network_shape=layout,
    fitness_function = 'score',
    stochastic_spawning = True
    )       