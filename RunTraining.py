# %%
from ga import run_ga



    
network_layouts = [[5,10,3],[5,10,10,3],[5,10,10,10,3],
                            [77,10,3],[77,10,10,3],[77,10,10,10,3]]

stochastic_spawns = [True, False]
counts = 0
for layout in network_layouts:
    for val in stochastic_spawns:    
        counts += 1
        print("counts=",counts)
        run_ga(
            path='/Network/Network'+str(counts),
            network_shape=layout,
            fitness_function = 'score',
            stochastic_spawning = False
            )

'''
def run_ga(
    path=None,
    network_shape=[77, 10, 3],
    generations = consts.N_GENERATIONS,
    population_size = consts.POPULATION_SIZE
    ):
'''