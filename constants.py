# Game constants


# Network constants



# GA constants
POPULATION_SIZE = 20    # population size
MAX_FRAMES = 60*60      # maximum number of frames
MIN_MUT_PROB = 0.04     # minimum mutation probability (1-> MIN_MUT_PROB)
MUT_RED_RATE = 0.1      # mutation reduction rate (mut <- exp(-gen*MUT_RED_RATE))
CREEP_RATE = 0.4        # creep rate for mutation
TS_PARAM = 0.75         # tournament selection parameter
TS_SIZE = 2             # tournament selection size
N_GENERATIONS = 25     # number of generations
N_COPIES = 2            # number of copies in elitism
CROSS_PROB = 0.8        # crossover probability