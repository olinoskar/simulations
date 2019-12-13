#%%
from Network import Network
from dxball import play_game
import numpy as np
import copy
from pprint import pprint
import constants as consts
#import test_constants as consts
import argparse
import multiprocessing as mp

"""
File dedicated to training using a genetic algorithm. Saves results to ga_results_info.txt.

Arguments:

    * path: path to file to save the best network. Note that existing networks will be overritten.
        - Example: python3 ga.py -p=results/test_run2

"""


def main():

    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    inputs = [5,77]
    hidden = [5,10,20]
    pop_size = 20
    nbr_generations = 30
    stoch_bool = False
    
    for n_inputs in inputs:
        for n_hidden in hidden:
            shapes = [
             [n_inputs,n_hidden,3],
             [n_inputs,n_hidden,n_hidden,3],
             [n_inputs,n_hidden,n_hidden,n_hidden,3]
            ]
            nbr_layers = 0
            for shape in shapes:
                nbr_layers += 1
                run_ga(
                    path='ResultsGustaf/Stoch=False/inputs='+str(n_inputs)+'_n_hidden_layers='+str(nbr_layers)+'_neurons='+str(n_hidden),
                    network_shape=shape,
                    generations = nbr_generations,
                    population_size = pop_size,
                    fitness_function = 'score',
                    stochastic_spawning = stoch_bool
                )

    return

def run_ga(
    path=None,
    network_shape=[5, 20, 20, 3],
    generations = consts.N_GENERATIONS,
    population_size = consts.POPULATION_SIZE,
    fitness_function = 'score',
    stochastic_spawning = True
    ):


    """
    Train a neural network using a genetic algorithm. Saves results to ga_results_info.txt.

    Arguments:

        * path (str): save the best network to this path (directory). 
        * network_shape (list): shape of the neural network. List of integers.
        * generations (int): Number of generations.
        * population_size (int): Number of individals in the population.
    """


    training_courses = [666, 666, 666]
    validation_courses = [666, 666, 666] 
    testing_courses = [666, 666, 666] 

    population = initialize(population_size, network_shape)

    time_pen = np.vectorize(time_effect) # Used to penalize the elapsed time of a run


    best_train_fitness_ever = 0
    best_validation_fitness = 0
    best_validation_generation = 0
    best_individual_ever = None
    
       
    # Initiate mutation as 1/m, where m is the number of genes
    min_mut_rate = 0
    for i in range(len(network_shape)-1):
        min_mut_rate += network_shape[i+1]*network_shape[i]
        min_mut_rate += network_shape[i+1]
    min_mut_rate = 1/min_mut_rate


    # Start training
    for generation in range(generations):

        # Set mutation rate according to a + exp(-g*b)
        mutation_rate = min_mut_rate + np.exp(-generation*consts.MUT_RED_RATE)
        if mutation_rate > 1:
            mutation_rate = 0.9999999999

        # Evaluate population on training courses
        score_matr, time_matr = decode_population(population, training_courses, consts.MAX_FRAMES, stochastic_spawning)
        fitness, best_index = evaluate_population(score_matr, time_matr, time_pen, consts.MAX_FRAMES, fitness_function)
        best_individual = copy.deepcopy(population[best_index])
        max_train_fitness = fitness[best_index]

        # Store (and save) the best individual if this is the best fitness ever
        if max_train_fitness > best_train_fitness_ever:
            best_train_fitness_ever = max_train_fitness
            best_individual_ever = copy.deepcopy(best_individual)
            if path:
                print('Saving network to: {}'.format(path))
                best_individual_ever.save(path = path)


        # Create a temporary population and perform tournament selection, crossover and mutation
        tmp_pop = copy.deepcopy(population)
        for i in range(0, population_size, 2):
            i1 = tournament_select(fitness, consts.TS_PARAM, consts.TS_SIZE)
            i2 = tournament_select(fitness, consts.TS_PARAM, consts.TS_SIZE)

            chromosome1 = population[i1]
            chromosome2 = population[i2]

            if np.random.random() < consts.CROSS_PROB:
                chromosome1, chromosome2 = cross(chromosome1, chromosome2)
                if type(chromosome1) == list or type(chromosome2) == list: # DEBUGGING
                    raise ValueError("Cross returns a list")  

            tmp_pop[i] = chromosome1
            tmp_pop[i+1] = chromosome2

        for i in range(population_size):
            chromosome = copy.deepcopy(population[i])    
            chromosome.mutate(mutationrate = mutation_rate, creeprate = consts.CREEP_RATE)
            mutated_chromosome = chromosome

            if type(mutated_chromosome) == list: # DEBUGGING
                raise ValueError("Mutation returns a list")

            tmp_pop[i] = mutated_chromosome

        # Elistism step
        tmp_pop = insert_best_individual(tmp_pop, best_individual, consts.N_COPIES)

        # Set population to the temporary
        population = copy.deepcopy(tmp_pop)

        # Run the best individual on the validation courses (sometimes)
        if generation > 100 and generation%5==0:
            score_matr, time_matr = decode_population([best_individual], validation_courses, consts.MAX_FRAMES, stochastic_spawning)
            fitness,_ = evaluate_population(score_matr, time_matr, time_pen, consts.MAX_FRAMES, fitness_function)
            val_fitness = fitness[0]

            if val_fitness > best_validation_fitness:
                best_validation_fitness = val_fitness
                best_validation_generation = generation
            elif generation - best_validation_generation > 50:
                print("Results are not improving. Done training!")
                break;
  
            print('Generation {}: Training fitness: {}, Validation fitness: {}'.format(
                generation, round(max_train_fitness,2), round(best_validation_fitness,2) )
            )
        else:
            print('Generation {}: Training fitness: {}'.format(
                generation, round(max_train_fitness,2))
            )

    # Get results from training courses
    score_matr, time_matr = decode_population([best_individual_ever], training_courses, consts.MAX_FRAMES, stochastic_spawning)
    fitness,_ = evaluate_population(score_matr, time_matr, time_pen, consts.MAX_FRAMES, fitness_function)
    train_fitness = round(fitness[0],2)
    mean_train_score = round(np.mean(score_matr))

    # Get results from validation courses
    score_matr, time_matr = decode_population([best_individual_ever], validation_courses, consts.MAX_FRAMES, stochastic_spawning)
    fitness,_ = evaluate_population(score_matr, time_matr, time_pen, consts.MAX_FRAMES, fitness_function)
    val_fitness = round(fitness[0],2)
    mean_val_score = round(np.mean(score_matr))


    # Get results from testing courses
    score_matr, time_matr = decode_population([best_individual_ever], testing_courses, consts.MAX_FRAMES, stochastic_spawning)
    fitness,_ = evaluate_population(score_matr, time_matr, time_pen, consts.MAX_FRAMES, fitness_function)
    test_fitness = round(fitness[0],2)
    mean_test_score = round(np.mean(score_matr))

    print("\nResults on test sets:")
    print("Fitness:", round(test_fitness, 2))
    print("Score:", round(mean_test_score))



    res_line = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
        path,
        mean_train_score,
        mean_val_score,
        mean_test_score,
        round(train_fitness,2),
        round(val_fitness,2),
        round(test_fitness,2),
        generation,
        population_size,
        fitness_function,
        stochastic_spawning,
        len(training_courses),
        network_shape,
        mutation_rate
        )

    with open('ga_results.txt', 'a') as f:
        f.write(res_line)


    

def initialize(pop_size, network_shape):
    """ Initialize population of Neural Networks """
    population = []
    for _ in range(pop_size):
        nn = Network(shape = network_shape)
        population.append(nn)
    return population

def decode_population(population, courses, frames, stochastic_spawning):
    pop_size = len(population)
    score_matr = np.zeros(shape = (pop_size, len(courses)) )
    time_matr = np.zeros(shape = (pop_size, len(courses)) )
    for i, individual in enumerate(population):
        chromosome = individual
        scores, play_times = decode_chromosome(chromosome, courses, frames, stochastic_spawning)

        score_matr[i,:] = scores
        time_matr[i,:] = play_times

    return score_matr, time_matr

def decode_chromosome(chromosome, courses, frames, stochastic_spawning):
    """ Decode chromosome by letting it play the game"""
    network = chromosome

    scores, play_times = [], []
    for course in courses:
        score, play_time = play_game(
            network = network,
            use_network = 1,
            display_game = 0,
            course_nbr=course,
            max_nbr_frames = frames,
            fps=5000,
            stochastic_spawning = stochastic_spawning
        )

        scores.append(score)
        play_times.append(play_time)

    return np.array(scores), np.array(play_times)


def evaluate_population(score_matr, time_matr, time_pen, frames, fitness_function):
    """
    Evaluate all individuals in population. At the moment the fitness is taken
    as the mean of score*play_time on each score.
    """

    n_courses = score_matr.shape[1]

    if fitness_function == 'score':
        score_sum = score_matr.sum(axis=1)
        fitness = score_sum / n_courses


    elif fitness_function == 'score_time':
        score_time = score_matr*time_matr
        score_time_sum = score_time.sum(axis=1)
        fitness = score_time_sum / n_courses

    elif fitness_function == 'score_time_pen':
        time_matr = time_pen(time_matr, frames)
        score_time = score_matr*time_matr
        score_time_sum = score_time.sum(axis=1)
        fitness = score_time_sum / n_courses

    else:
        raise ValueError("Fitness functions not allowed: {}".format(fitness_function))

    res = np.where(fitness == max(fitness))
    best_index = res[0][0]
    return fitness, best_index



def mutate_individual(population, mutation_parameter, creep_rate):
    """ Mutate individual based """
    for individual in population:
        individual.mutate(mutation_parameter, creep_rate)
    return population


def tournament_select(fitness, ts_parameter, ts_size):
    """ Tournament selection according to the fitness list. """
    pop_size = len(fitness)
    participants = []
    for _ in range(ts_size):
        i = np.random.choice(pop_size)
        participants.append(i)

    indices = np.argsort(participants)

    i_selected = -1
    i_count = 0
    while i_selected==-1:
        
        r = np.random.random()
        if r < ts_parameter or  i_count == len(participants)-1:
            i_selected = indices[i_count];
            return i_selected
        i_count +=1
    return i_selected



def cross(chromosome1, chromosome2):
    """ Cross two chromosomes using the cross function in the Network class. """
    network1, network2 = chromosome1, chromosome2

    for i in range(len(network1.W)):
        w1, w2 = network1.W[i], network2.W[i]
        (n,m) = w1.shape
        assert (n,m) == w2.shape, "w1 and w2 not the same shape"

        w1_vec, w2_vec = w1.reshape(n*m), w2.reshape(n*m)
        index = np.random.randint(n*m)

        tmp = w1_vec[:index].copy()
        w1_vec[:index] = w2_vec[:index].copy()
        w2_vec[:index] = tmp.copy()

        w1, w2 = w1_vec.reshape(n,m), w2_vec.reshape(n,m)
        network1.W[i], network2.W[i] = w1, w2

    for i in range(len(network1.Theta)):
        t1, t2 = network1.Theta[i], network2.Theta[i]
        n = len(t1)
        assert n == len(t2), "Threshold vectors not the same length"

        index = np.random.randint(n)

        tmp = t1[:index].copy()
        t1[:index] = t2[:index].copy()
        t2[:index] = tmp.copy()

        network1.Theta[i], network2.Theta[i] = t1, t2

    chromosome1, chromosome2 = network1, network2

    return chromosome1, chromosome2



def insert_best_individual(population, best_individual, n_copies):
    """ Insert the best individual into the population"""

    indices = range(n_copies)
    indices = np.random.choice(len(population), n_copies, replace = False)

    for i in indices:
        population[i] = copy.deepcopy(best_individual)
    return population




##################################################
# Helper functions

def time_effect(played_time, frames):
    return 1 - 1/(1+np.exp(-played_time/frames+1))

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