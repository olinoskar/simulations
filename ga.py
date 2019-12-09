from Network import Network
from dxball import play_game
import numpy as np
import copy
from pprint import pprint

import argparse



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', help='Name of the first file')
    args = parser.parse_args()
    run(path = args.path)



def run(path = None):

    print('\n')
    population_size = 20

    frames = 60*30



    mutation_probability = 0.04
    creep_rate = 0.4
    ts_parameter = 0.75
    ts_size = 2
    number_of_generations = 200
    number_of_copies = 2
    crossover_probability = 0.8

    training_courses = [666]
    validation_courses = [666] 

    network_shape = [5, 10, 3]
    population = initialize(population_size, network_shape)

    time_fun = np.vectorize(time_effect)


    assert len(population)==population_size

    best_fitness_ever = 0
    best_individual_ever = 0
       

    for generation in range(number_of_generations):


        mutation_rate = mutation_probability + np.exp(-generation/10)
        if mutation_rate > 1:
            mutation_rate = 0.9999999999

        print('\n\nGeneration: {}'.format(generation))

        score_matr, time_matr = decode_population(population, training_courses)
        fitness = evaluate_population(score_matr, time_matr, frames)


        #print('Average training fitness: {}'.format(np.mean(fitness)))
        res = np.where(fitness == max(fitness))
        best_index = res[0][0]
  
        best_individual = copy.deepcopy(population[best_index])
        max_train_fitness = fitness[best_index]
        print('Max training fitness: {}'.format(max_train_fitness))



        tmp_pop = copy.deepcopy(population)
        for i in range(0, population_size, 2):
            i1 = tournament_select(fitness, ts_parameter, ts_size)
            i2 = tournament_select(fitness, ts_parameter, ts_size)

            assert i1 >= 0  and i2 >= 0, 'Something went wrong in tournament selection'

            chromosome1 = population[i1]
            chromosome2 = population[i2]

            if np.random.random() < crossover_probability:
                chromosome1, chromosome2 = cross(chromosome1, chromosome2)  
            tmp_pop[i] = chromosome1
            tmp_pop[i+1] = chromosome2

        for i in range(population_size):
            chromosome = copy.deepcopy(population[i])    
            chromosome.mutate(mutationrate = mutation_rate, creeprate = creep_rate)
            mutated_chromosome = chromosome

            tmp_pop[i] = mutated_chromosome



        score, play_time,_ = play_game(best_individual, course_nbr=training_courses[0], max_nbr_frames = 60*30, fps=50000)
        score_matr, time_matr = decode_population(tmp_pop, training_courses)
        fitness = evaluate_population(score_matr, time_matr)



        tmp_pop = insert_best_individual(tmp_pop, best_individual, number_of_copies)
        population = copy.deepcopy(tmp_pop)

        
        # Validation
        score_matr, time_matr = decode_population(population, validation_courses)
        fitness = evaluate_population(score_matr, time_matr, frames)

        #print('Average validation fitness: {}'.format(np.mean(fitness)))
        res = np.where(fitness == np.amax(fitness))
        best_index = res[0][0]

        best_individual_validation = copy.deepcopy(population[best_index])
        max_validation_fitness = fitness[best_index]
        print('Max validation fitness: {}'.format(max_validation_fitness))
        if max_validation_fitness > best_fitness_ever:
            print('best_fitness_ever: {}'.format(best_fitness_ever))
            best_fitness_ever = max_validation_fitness
            best_individual_ever = copy.deepcopy(best_individual_validation)
            if path:
                print('Saving network to: {}'.format(path))
                best_individual_ever.save(path = path)

        
        print('Generation {}: Training fitness: {}, Validation fitness: {}'.format(
            generation, max_train_fitness, max_validation_fitness)
        )


       


def initialize(pop_size, network_shape):
    """ Initialize population of Neural Networks """
    population = []
    for _ in range(pop_size):
        nn = Network(shape = network_shape)
        population.append(nn)
    return population

def decode_population(population, courses):
    pop_size = len(population)
    score_matr = np.zeros(shape = (pop_size, len(courses)) )
    time_matr = np.zeros(shape = (pop_size, len(courses)) )
    for i, individual in enumerate(population):
        chromosome = individual
        scores, play_times = decode_chromosome(chromosome, courses)
        score_matr[i,:] = scores
        time_matr[i,:] = play_times
    return score_matr, time_matr


def decode_chromosome(chromosome, courses, frames):
    """ Decode chromosome by letting it play the game"""
    network = chromosome
    scores, play_times = [], []
    for course in courses:
        score, play_time,_ = play_game(network, course_nbr=course, max_nbr_frames = frames, fps=5000)
        scores.append(score)
        play_times.append(play_time)
    return np.array(scores), np.array(play_times)


def evaluate_population(score_matr, time_matr, frames):
    """
    Evaluate all individuals in population. At the moment the fitness is taken
    as the mean of score*play_time on each score.
    """
    n_courses = score_matr.shape[1]
    time_matr = time_fun(time_matr, frames)

    score_time = score_matr*time_matr
    score_time_sum = score_time.sum(axis=1)
    fitness = score_time_sum / n_courses
    #fitness = score_matr.sum(axis=1) / n_courses
    return fitness


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
    return 1 - 1/(1+np.exp(-play_time/frames+1))












if __name__ == '__main__':
    main()