from neural_network import NeuralNetwork
from dxball import play_game
import numpy as np
import copy



def main():
    pass






def run(params):


    population_size = 100
    mutation_probability = 0.04
    creep_rate = 0.4
    ts_parameter = 0.75
    ts_size = 2
    number_of_generations = 200
    number_of_copies = 5
    crossover_probability = 0.8

    training_courses = []
    validation_courses = [] 

    network_shape = [4, 4, 2]
    population = initialize(pop_size, network_shape)

    best_fitness_ever = 0
    best_individual_ever = 0
       

    for generation in range(number_of_generations):

        score_matr, time_matr = decode_population(population, training_courses)
        fitness = evaluate_population(score_matr, time_matr)
        res = np.where(fitness == np.amax(fitness))
        best_index = res[0][0]
        best_individual = copy.deepcopy(population[best_index])
        max_train_fitness = fitness[i]


        tmp_pop = copy.deepcopy(population)

        for i in range(0, population_size, 2):
            i1 = tournament_selection(fitness, ts_parameter, ts_size)
            i2 = tournament_selection(fitness, ts_parameter, ts_size)

            assert i1 >= 0  and i2 >= 0, 'Something went wrong in tournament selection'

            chromosome1 = population[i1]
            chromosome2 = population[i2]

            if np.random.random() < crossover_probability:
                chromosome1, chromosome2 = cross(chromosome1, chromosome2)  
            tmp_pop[i] = chromosome1
            tmp_pop[i+1] = chromosome2

        
        for i in range(population_size):
            chromosome = copy.deepcopy(population[i])
            mutated_chromosome = mutate_individual(chromosome, mutation_probability, creep_rate)
            tmp_pop[i] = mutated_chromosome

        tmp_pop = insert_best_individual(tmp_pop, best_individual, number_of_copies)
        population = tmp_pop


        # Validation
        score_matr, time_matr = decode_population(population, validation_courses)
        fitness = evaluate_population(score_matr, time_matr)
        res = np.where(fitness == np.amax(fitness))
        best_index = res[0][0]
        best_individual_validation = copy.deepcopy(population[best_index])
        max_validation_fitness = fitness[i]
        if max_validation_fitness > best_fitness_ever:
            best_fitness_ever = max_validation_fitness
            best_individual_ever = best_individual_validation

        print('Generation {}: Training fitness: {}, Validation fitness: {}'.format(
            generation, max_train_fitness, max_validation_fitness)
        )



        






def initialize(pop_size, network_shape):
    """ Initialize population of Neural Networks """
    population = []
    for _ in range(pop_size):
        nn = NeuralNetwork(shape = network_shape)
        population.append(nn)
    return population

def decode_population(population, courses):
    pop_size = len(population)
    score_matr, time_matr = np.zeros(shape = (population_size, len(courses)) )
    for i, individual in enumerate(population):
        chromosome = individual
        scores, play_times = decode_chromosome(chromosome, courses)
        score_matr[i,:] = scores
        time_matr[i,:] = play_times
    return score_matr, time_matr


def decode_chromosome(chromosome, courses):
    """ Decode chromosome by letting it play the game"""
    max_play_time = 60
    network = chromosome
    scores, play_times = [], []
    for course in courses:
        score, play_time = play_game(network, course, max_play_time)
        scores.append(score)
        play_times.append(play_time)
    return np.array(scores), np.array(play_times)


def evaluate_population(score_matr, time_matr):
    """
    Evaluate all individuals in population. At the moment the fitness is taken
    as the mean of score*play_time on each score
    """
    n_courses = scores.shape[1]
    score_time = scores*play_times
    score_time_sum = score_time.sum(axis=1)
    fitness = score_time_sum / n_courses
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
    i_count = -1
    while i_selected==0:
        i_count +=1
        r = np.random.random()

        if r < ts_parameter or  i_count == length(participants):
            i_selected = indices[i_count];
            return i_selected
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

    for i in range(len(network1.T)):
        t1, t2 = network1.T[i], network2.T[i]
        n = len(t1)
        assert n == len(t2), "Threshold vectors not the same length"

        index = np.random.randint(n)

        tmp = t1[:index].copy()
        t1[:index] = t2[:index].copy()
        t2[:index] = tmp.copy()

        network1.T[i], network2.T[i] = t1, t2

    chromosome1, chromosome2 = network1, network2
    return chromosome1, chromosome2



def insert_best_individual(population, best_individual, n_copies):
    """ Insert the best individual into the population"""
    for i in range(n_copies):
        population[i] = copy.deepcopy(best_individual)
    return population
















if __name__ == '__main__':
    main()