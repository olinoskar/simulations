from neural_network import NeuralNetwork
from dxball import play_game
import numpy as np
import copy



def main():
    pass



def run(params):


    training_courses = []
    validation_courses = []
    pop_size = 100

    network_shape = [4, 4, 2]

    population = initialize(pop_size, network_shape)

    max_train_fitness = 0    

    for generation in range(n_generations):


        score_matr, time_matr = np.zeros(shape = (pop_size, len(training_courses)) )
        for i, individual in enumerate(population):
            scores, play_times = decode_chromosome(chromosome, training_courses)
            score_matr[i,:] = scores
            time_matr[i,:] = play_times

        fitness = evaluate_population(score_matr, time_matr)
        res = np.where(fitness == np.amax(fitness))
        best_index = res[0][0]
        best_individual = copy.deepcopy(population[best_index])
        if fitness[i] > max_train_fitness:
            max_train_fitness = fitness[i]


        tmp_pop = copy.deepcopy(population)

        for i in range(0, pop_size, 2):
            i1 = tournament_selection(fitness, ts_parameter, ts_size)
            i2 = tournament_selection(fitness, ts_parameter, ts_size)

            assert i1 >= 0  and i2 >= 0, 'Something went wrong in tournament selection'

            tmp_pop[i] = population[i1]
            tmp_pop[i+1] = population[i2]

        
        for i in range(pop_size):
            chromosome = copy.deepcopy(population[i])
            mutated_chromosome = mutate_individual(chromosome, mutation_prob, creep_rate)
            tmp_pop[i] = mutated_chromosome

        tmp_pop = insert_best_individual(tmp_pop, best_individual, n_copies)
        population = tmp_pop









def initialize(pop_size, network_shape):
    """ Initialize population of Neural Networks """
    population = []
    for _ in range(pop_size):
        nn = NeuralNetwork(shape = network_shape)
        population.append(nn)
    return population


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


def mutate_individual(population, mutation_parameter):
    """ Mutate individual based """
    for individual in population:
        if np.random.random() < mutation_parameter:
            individual.mutate()
    return population



def tournament_select(fitness, ts_parameter, ts_size):
    
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





def cross():
    pass

def insert_best_individual(population, best_individual, n_copies):
    """ Insert the best individual into the population"""
    for i in range(n_copies):
        population[i] = copy.deepcopy(best_individual)
    return population
















if __name__ == '__main__':
    main()