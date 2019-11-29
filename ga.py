from neural_network import NeuralNetwork

from dxball_edited import Bricka

import numpy as np



def main():
    pass



def initialize(pop_size, params):

    population = []
    for _ in range(pop_size):
        nn = NeuralNetwork(params)
        population.append(nn)
    return population


def decode_chromosome(chromosome, courses):

    network = chromosome
    scores = []
    for course in courses:
        score = rame.run(network, course)
        scores.append(score)
    return scores


def evaluate_pop():
    pass

def mutate_pop():
    pass


def select():
    pass

def cross():
    pass















if __name__ == '__main__':
    main()