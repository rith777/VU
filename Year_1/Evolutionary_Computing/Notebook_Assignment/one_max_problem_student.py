import numpy as np
from numpy.random import randint
from numpy.random import rand
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

np.random.seed(5)


# Define the population:

def init_population(n_population: int, bit_length: int) -> list:
    '''This returns a randomly initialized list of individual solutions of size n_population.'''

    '''
    ToDo:

    Please write a code that initializes a set of random solution to the problem.
    '''

    l1 = []

    for i in range(n_population):
        l2 = []
        for j in range(bit_length):
            addition = randint(0, 2)
            l2.append(addition)
        l1.append(l2)
    l1 = np.array(l1)
    return l1


# Define the fitness function. The fitness function represents the environment.
def one_max(x: list) -> float:
    '''Takes a list of length bit_length and returns the sum of its elements.'''
    return np.sum(x)


def fittest_solution(fitness_function: callable, generation) -> float:
    '''This returns the highest fitness value of the whole generation.'''
    return np.max([fitness_function(generation[i]) for i in range(generation.shape[0])])


# Define the mutation operator

# First define the bit flipping operator

def bit_flipping(x: list) -> list:
    '''
    ToDo:

    Please write a code that flips the bit, i.e. changes a 1 to a 0 and vice versa.
    '''
    for i in range(len(x)):
        if x[i] == 0:
            x[i] = 1
        else:
            x[i] = 0

    return x


def mutation_operator(mutation_function: callable, p_mutation: float, x: list) -> np.ndarray:
    '''This function takes the mutation function and applies it
    element-wise to the genes according to the mutation rate.'''

    return np.asarray([mutation_function(gene) if (np.random.uniform() <= p_mutation) else gene for gene in x])


def cross_over(parent_1: list, parent_2: list, p_crossover: float, p_uni: float = 0.5, uniform: bool = False) -> tuple:
    '''This function applies crossover for the case of two parents.'''

    # Check if cross-over is applied
    if p_crossover > np.random.uniform():
        # Random uniform crossover
        if uniform:
            child_1 = []
            for gene in range(len(parent_1)):
                if p_uni > np.random.uniform():
                    # Choose first parent
                    child_1.append(parent_1[gene])
                else:
                    child_1.append(parent_2[gene])

            # The second child is used by using an inverse mapping,
            # We use the bit-flipping function defined above.
            child_2 = [bit_flipping(gene) for gene in child_1]

            return child_1, child_2

        # If no uniform crossover is selected, i.e. 1-point crossover is applied
        else:
            # We exclude the splitpoints in the beginning and the end
            split_point = randint(1, len(parent_1) - 1)

            # Now return perform the one-point crossover
            child_1 = np.array([parent_1[gene] if gene <= split_point else parent_2[gene]
                                for gene in range(len(parent_1))])
            child_2 = np.array([parent_2[gene] if gene <= split_point else parent_1[gene]
                                for gene in range(len(parent_1))])

            return child_1, child_2
    else:
        # Just returns the original parents
        return parent_1, parent_2


def selection_probabilities(generation, fitness_function: callable, sigma_scaling=False) -> list:
    '''
    Calculates the individual selection probabilities based on the fitness function.
    Applies sigma-scaling if desired.
    '''

    number_individuals = generation.shape[0]
    total_fitness = np.sum([fitness_function(generation[i]) for i in range(number_individuals)])

    if sigma_scaling == True:

        mean_fitness = total_fitness / number_individuals
        std_fitness = np.std([fitness_function(generation[i]) for i in range(number_individuals)])
        c = 2  # Constant

        fitness_sigma = [np.max(fitness_function(generation[i]) - (mean_fitness - (c * std_fitness)), 0) for i
                         in range(number_individuals)]

        # Now we need to sum up the sigma-scaled fitnesses
        total_fitness_sigma = np.sum(fitness_sigma)
        selection_prob = [fitness_sigma[i] / total_fitness_sigma for i in range(number_individuals)]
    else:
        # Apply normal inverse scaling
        selection_prob = [(fitness_function(generation[i]) / total_fitness) for i in range(number_individuals)]
    return selection_prob


def cumulative_probability_distribution(selection_probability: list) -> list:
    '''Calculates the cumulative probability distribution based on individual selection probabilities.'''
    cum_prob_distribution = []
    current_cum_prob_dis = 0
    for i in range(len(selection_probability)):
        current_cum_prob_dis += selection_probability[i]
        cum_prob_distribution.append(current_cum_prob_dis)
    return cum_prob_distribution


def roulette_wheel_algorithm(cum_prob_distribution, number_of_parents=2) -> list:
    '''
    Implements the roulette wheel algorithm as discussed in the
    accompanying text book by Eiben and Smith (2015).
    '''

    current_member = 1
    mating_pool = []
    while current_member <= number_of_parents:

        for _ in range(number_of_parents):
            # Generate a random number between 0 and 1
            r = np.random.uniform(0, 1)

            # Find the first individual whose cumulative probability exceeds the random number
            for i, cum_prob in enumerate(cum_prob_distribution):
                if r <= cum_prob:
                    mating_pool.append(i)
                    break

    return mating_pool


def tournament_selection(generation: list, fitness_function: callable, k: int) -> list:
    '''
    This implements the tournament selection. K random individual (with replacement) are
    chosen and compete with each other. The index of the best individual is returned.
    '''

    # First step: Choose a random individual and score it
    number_individuals = generation.shape[0]
    current_winner = randint(0, number_individuals)
    # Get the score which is the one to beat!
    score = fitness_function(current_winner)

    for candidates in range(k - 1):  # We already have one candidate, so we are left with k-1 to choose
        candidate = randint(0, number_individuals)
        candidate_score = fitness_function(candidate)

        # Update the current winner if the candidate has a better fitness (score)
        if candidate_score > score:
            current_winner = candidate
            score = candidate_score
    '''
    ToDo:

    Please try to finish the implementation of the roulette_wheel_algorithm.
    You will need the cum_prob_distribution function which is defined already above for you±
    '''

    return current_winner


# Now we can re-run the experiment from above, this time using tournament selection:

# Define the hyperparameters,
# following the recommendations presented in the textbook
# Eiben, A.E., Smith, J.E., Introduction to Evolutionary Computing., Springer, 2015, 2nd edition, page 100

# Define population size
n_population = 20

# Define length of the bitstring
bit_length = 50

# Define mutation rate
p_mutation = 1 / bit_length

# Crossover probability
p_crossover = 0.6

# Number of iterations
n_iter = 500

number_of_children = 2

# Tournament size
k = 5

# Initiliaze the generation
generation = init_population(n_population, bit_length)
best = fittest_solution(one_max, generation)
print('The current best solution in the initial generation is {0}'.format(best))

for i in range(1, n_iter + 1):

    new_generation = []
    selection_prob_gen = selection_probabilities(generation, one_max)
    cum_prob_distribution = cumulative_probability_distribution(selection_prob_gen)
    # First step: Parent selection using roulette wheel algorithm

    # We loop over the number of parent pairs we need to get
    for j in range(int(n_population / number_of_children)):

        mating_pool = []
        for child in range(number_of_children):
            mate = tournament_selection(generation, one_max, k)
            mating_pool.append(mate)

        # Cross-over
        child_1, child_2 = cross_over(generation[mating_pool[0]], generation[mating_pool[1]], p_crossover, uniform=True)

        # Mutation for each child
        child_1 = mutation_operator(bit_flipping, p_mutation, child_1)
        child_2 = mutation_operator(bit_flipping, p_mutation, child_2)

        # Survival selection is here generational, hence all children replace their parents

        new_generation.append(child_1.tolist())
        new_generation.append(child_2.tolist())

    generation = np.asarray(new_generation)
    best = fittest_solution(one_max, generation)
    if i % 10 == 0:
        print('The current best population in generation {0} is {1}'.format(i, best))

    # Include a condition that stops when the optimal solution is found
    if best == bit_length:
        print('---' * 20)
        print('Done! The algorithm has found the optimal solution!')
        print('The current best population in generation {0} is {1}'.format(i, best))
        break
