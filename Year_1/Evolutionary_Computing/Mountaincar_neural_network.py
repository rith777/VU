import numpy as np
import gymnasium as gym
from IPython.display import clear_output
import os


env = gym.make('MountainCar-v0', render_mode='human')

observation = env.reset()

for _ in range(200):
    action = env.action_space.sample ()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated:
        env.reset()

env.close()

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

class Perceptron:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def activation(self, input, weights):
        x = np.dot(input, weights.reshape (self.n_inputs, self.n_outputs))
        return np.argmax(x)

individual = np.array ([0.1, -0.1, 0, 0.5, -0.5, 0.3], dtype=np.float64)

network = Perceptron (2, 3)


env = gym.make("MountainCar-v0", render_mode="human")

observation, info = env.reset()

for _ in range(200):
    action = network.activation (observation, individual)

    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated:
        break

env.close()

def evaluate(weights, net, env):
    observation, info = env.reset()
    total_reward = 0

    for _ in range(1000):
        action = net.activation(observation, weights)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated:
            observation = env.reset()
            break

    return total_reward


env = gym.make("MountainCar-v0")
print (evaluate (individual, network, env))
env.close()

def initialize_population(population_size, lower, upper, n_weights = 6):
    return np.random.uniform(lower, upper, (population_size, n_weights))

pop = initialize_population(10, -2, 2)
pop

def evaluate_population(pop, number_of_evaluations, net, env):
    population_fitness = np.zeros(pop.shape[0])

    for i in range (pop.shape [0]):
        population_fitness[i] = np.mean ([evaluate (pop[i], net, env) for _ in range (number_of_evaluations)])

    return population_fitness

env = gym.make("MountainCar-v0")
print (evaluate_population(pop, 10, network, env))
env.close()


def parent_selection(pop, pop_fit, n_parents, smoothing = 1):
    fitness  = pop_fit + smoothing - np.min(pop_fit)

    # Fitness proportional selection probability
    fps = fitness / np.sum (fitness)

    # make a random selection of indices
    parent_indices = np.random.choice (np.arange(0,pop.shape[0]), (n_parents,2), p=fps)
    return pop [parent_indices]

def crossover(parents):
    parentsA, parentsB = np.hsplit (parents,2)
    roll = np.random.uniform (size = parentsA.shape)
    offspring = parentsA * (roll >= 0.5) + parentsB * (roll < 0.5)
    # squeeze to get rid of the extra dimension created during parent selecting
    return np.squeeze(offspring,1)

def mutate(pop,min_value,max_value, sigma):
    mutation = np.random.normal(0, sigma, size=pop.shape)
    new_pop = pop + mutation
    new_pop[new_pop>max_value] = max_value
    new_pop[new_pop<min_value] = min_value
    return new_pop

def survivor_selection(pop, pop_fit, n_pop):
    best_fit_indices = np.argsort(pop_fit * -1) # -1 since we are maximizing
    survivor_indices = best_fit_indices [:n_pop]
    return pop [survivor_indices], pop_fit[survivor_indices]

# Parameters
population_size = 50
n_evaluations = 3
n_offspring = 50
weight_upper_bound = 2
weight_lower_bound = -2
mutation_sigma = .1
generations = 10

# Initialize environment, network and population. Perform an initial evaluation
env = gym.make("MountainCar-v0")
net = Perceptron (2, 3)
pop = initialize_population(population_size, weight_lower_bound, weight_upper_bound)
pop_fit = evaluate_population(pop, n_evaluations, net, env)

for i in range (generations):
    parents = parent_selection(pop, pop_fit, n_offspring)
    offspring = crossover (parents)
    offspring = mutate (offspring, weight_lower_bound, weight_upper_bound, mutation_sigma)

    offspring_fit = evaluate_population(offspring, n_evaluations, net, env)

    # concatenating to form a new population
    pop = np.vstack((pop,offspring))
    pop_fit = np.concatenate([pop_fit,offspring_fit])

    pop, pop_fit = survivor_selection(pop, pop_fit, population_size)

    print (f"Gen {i} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")
    clear_output(wait=True)
env.close()

individual = pop [np.argmax(pop_fit)]

network = Perceptron (2, 3)


env = gym.make("MountainCar-v0", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = network.activation (observation, individual)
    observation, reward, done, truncated, info = env.step(action)
    env.render()

    if done:
        break

env.close()

class DataGatherer:

    def __init__(self, name):
        self.name = name
        self.mean_fitness = np.array ([])
        self.best_fitness = np.array ([])
        self.generations = np.array ([])
        self.stats = []

        if not os.path.exists(name):
            os.mkdir(name)
            os.mkdir(name+"/best")

    def gather(self, pop, pop_fit, gen):
        self.mean_fitness = np.concatenate([self.mean_fitness, [np.mean (pop_fit)]])
        self.best_fitness = np.concatenate([self.best_fitness, [np.max (pop_fit)]])
        self.generations = np.concatenate([self.generations, [gen]])

        self.stats = np.stack([self.generations, self.mean_fitness,self.best_fitness])

        np.savetxt(f"{self.name}/stats.out", self.stats.T, delimiter=',',fmt='%1.2e')
        np.savetxt(f"{self.name}/best/{gen}.out", pop [np.argmax(pop_fit)],  delimiter=',',fmt='%1.2e')

# Parameters
population_size = 100
n_evaluations = 3
n_offspring = 100
weight_upper_bound = 2
weight_lower_bound = -2
mutation_sigma = .1
generations = 10

# Initialize environment, network and population. Perform an initial evaluation
env = gym.make("MountainCar-v0")
net = Perceptron (2, 3)
pop = initialize_population(population_size, weight_lower_bound, weight_upper_bound)
pop_fit = evaluate_population(pop, n_evaluations, net, env)
data = DataGatherer ("simple_example") # think of a good naming convention

for gen in range (generations):
    parents = parent_selection(pop, pop_fit, n_offspring)
    offspring = crossover (parents)
    offspring = mutate (offspring, weight_lower_bound, weight_upper_bound, mutation_sigma)

    offspring_fit = evaluate_population(offspring, n_evaluations, net, env)

    # concatenating to form a new population
    pop = np.vstack((pop,offspring))
    pop_fit = np.concatenate([pop_fit,offspring_fit])

    pop, pop_fit = survivor_selection(pop, pop_fit, population_size)

    data.gather (pop, pop_fit, gen)
    print (f"Gen {gen} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")
    clear_output(wait=True)

import plotly.express as px
import pandas as pd

raw_data = np.loadtxt("simple_example/stats.out",delimiter=",")
data_plot = pd.DataFrame(raw_data, columns=["Generation", "Mean", "Best"])
px.line (data_plot, x="Generation", y=["Mean", "Best"], labels={"value": "Performance"})