""" CMA-ES learner """
from copy import deepcopy

from database_components import Genotype

from revolve2.experimentation.evolution.abstract_elements import Learner, Evaluator
from revolve2.modular_robot import ModularRobot

import cma
import logging
import numpy as np


class CMAESLearner(Learner):

    _reward_function: Evaluator
    _options: cma.CMAOptions
    _generations: int

    def __init__(self,
                 evaluator: Evaluator,
                 generations: int,
                 pop_size: int,
                 seed):
        self._reward_function = evaluator
        self._options = {
            "popsize": pop_size,
            "seed": seed,
            "verbose": -1
        }
        self._generations = generations

    def learn(self, population: list[Genotype]) -> tuple[list[Genotype], list[float]]:
        """
        Optimize the brain using CMA-ES algorithm

        :param population: The population of robots.
        :return: The population of robots after optimization.
        """
        logging.info(
            "Start brain learner optimization process for the population")

        individual_index = 0
        child_task_performance = []
        for individual in population:
            logging.info(f"CMAES-Learner Individual: {individual_index + 1}")

            robot = individual.develop()

            brain_weights = robot.brain._weight_matrix.copy()
            flat_brain_weights = brain_weights.flatten()

            if flat_brain_weights.shape == (0,):
                child_task_performance.append(0.0)
                continue
            # Create the CMA-ES optimizer
            optimizer = cma.CMAEvolutionStrategy(
                flat_brain_weights,
                0.5,
                self._options
            )

            generation_index = 0
            while generation_index < self._generations:
                logging.info(
                    f"CMAES-Learner Gen: {generation_index + 1} / {self._generations}.")

                # Get the sampled solutions(parameters) from cma.
                candidate_solutions = optimizer.ask()
                candidate_population = []
                # Create a population of individuals from the candidate solutions
                # The body genotype is the same as the original individual
                # the brain genotype is updated for each individual using the solution
                for solution in candidate_solutions:
                    temp_robot_body = deepcopy(robot.body)
                    temp_robot_brain = deepcopy(robot.brain)
                    temp_robot_brain._weight_matrix = solution.reshape(
                        brain_weights.shape)
                    candidate_robot = ModularRobot(
                        temp_robot_body, temp_robot_brain)
                    candidate_population.append(candidate_robot)

                # Evalue the individual with the reward function
                # The fitness is the negative of the reward function as CMA-ES minimizes the objective function.
                fitness = self._reward_function.evaluate(candidate_population)
                np_fitness = -np.array(fitness)

                # Update the optimizer
                optimizer.tell(candidate_solutions, np_fitness)

                # Increase the generation index counter.
                generation_index += 1

            # Get the best solution from the optimizer
            best_solution = optimizer.result.xbest
            best_fitness = optimizer.result.fbest

            child_task_performance.append(-best_fitness)

            logging.info(
                f"Optimization completed for an individual {individual_index + 1} with best fitness: " +
                f"{best_fitness}")

            # update the fitness of the original individual
            individual_index += 1

        logging.info("End brain learner optimization process.")

        return population, child_task_performance
