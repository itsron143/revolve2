""" CMA-ES learner """
from copy import deepcopy

from database_components import Genotype

from revolve2.experimentation.evolution.abstract_elements import Learner, Evaluator

import multineat
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
                 initial_std: float, pop_size: int, bounds: list[float], seed):
        self._reward_function = evaluator
        self._options = {
            "popsize": pop_size,
            "bounds": bounds,
            "seed": seed,
            "verbose": -1
        }
        self._generations = generations

    def _extract_parameters(self, genotype: multineat.Genome) -> list[float]:
        """
        Extract the parameters from the genotype.
        :param genotype: The genotype.
        :return: The parameters.
        """
        sorted_links = sorted(genotype.LinkGenes, key=lambda x: x.InnovationID)
        parameters = [link.Weight for link in sorted_links]
        return parameters

    def _update_genome_with_parameters(self, genotype: multineat.Genome, new_params):
        # Get sorted indices of the original list
        sorted_indices = sorted(range(len(genotype.LinkGenes)),
                                key=lambda x: genotype.LinkGenes[x].InnovationID)

        # Update weights in the original list using sorted indices
        for i, idx in enumerate(sorted_indices):
            genotype.LinkGenes[idx].Weight = new_params[i]

    def learn(self, population: list[Genotype]) -> list[Genotype]:
        """
        Optimize the brain using CMA-ES algorithm

        :param population: The population of robots.
        :return: The population of robots after optimization.
        """
        logging.info(
            "Start brain learner optimization process for the population")

        for individual in population:
            # Get the brain genotype of the individual
            brain_genotype = individual.brain

            # Get the parameters from the brain genotype
            parameters = self._extract_parameters(brain_genotype)

            # Create the CMA-ES optimizer
            optimizer = cma.CMAEvolutionStrategy(
                parameters,
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
                    temp_brain_genotype = deepcopy(brain_genotype)
                    self._update_genome_with_parameters(
                        temp_brain_genotype, solution)
                    candidate_individual = Genotype(
                        body=individual.body,
                        brain=temp_brain_genotype
                    )
                    candidate_population.append(candidate_individual)

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

            logging.info(
                "Optimization completed for an individual with best fitness: "
                f"{best_fitness} and parameters: {best_solution}")

            # update the brain genotype of the original individual with the best solution
            self._update_genome_with_parameters(brain_genotype, best_solution)
            # update the fitness of the original individual

        logging.info("End brain learner optimization process.")

        return population
