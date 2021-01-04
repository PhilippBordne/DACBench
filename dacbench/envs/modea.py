# Most of this code is taken from the paper "Online CMA-ES Selection" by Vermetten et al.
# Github: https://github.com/Dvermetten/Online_CMA-ES_Selection

from modea.Algorithms import CustomizedES
import modea.Sampling as Sam
import modea.Mutation as Mut
import modea.Selection as Sel
import modea.Recombination as Rec
from dacbench import AbstractEnv
from modea.Utils import getOpts, getVals, options, initializable_parameters
from cma import bbobbenchmarks as bn
from functools import partial
import numpy as np


class ModeaEnv(AbstractEnv):
    def __init__(self, config):
        super(ModeaEnv, self).__init__(config)
        self.es = None
        self.budget = config.budget

    def reset(self):
        super(ModeaEnv, self).reset_()
        self.dim = self.instance[0]
        self.function_id = self.instance[1]
        self.instance_id = self.instance[2]
        self.representation = self.ensureFullLengthRepresentation(self.instance[3])

        opts = getOpts(self.representation[: len(options)])
        self.lambda_ = self.representation[len(options)]
        self.mu = self.representation[len(options) + 1]
        values = getVals(self.representation[len(options) + 2 :])

        self.function = bn.instantiate(int(self.function_id))[0]
        self.es = CustomizedES(
            self.dim, self.function, self.budget, self.mu, self.lambda_, opts, values
        )
        self.es.mutateParameters = self.es.parameters.adaptCovarianceMatrix
        self.update_parameters()
        return self.get_state()

    def step(self, action):
        done = super(ModeaEnv, self).step_()
        self.representation = self.ensureFullLengthRepresentation(action)
        opts = getOpts(self.representation[: len(options)])
        self.switchConfiguration(opts)

        # TODO: add ipop run (restarts)
        self.es.runOneGeneration()
        self.es.recordStatistics()

        if (
            self.es.budget <= self.es.used_budget
            or self.es.parameters.checkLocalRestartConditions(self.es.used_budget)
        ):
            self.restart()
            if self.es.total_used_budget < self.es.total_budget:
                self.update_parameters()
            else:
                done = True

        return self.get_state(), self.get_reward(), done, {}

    def update_parameters(self):
        # Every local restart needs its own parameters, so parameter update/mutation must also be linked every time
        self.es.parameters = Parameters(**parameter_opts)
        self.es.seq_cutoff = self.parameters.mu_int * self.parameters.seq_cutoff
        self.es.mutateParameters = self.parameters.adaptCovarianceMatrix

        self.es.initializePopulation()
        parameter_opts['wcm'] = self.es.population[0].genotype
        self.es.new_population = self.es.recombine(self.es.population, self.es.parameters)


    def restart(self):
        self.es.total_used_budget += self.es.used_budget
        if target is not None:
            if self.es.best_individual.fitness - target <= threshold:
                break
        # Increasing Population Strategies
        if parameter_opts['local_restart'] == 'IPOP':
            parameter_opts['lambda_'] *= 2

        elif parameter_opts['local_restart'] == 'BIPOP':
            try:
                self.es.budgets[self.es.regime] -= self.es.used_budget
                self.es.determineRegime()
            except KeyError:  # Setup of the two regimes after running regularily for the first time
                remaining_budget = self.total_budget - self.es.used_budget
                self.es.budgets['small'] = remaining_budget // 2
                self.es.budgets['large'] = remaining_budget - self.es.budgets['small']
                self.es.regime = 'large'

            if self.es.regime == 'large':
                self.es.lambda_['large'] *= 2
                parameter_opts['sigma'] = 2
            elif self.es.regime == 'small':
                rand_val = np.random.random() ** 2
                self.es.lambda_['small'] = int(floor(lambda_init * (.5 * self.es.lambda_['large'] / lambda_init) ** rand_val))
                parameter_opts['sigma'] = 2e-2 * np.random.random()

            self.es.budget = self.es.budgets[self.es.regime]
            self.es.used_budget = 0
            parameter_opts['budget'] = self.es.budget
            parameter_opts['lambda_'] = self.es.lambda_[self.es.regime]

    def get_state(self):
        return [
            self.es.gen_size,
            self.es.parameters.sigma,
            self.budget - self.es.used_budget,
            self.function_id,
            self.instance_id,
        ]

    def get_reward(self):
        return max(
            self.reward_range[0],
            min(self.reward_range[1], -self.es.best_individual.fitness),
        )

    def close(self):
        return True

    def switchConfiguration(self, opts):
        selector = Sel.pairwise if opts['selection'] == 'pairwise' else Sel.best

        def select(pop, new_pop, _, param):
            return selector(pop, new_pop, param)

        # Pick the lowest-level sampler
        if opts['base-sampler'] == 'quasi-sobol':
            sampler = Sam.QuasiGaussianSobolSampling(self.n)
        elif opts['base-sampler'] == 'quasi-halton' and Sam.halton_available:
            sampler = Sam.QuasiGaussianHaltonSampling(self.n)
        else:
            sampler = Sam.GaussianSampling(self.n)

        # Create an orthogonal sampler using the determined base_sampler
        if opts['orthogonal']:
            orth_lambda = self.parameters.eff_lambda
            if opts['mirrored']:
                orth_lambda = max(orth_lambda // 2, 1)
            sampler = Sam.OrthogonalSampling(self.n, lambda_=orth_lambda, base_sampler=sampler)

        # Create a mirrored sampler using the sampler (structure) chosen so far
        if opts['mirrored']:
            sampler = Sam.MirroredSampling(self.n, base_sampler=sampler)

        parameter_opts = {
            'weights_option': opts['weights_option'], 'active': opts['active'],
            'elitist': opts['elitist'],
            'sequential': opts['sequential'], 'tpa': opts['tpa'], 'local_restart': opts['ipop'],

        }

        # In case of pairwise selection, sequential evaluation may only stop after 2mu instead of mu individuals

        if opts['sequential'] and opts['selection'] == 'pairwise':
            parameter_opts['seq_cutoff'] = 2
            self.parameters.seq_cutoff = 2

        # Init all individuals of the first population at the same random point in the search space

        # We use functions/partials here to 'hide' the additional passing of parameters that are algorithm specific
        recombine = Rec.weighted
        mutate = partial(Mut.CMAMutation, sampler=sampler, threshold_convergence=opts['threshold'])

        functions = {
            'recombine': recombine,
            'mutate': mutate,
            'select': select,
            # 'mutateParameters': None
        }
        self.setConfigurationParameters(functions, parameter_opts)
        lambda_, eff_lambda, mu = self.calculateDependencies(opts, None, None)
        self.parameters.lambda_ = lambda_
        self.parameters.eff_lambda = eff_lambda
        self.parameters.mu = mu
        self.parameters.weights = self.parameters.getWeights(self.parameters.weights_option)
        self.parameters.mu_eff = 1 / sum(np.square(self.parameters.weights))
        mu_eff = self.parameters.mu_eff  # Local copy
        n = self.parameters.n
        self.parameters.c_sigma = (mu_eff + 2) / (mu_eff + n + 5)
        self.parameters.c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        self.parameters.c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        self.parameters.c_mu = min(1 - self.parameters.c_1, self.parameters.alpha_mu * (
                    (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + self.parameters.alpha_mu * mu_eff / 2)))
        self.parameters.damps = 1 + 2 * np.max([0, np.sqrt((mu_eff - 1) / (n + 1)) - 1]) + self.parameters.c_sigma
        self.seq_cutoff = self.parameters.mu_int * self.parameters.seq_cutoff

    def ensureFullLengthRepresentation(self, representation):
        """
        Given a (partial) representation, ensure that it is padded to become a full length customizedES representation,
        consisting of the required number of structure, population and parameter values.
        >>> ensureFullLengthRepresentation([])
        [0,0,0,0,0,0,0,0,0,0,0, None,None, None,None,None,None,None,None,None,None,None,None,None,None,None]
        :param representation:  List representation of a customizedES instance to check and pad if needed
        :return:                Guaranteed full-length version of the representation
        """
        default_rep = (
            [0] * len(options) + [None, None] + [None] * len(initializable_parameters)
        )
        if len(representation) < len(default_rep):
            representation = np.append(
                representation, default_rep[len(representation) :]
            ).flatten()
        return representation
