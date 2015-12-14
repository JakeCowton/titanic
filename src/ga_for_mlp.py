import numpy as np
from deap import creator, base, tools, algorithms
from utils import get_training_data, get_evaluation_data,\
                  get_testing_data, normalise_data
from evaluation import EvaluationMetrics
from nn_manager import create_nn, call_nn
import fuckit


class MLPFeatureSelector(object):

    def __init__(self):
        creator.create("FitnessMulti", base.Fitness, weights=(1.,))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Attribute generator
        self.toolbox.register("attr_gen",
                              self.individual_generator)

        # Structure initialisers
        self.toolbox.register("individual",
                              tools.initIterate,
                              creator.Individual,
                              self.toolbox.attr_gen)
        self.toolbox.register("population",
                              tools.initRepeat,
                              list,
                              self.toolbox.individual)

        # Register the toolbox functions
        self.register_functions()

        self.early_solution = None

    def register_functions(self):
        """
        Register the following functions:
            Selection: selTournament
            Mutation: mutShuffleIndexes
            Crossover: cxPartialyMatched
        """

        # Functions
        self.toolbox.register("evaluate", self.evaluate_ind)
        self.toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def individual_generator(self):
        return np.random.randint(2, size=8)

    def codec_to_english(self, codec):
        translation = {
            "Pclass": codec[0],
            "Name": code[1],
            "Sex": codec[2],
            "Age": codec[3],
            "SibSp": codec[4],
            "Parch": codec[5],
            "Fare": codec[6],
            "Embarked": codec[7]
        }

        return translation

    def get_feature_name(self, index):
        translation = [
            "Pclass",
            "Name"
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]
        return translation[index]

    def evaluate_ind(self, ind):
        no_of_inputs = ind.count(1)

        if no_of_inputs <= 1:
            return 0.0

        test_data = self.massage_data_with_outputs(get_training_data(), ind)
        nn = create_nn(test_data, (no_of_inputs, 3, 1))

        expected_eval_outputs = get_evaluation_data().Survived.values
        eval_data = self.massage_data_with_outputs(get_evaluation_data(), ind)

        evaluation = []
        for sample in eval_data:
            out = call_nn(nn, sample[0])
            if out >= 0.5:
                evaluation.append(1)
            else:
                evaluation.append(0)

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        f1 = em.calculate_f1()
        print "Accuracy: " + str(em.calculate_accuracy())
        print "Precision:" + str(em.calculate_precision())
        print "Recall: " + str(em.calculate_recall())
        print "F1 measure:" + str(f1)

        # Optional
        if f1 > 0.74:
            self.early_solution = ind
            raise FoundEarlySolution

        return f1

    def massage_data_with_outputs(self, raw_data, individual):
        outputs = raw_data.Survived.values

        inputs = raw_data.drop([
                                    "Survived",
                                    "PassengerId",
                                    "Ticket",
                                    "Cabin"
                               ], axis=1)
        inputs_to_drop = []
        for i in range(len(individual)):
            if individual[i] == 0:
                inputs_to_drop.append(self.get_feature_name(i))

        inputs = inputs.drop(inputs_to_drop, axis=1)

        inputs = normalise_data(inputs).values

        no_of_inputs = len(inputs[0])

        nn_data = np.zeros(len(raw_data),
                           dtype=[('inputs',  float, no_of_inputs),
                                  ('outputs', float, 1)])

        nn_data['outputs'] = outputs
        nn_data["inputs"] = inputs

        return nn_data

    def massage_data_without_outputs(self, raw_data, individual):
        inputs = raw_data.drop([
                                    "PassengerId",
                                    "Ticket",
                                    "Cabin"
                               ], axis=1)

        inputs_to_drop = []
        for i in range(len(individual)):
            if individual[i] == 0:
                inputs_to_drop.append(self.get_feature_name(i))

        inputs = inputs.drop(inputs_to_drop, axis=1)

        inputs = normalise_data(inputs).values

        return inputs

    def calculate(self):
        pop = self.toolbox.population(n=300)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        no_of_iterations = 200

        try:
            algorithms.eaSimple(pop, self.toolbox, 0.7, 0.2,
                                no_of_iterations, stats=stats,
                                halloffame=hof, verbose=False)
        except FoundEarlySolution:
            return self.early_solution

        return hof[0]

class FoundEarlySolution(Exception):
    pass
