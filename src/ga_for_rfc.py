import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deap import creator, base, tools, algorithms
from utils import get_training_data, get_evaluation_data,\
                  get_testing_data, calculate_accuracy, normalise_data
from nn_manager import create_nn, call_nn
import fuckit


class RFCFeatureSelector(object):

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
        return np.random.randint(2, size=7)

    def codec_to_english(self, codec):
        translation = {
            "Pclass": codec[0],
            "Sex": codec[1],
            "Age": codec[2],
            "SibSp": codec[3],
            "Parch": codec[4],
            "Fare": codec[5],
            "Embarked": codec[6]
        }

        return translation

    def get_feature_name(self, index):
        translation = [
            "Pclass",
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
        expected_outputs = get_training_data().Survived.values
        train_data = self.massage_data_with_outputs(get_training_data(), ind)

        forest = RandomForestClassifier(n_estimators=1000,
                                        n_jobs=-1,
                                        criterion="entropy")

        forest.fit(train_data, expected_outputs)

        expected_eval_outputs = get_evaluation_data().Survived.values
        eval_data = self.massage_data_with_outputs(get_evaluation_data(), ind)

        forest = RandomForestClassifier(n_estimators=1000,
                                        n_jobs=-1,
                                        criterion="entropy")

        forest = forest.fit(train_data, expected_outputs)

        evaluation = forest.predict(eval_data)

        accuracy = calculate_accuracy(evaluation, expected_eval_outputs)
        print "Accuracy: {:10.4f}".format(accuracy)

        # Optional
        if accuracy > 0.8:
            self.early_solution = ind
            raise FoundEarlySolution

        return accuracy

    def massage_data_with_outputs(self, raw_data, individual):
        inputs = raw_data.drop([
                                    "Survived",
                                    "PassengerId",
                                    "Name",
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

    def massage_data_without_outputs(self, raw_data, individual):
        inputs = raw_data.drop([
                                    "PassengerId",
                                    "Name",
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
