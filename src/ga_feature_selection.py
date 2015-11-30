import numpy as np
from deap import creator, base, tools, algorithms
from utils import get_training_data, get_evaluation_data,\
                  get_testing_data, calculate_accuracy
from nn_manager import create_nn, call_nn


class FeatureSelector(object):

    def __init__(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1., 1.))
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
        test_data = self.massage_data_with_outputs(get_training_data(), ind)
        nn = create_nn(test_data, (no_of_inputs, 3, 3, 1))

        eval_data = self.massage_data_with_outputs(get_evaluation_data(), ind)

        evaluation = []
        for sample in eval_data:
            out = call_nn(nn, sample[0])
            if out >= 0.5:
                evaluation.append(1)
            else:
                evaluation.append(0)

        accuracy = calculate_accuracy(evaluation)

        # # Optional
        # if accuracy > 0.8:
        #     raise FoundEarlySolution

        return accuracy

    def massage_data_with_outputs(self, raw_data, individual):
        outputs = raw_data.Survived.values

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

        inputs = self.normalise_data(inputs)

        nn_data = np.zeros(len(raw_data),
                           dtype=[('inputs',  int, len(inputs[0])),
                                  ('outputs', int, 1)])

        nn_data['outputs'] = outputs
        nn_data["inputs"] = inputs

        return nn_data

    def massage_data_without_outputs(self, raw_data, individual):
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

        inputs = self.normalise_data(inputs)

        return inputs

    def normalise_data(self, data):
        out = []

        try:
            data["Age"] = data["Age"].fillna(data["Age"].median())
        except KeyError:
            # This means it was dropped because of the GA
            pass

        for sample in data.iterrows():
            # Sex
            try:
                if sample[1].Sex == "male": sample[1].Sex = 1.0
                else: sample[1].Sex = 0.0
            except AttributeError:
                # This means it was dropped because of the GA
                pass
            # Embarked
            try:
                if sample[1].Embarked == "C": sample[1].Embarked = 0.0
                elif sample[1].Embarked == "S": sample[1].Embarked = 1.0
                else: sample[1].Embarked = 2.0

            except AttributeError:
                # This means it was dropped because of the GA
                pass
            # Fare (ignore after the decimal)
            try:
                fare = float(sample[1].Fare)

            except AttributeError:
                # This means it was dropped because of the GA
                pass
            try:
                sample[1].Age = float(sample[1].Age)
            except AttributeError:
                # This means it was dropped because of the GA
                pass

            out.append(sample[1].values)

        return out

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
            return hof[0]

        return hof[0]

class FoundEarlySolution(Exception):
    pass
