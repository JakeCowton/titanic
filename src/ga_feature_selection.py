import numpy as np
from deap import creator, base, tools, algorithms
import fuckit


class FeatureSelection(object):

    def __init__(self, data):
        # Massage data
        pass

    def setup_ga(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1., 1.))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Attribute generator
        self.toolbox.register("attr_gen",
                              self.individual_generator,
                              self.event_plan[:])

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
        return np.random.randint(2, size=11)

    def codec_to_english(codec):
        translation = {
            "Pclass": codec[0],
            "Sex": codec[1],
            "Age": codec[2],
            "SibSp": codec[3],
            "Parch": codec[4],
            "Ticket": codec[5],
            "Fare": codec[6],
            "Cabin": codec[7],
            "Embarked": codec[8]
        }

        return translation

    def get_feature_name(index):
        translation = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Cabin",
            "Embarked"
        ]
        return translation[index]

    def evaluate_ind(self, ind):
        pass

    def massage_data(raw_data, individual):

        outputs = raw_data.Survived.values

        inputs = raw_data.drop([
                                    "Survived",
                                    "PassengerId",
                                    "Name",
                                    "Ticket",
                                    "Cabin"
                               ], axis=1)

        for i in range(len(individual)):
            if individual[i] == 0:
                inputs.drop(get_feature_name(i))

        inputs = normalise_data(inputs)

        nn_data = np.zeros(800, dtype=[('inputs',  int, len(inputs[0])),
                                       ('outputs', int, 1)])

        nn_data['outputs'] = outputs
        nn_data["inputs"] = inputs

        return nn_data

    def normalise_data(data):
        out = []
        for sample in data.iterrows():
            # Sex
            if sample[1].Sex == "male": sample[1].Sex = 1
            else: sample[1].Sex = 0

            # Embarked
            if sample[1].Embarked == "C": sample[1].Embarked = 0
            elif sample[1].Embarked == "S": sample[1].Embarked = 1
            else: sample[1].Embarked = 2

            # Fare (ignore after the decimal)
            sample[1].Fare = int(sample[1].Fare)
            sample[1].Age = int(sample[1].Age)

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

        no_of_iterations = (self.no_of_events + 1) * 10

        if no_of_iterations > 200: no_of_iterations = 200
        try:
            algorithms.eaSimple(pop, self.toolbox, 0.7, 0.2,
                                no_of_iterations, stats=stats,
                                halloffame=hof, verbose=False)
        except FoundEarlySolution:
            return self.to_period(self.early_solution)

        return self.to_period(hof[0])
