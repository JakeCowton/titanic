import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import write_results, get_training_data,\
                  get_evaluation_data, get_testing_data,\
                  calculate_accuracy, get_all_training_data,\
                  normalise_data
from slp import create_slp
from nn_manager import create_nn, call_nn
from ga_feature_selection import FeatureSelector


def random_forest():
    """
    A random forest classifier
    Inputs: Pclass
    Outputs: Survived
    """

    print "Random Forest Classifier"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()

    ids = test_df.PassengerId.values

    print "Massaging data..."

    expected_training_outputs = train_df.Survived.values
    train_df = train_df.drop(["PassengerId", "Survived", "Name",
                              "Ticket", "Cabin"],
                              axis=1)

    expected_eval_outputs = eval_df.Survived.values
    eval_df = eval_df.drop(["PassengerId", "Survived", "Name",\
                            "Ticket", "Cabin"],
                            axis=1)

    # Drop all but class
    test_df = test_df.drop(["PassengerId", "Name", "Ticket", "Cabin",],
                             axis=1)

    train_data = normalise_data(train_df).values
    eval_data = normalise_data(eval_df).values
    test_data = normalise_data(test_df).values

    print "Training... (using entropy)"
    forest = RandomForestClassifier(n_estimators=1000,
                                    n_jobs=-1,
                                    criterion="entropy")

    forest = forest.fit(train_data, expected_training_outputs)

    print "Evaluating..."
    evaluation = forest.predict(eval_data)

    print "Accuracy: {:10.4f}".format(calculate_accuracy(evaluation))

    print "Predicting..."
    output = forest.predict(test_data)

    print "Writing results..."
    write_results("rand_forest_entropy.csv", ids, output)

    print "--- Done ---"

    return True

def slp():
    """
    A single layer perceptron
    Inputs: Pclass
    Outputs: Survived
    """

    print "SLP"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()

    ids = test_df.PassengerId.values

    print "Massaging data..."

    expected_training_outputs = train_df.Survived.values
    train_df = train_df.drop(["PassengerId", "Survived", "Name",
                              "Ticket", "Cabin"],
                              axis=1)

    expected_eval_outputs = eval_df.Survived.values
    eval_df = eval_df.drop(["PassengerId", "Survived", "Name",\
                            "Ticket", "Cabin"],
                            axis=1)

    # Drop all but class
    test_df = test_df.drop(["PassengerId", "Name", "Ticket", "Cabin",],
                             axis=1)

    train_data = normalise_data(train_df).values
    eval_data = normalise_data(eval_df).values
    test_data = normalise_data(test_df).values

    print "Training..."

    perceptron = create_slp(train_data, expected_training_outputs)

    print "Evaluating..."

    evaluation = []
    for sample in eval_data:
        evaluation.append(perceptron.recall(sample))

    print "Accuracy: {:10.4f}".format(calculate_accuracy(evaluation))

    print "Predicting..."

    output = []
    for sample in test_data:
        output.append(perceptron.recall(sample))

    print "Writing results..."
    write_results("slp.csv", ids, output)

    print "--- Done ---"

    return True

def mlp():
    """
    Multi-layer perceptron
    Inputs: Pclass
    Outputs: Survived
    """
    print "MLP"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()

    ids = test_df.PassengerId.values

    print "Massaging data..."

    # Drop all but class and survived
    train_df = train_df.drop(["PassengerId",
                              "Name", "Sex", "Age", "SibSp", "Parch",
                              "Ticket", "Fare", "Cabin", "Embarked"],
                              axis=1)
    # Drop all but class
    eval_df = eval_df.drop(["PassengerId", "Survived",
                            "Name", "Sex", "Age", "SibSp", "Parch",
                            "Ticket", "Fare", "Cabin", "Embarked"],
                            axis=1)

    # Drop all but class
    test_df = test_df.drop(["PassengerId",
                             "Name", "Sex", "Age", "SibSp", "Parch",
                             "Ticket", "Fare", "Cabin", "Embarked"],
                             axis=1)

    train_data = train_df.values
    eval_data = eval_df.values
    test_data = test_df.values

    data = np.zeros(800, dtype=[('inputs',  int, 1),
                                ('outputs', int, 1)])

    for i in range(len(train_data)):
        data[i]['inputs'] = train_data[i][1]
        data[i]['outputs'] = train_data[i][0]

    print "Training..."
    nn = create_nn(data, (1,2,1))


    print "Predicting..."
    evaluation = []
    for sample in eval_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            evaluation.append(1)
        else:
            evaluation.append(0)

    print "Accuracy: {:10.4f}".format(calculate_accuracy(evaluation))

    output = []
    for sample in test_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            output.append(1)
        else:
            output.append(0)

    print "Writing results..."
    write_results("mlp.csv", ids, output)

    print "--- Done ---"

    return True

def ga_mlp():
    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()
    all_train_df = get_all_training_data()

    ids = test_df.PassengerId.values

    ga = FeatureSelector()
    features = ga.calculate()

    train_data = ga.massage_data_with_outputs(train_df, features)
    eval_data = ga.massage_data_with_outputs(eval_df, features)
    test_data = ga.massage_data_without_outputs(test_df, features)

    no_of_inputs = features.count(1)
    nn = create_nn(train_data, (no_of_inputs, 3, 1))

    evaluation = []
    for sample in eval_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            evaluation.append(1)
        else:
            evaluation.append(0)

    print "Accuracy: {:10.4f}".format(calculate_accuracy(evaluation))

    all_train_df = get_all_training_data()
    all_train_data = ga.massage_data_with_outputs(all_train_df, features)
    nn = create_nn(all_train_data, (no_of_inputs, 3, 1))

    output = []
    for sample in test_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            output.append(1)
        else:
            output.append(0)

    print "Writing results..."
    write_results("ga_mlp.csv", ids, output)

    print "--- Done ---"

    return True

def ga_rfc():
    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()
    all_train_df = get_all_training_data()

    ids = test_df.PassengerId.values

    ga = FeatureSelector()
    features = ga.calculate()

    train_data = ga.massage_data_with_outputs(train_df, features)
    eval_data = ga.massage_data_with_outputs(eval_df, features)
    test_data = ga.massage_data_without_outputs(test_df, features)

    no_of_inputs = features.count(1)
    nn = create_nn(train_data, (no_of_inputs, 3, 1))

    evaluation = []
    for sample in eval_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            evaluation.append(1)
        else:
            evaluation.append(0)

    print "Accuracy: {:10.4f}".format(calculate_accuracy(evaluation))

    all_train_df = get_all_training_data()
    all_train_data = ga.massage_data_with_outputs(all_train_df, features)
    nn = create_nn(all_train_data, (no_of_inputs, 3, 1))

    inputs = all_train_data[0::,1::]
    expected_outputs = all_train_data[0::,0]
    forest = forest.fit(inputs, expected_outputs)

    output = forest.predict(test_data)

    print "Writing results..."
    write_results("ga_rfc.csv", ids, output)

    print "--- Done ---"

    return True