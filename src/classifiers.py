import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import write_results, get_training_data,\
                  get_evaluation_data, get_testing_data,\
                  get_all_training_data, normalise_data
from evaluation import EvaluationMetrics
from slp import create_slp
from nn_manager import create_nn, call_nn
from ga_for_mlp import MLPFeatureSelector
from ga_for_rfc import RFCFeatureSelector
from ga_for_svm import SVMFeatureSelector
from sklearn import svm


K_FOLDS = 10

def random_forest():
    test_df = get_testing_data()
    ids = test_df.PassengerId.values
    # Drop all but class
    test_df = test_df.drop(["PassengerId", "Ticket", "Cabin",],
                             axis=1)
    test_data = normalise_data(test_df).values

    f_scores = []

    for i in range(K_FOLDS):
        train_df = get_training_data(fold=i)
        eval_df = get_evaluation_data(fold=i)

        expected_training_outputs = train_df.Survived.values
        train_df = train_df.drop(["PassengerId", "Survived",
                                  "Ticket", "Cabin"],
                                  axis=1)

        expected_eval_outputs = eval_df.Survived.values
        eval_df = eval_df.drop(["PassengerId", "Survived",\
                                "Ticket", "Cabin"],
                                axis=1)

        train_data = normalise_data(train_df).values
        eval_data = normalise_data(eval_df).values


        forest = RandomForestClassifier(n_estimators=200,
                                        n_jobs=-1,
                                        criterion="entropy")

        forest = forest.fit(train_data, expected_training_outputs)


        evaluation = forest.predict(eval_data)

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        f1 = em.calculate_f1()
        f_scores.append(f1)


    output = forest.predict(test_data)

    write_results("rand_forest_entropy.csv", ids, output)


    return f_scores

def slp():
    test_df = get_testing_data()
    ids = test_df.PassengerId.values
    test_df = test_df.drop(["PassengerId", "Ticket", "Cabin",],
                             axis=1)
    test_data = normalise_data(test_df).values

    f_scores = []

    for i in range(K_FOLDS):

        train_df = get_training_data(fold=i)
        eval_df = get_evaluation_data(fold=i)

        expected_training_outputs = train_df.Survived.values
        train_df = train_df.drop(["PassengerId", "Survived",
                                  "Ticket", "Cabin"],
                                  axis=1)

        expected_eval_outputs = eval_df.Survived.values
        eval_df = eval_df.drop(["PassengerId", "Survived",\
                                "Ticket", "Cabin"],
                                axis=1)

        train_data = normalise_data(train_df).values
        eval_data = normalise_data(eval_df).values

        perceptron = create_slp(train_data, expected_training_outputs)

        evaluation = []
        for sample in eval_data:
            evaluation.append(perceptron.recall(sample))

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        f1 = em.calculate_f1()
        f_scores.append(f1)

    output = []
    for sample in test_data:
        output.append(perceptron.recall(sample))

    write_results("slp.csv", ids, output)


    return f_scores

def mlp():
    test_df = get_testing_data()
    ids = test_df.PassengerId.values
    test_df = test_df.drop(["PassengerId", "Ticket", "Cabin",],
                             axis=1)
    test_data = normalise_data(test_df).values

    f_scores = []

    for i in range(K_FOLDS):

        train_df = get_training_data(fold=i)
        eval_df = get_evaluation_data(fold=i)

        expected_training_outputs = train_df.Survived.values
        train_df = train_df.drop(["PassengerId", "Survived",
                                  "Ticket", "Cabin"],
                                  axis=1)

        expected_eval_outputs = eval_df.Survived.values
        eval_df = eval_df.drop(["PassengerId", "Survived",\
                                "Ticket", "Cabin"],
                                axis=1)

        train_data = normalise_data(train_df).values
        eval_data = normalise_data(eval_df).values

        no_of_inputs = len(train_data[0])
        no_of_samples = len(train_data)

        data = np.zeros(no_of_samples,dtype=[('inputs',  float, no_of_inputs),
                                             ('outputs', float, 1)])

        for i in range(len(train_data)):
            data[i]['inputs'] = train_data[i]
            data[i]['outputs'] = expected_training_outputs[i]

        nn = create_nn(data, (no_of_inputs,100,1))

        evaluation = []
        for sample in eval_data:
            out = call_nn(nn, sample[0])
            if out >= 0.5:
                evaluation.append(1)
            else:
                evaluation.append(0)

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        f1 = em.calculate_f1()
        f_scores.append(f1)

    output = []
    for sample in test_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            output.append(1)
        else:
            output.append(0)

    write_results("mlp.csv", ids, output)

    return f_scores

def sk_svm():
    test_df = get_testing_data()
    ids = test_df.PassengerId.values
    test_df = test_df.drop(["PassengerId", "Ticket", "Cabin",],
                             axis=1)
    test_data = normalise_data(test_df).values

    f_scores = []

    for i in range(K_FOLDS):
        train_df = get_training_data(fold=i)
        eval_df = get_evaluation_data(fold=i)

        expected_training_outputs = train_df.Survived.values
        train_df = train_df.drop(["PassengerId", "Survived",
                                  "Ticket", "Cabin"],
                                  axis=1)

        expected_eval_outputs = eval_df.Survived.values
        eval_df = eval_df.drop(["PassengerId", "Survived",\
                                "Ticket", "Cabin"],
                                axis=1)

        train_data = normalise_data(train_df).values
        eval_data = normalise_data(eval_df).values

        clf = svm.LinearSVC()
        clf.fit(train_data, expected_training_outputs)

        evaluation = clf.predict(eval_data)

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        f1 = em.calculate_f1()
        f_scores.append(f1)

    output = clf.predict(test_data)

    write_results("svm.csv", ids, output)

    return f_scores

def ga_mlp():
    test_df = get_testing_data()
    ids = test_df.PassengerId.values

    ga = MLPFeatureSelector()
    features = ga.calculate()

    print features

    test_data = ga.massage_data_without_outputs(test_df, features)

    f_scores = []

    for i in range(K_FOLDS):
        train_df = get_training_data(fold=i)
        eval_df = get_evaluation_data(fold=i)

        expected_training_outputs = train_df.Survived.values
        train_data = ga.massage_data_with_outputs(train_df, features)

        expected_eval_outputs = eval_df.Survived.values
        eval_data = ga.massage_data_with_outputs(eval_df, features)

        no_of_inputs = features.count(1)

        nn = create_nn(train_data, (no_of_inputs, 3, 1))

        evaluation = []
        for sample in eval_data:
            out = call_nn(nn, sample[0])
            if out >= 0.5:
                evaluation.append(1)
            else:
                evaluation.append(0)

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        f1 = em.calculate_f1()
        f_scores.append(f1)

    output = []
    for sample in test_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            output.append(1)
        else:
            output.append(0)

    write_results("ga_mlp.csv", ids, output)

    return f_scores

def ga_rfc():
    test_df = get_testing_data()
    ids = test_df.PassengerId.values

    ga = RFCFeatureSelector()
    features = ga.calculate()

    print features

    test_data = ga.massage_data_without_outputs(test_df, features)

    f_scores = []

    for i in range(K_FOLDS):
        train_df = get_training_data(fold=i)
        eval_df = get_evaluation_data(fold=i)

        expected_training_outputs = train_df.Survived.values
        train_data = ga.massage_data_with_outputs(train_df, features)

        expected_eval_outputs = eval_df.Survived.values
        eval_data = ga.massage_data_with_outputs(eval_df, features)

        no_of_inputs = features.count(1)

        forest = RandomForestClassifier(n_estimators=1000,
                                        n_jobs=-1,
                                        criterion="entropy")

        forest = forest.fit(train_data, expected_training_outputs)

        evaluation = forest.predict(eval_data)

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        f1 = em.calculate_f1()
        f_scores.append(f1)

    output = forest.predict(test_data)
    write_results("ga_rfc.csv", ids, output)

    return f_scores

def ga_svm():
    test_df = get_testing_data()
    ids = test_df.PassengerId.values

    ga = SVMFeatureSelector()
    features = ga.calculate()

    print features

    test_data = ga.massage_data_without_outputs(test_df, features)

    f_scores = []

    for i in range(K_FOLDS):
        train_df = get_training_data(fold=i)
        eval_df = get_evaluation_data(fold=i)

        expected_training_outputs = train_df.Survived.values
        train_data = ga.massage_data_with_outputs(train_df, features)

        expected_eval_outputs = eval_df.Survived.values
        eval_data = ga.massage_data_with_outputs(eval_df, features)

        no_of_inputs = features.count(1)

        clf = svm.SVC()

        clf = clf.fit(train_data, expected_training_outputs)

        evaluation = clf.predict(eval_data)

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        f1 = em.calculate_f1()
        f_scores.append(f1)

    output = clf.predict(test_data)
    write_results("ga_rfc.csv", ids, output)

    return f_scores
