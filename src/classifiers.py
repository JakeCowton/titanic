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
from sklearn import svm


K_FOLDS = 10

def random_forest():

    print "--- Random Forest Classifier ---"

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

        print "Massaging data..."

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

        print "Training..."
        forest = RandomForestClassifier(n_estimators=200,
                                        n_jobs=-1,
                                        criterion="entropy")

        forest = forest.fit(train_data, expected_training_outputs)

        print "Evaluating..."
        evaluation = forest.predict(eval_data)

        em = EvaluationMetrics(evaluation, expected_eval_outputs)
        print "Accuracy: " + str(em.calculate_accuracy())
        print "Precision:" + str(em.calculate_precision())
        print "Recall: " + str(em.calculate_recall())
        f1 = em.calculate_f1()
        f_scores.append(f1)
        print "F1 measure:" + str(f1)

    print "Predicting..."
    output = forest.predict(test_data)

    print "Writing results..."
    write_results("rand_forest_entropy.csv", ids, output)

    print "Done"

    return f_scores

def slp():

    print "--- SLP ---"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()

    ids = test_df.PassengerId.values

    print "Massaging data..."

    expected_training_outputs = train_df.Survived.values
    train_df = train_df.drop(["PassengerId", "Survived",
                              "Ticket", "Cabin"],
                              axis=1)

    expected_eval_outputs = eval_df.Survived.values
    eval_df = eval_df.drop(["PassengerId", "Survived",\
                            "Ticket", "Cabin"],
                            axis=1)

    # Drop all but class
    test_df = test_df.drop(["PassengerId", "Ticket", "Cabin",],
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

    em = EvaluationMetrics(evaluation, expected_eval_outputs)
    print "Accuracy: " + str(em.calculate_accuracy())
    print "Precision:" + str(em.calculate_precision())
    print "Recall: " + str(em.calculate_recall())
    print "F1 measure:" + str(em.calculate_f1())

    print "Predicting..."

    output = []
    for sample in test_data:
        output.append(perceptron.recall(sample))

    print "Writing results..."
    write_results("slp.csv", ids, output)

    print "Done"

    return True

def mlp():
    print "--- MLP ---"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()

    ids = test_df.PassengerId.values

    print "Massaging data..."

    expected_training_outputs = train_df.Survived.values
    train_df = train_df.drop(["PassengerId", "Survived",
                              "Ticket", "Cabin"],
                              axis=1)

    expected_eval_outputs = eval_df.Survived.values
    eval_df = eval_df.drop(["PassengerId", "Survived",\
                            "Ticket", "Cabin"],
                            axis=1)

    # Drop all but class
    test_df = test_df.drop(["PassengerId", "Ticket", "Cabin",],
                             axis=1)

    train_data = normalise_data(train_df).values
    eval_data = normalise_data(eval_df).values
    test_data = normalise_data(test_df).values

    no_of_inputs = len(train_data[0])

    data = np.zeros(800, dtype=[('inputs',  float, no_of_inputs),
                                ('outputs', float, 1)])

    for i in range(len(train_data)):
        data[i]['inputs'] = train_data[i]
        data[i]['outputs'] = expected_training_outputs[i]

    print "Training..."
    nn = create_nn(data, (no_of_inputs,100,1))

    print "Evaluating..."
    evaluation = []
    for sample in eval_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            evaluation.append(1)
        else:
            evaluation.append(0)

    em = EvaluationMetrics(evaluation, expected_eval_outputs)
    print "Accuracy: " + str(em.calculate_accuracy())
    print "Precision:" + str(em.calculate_precision())
    print "Recall: " + str(em.calculate_recall())
    print "F1 measure:" + str(em.calculate_f1())

    print "Predicting..."

    output = []
    for sample in test_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            output.append(1)
        else:
            output.append(0)

    print "Writing results..."
    write_results("mlp.csv", ids, output)

    print "Done"

    return True

def sk_svm():
    print "--- SVM ---"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()

    ids = test_df.PassengerId.values

    print "Massaging data..."

    expected_training_outputs = train_df.Survived.values
    train_df = train_df.drop(["PassengerId", "Survived",
                              "Ticket", "Cabin"],
                              axis=1)

    expected_eval_outputs = eval_df.Survived.values
    eval_df = eval_df.drop(["PassengerId", "Survived",\
                            "Ticket", "Cabin"],
                            axis=1)

    # Drop all but class
    test_df = test_df.drop(["PassengerId", "Ticket", "Cabin",],
                             axis=1)

    train_data = normalise_data(train_df).values
    eval_data = normalise_data(eval_df).values
    test_data = normalise_data(test_df).values

    clf = svm.LinearSVC()

    print "Training..."
    clf.fit(train_data, expected_training_outputs)

    print "Evaluating..."
    evaluation = clf.predict(eval_data)

    em = EvaluationMetrics(evaluation, expected_eval_outputs)
    print "Accuracy: " + str(em.calculate_accuracy())
    print "Precision:" + str(em.calculate_precision())
    print "Recall: " + str(em.calculate_recall())
    print "F1 measure:" + str(em.calculate_f1())

    print "Predicting..."
    output = clf.predict(test_data)

    print "Writing results..."
    write_results("svm.csv", ids, output)


    print "Done"

    return True

def ga_mlp():

    print "--- GA & MLP ---"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()
    all_train_df = get_all_training_data()

    ids = test_df.PassengerId.values

    print "Selecting features..."

    ga = MLPFeatureSelector()
    features = ga.calculate()

    print "Massaging data..."

    expected_training_outputs = train_df.Survived.values
    train_data = ga.massage_data_with_outputs(train_df, features)

    expected_eval_outputs = eval_df.Survived.values
    eval_data = ga.massage_data_with_outputs(eval_df, features)

    test_data = ga.massage_data_without_outputs(test_df, features)

    no_of_inputs = features.count(1)

    print "Retraining..."

    all_train_df = get_all_training_data()
    all_train_data = ga.massage_data_with_outputs(all_train_df, features)

    nn = create_nn(all_train_data, (no_of_inputs, 3, 1))

    print "Predicting..."

    output = []
    for sample in test_data:
        out = call_nn(nn, sample[0])
        if out >= 0.5:
            output.append(1)
        else:
            output.append(0)

    print "Writing results..."

    write_results("ga_mlp.csv", ids, output)

    print "Done"

    return True

def ga_rfc():

    print "--- GA & RFC ---"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()
    all_train_df = get_all_training_data()

    ids = test_df.PassengerId.values

    print "Selecting features..."

    ga = RFCFeatureSelector()
    features = ga.calculate()

    print "Massaging data..."
    expected_training_outputs = train_df.Survived.values
    train_data = ga.massage_data_with_outputs(train_df, features)

    expected_eval_outputs = eval_df.Survived.values
    eval_data = ga.massage_data_with_outputs(eval_df, features)

    test_data = ga.massage_data_without_outputs(test_df, features)

    all_expected_outputs = all_train_df.Survived.values
    all_train_data = ga.massage_data_with_outputs(all_train_df, features)

    no_of_inputs = features.count(1)

    print "Retraining..."

    forest = RandomForestClassifier(n_estimators=1000,
                                    n_jobs=-1,
                                    criterion="entropy")

    forest = forest.fit(all_train_data, all_expected_outputs)

    print "Predicting..."

    output = forest.predict(test_data)

    print "Writing results..."
    write_results("ga_rfc.csv", ids, output)

    print "Done"

    return True
