from sklearn.ensemble import RandomForestClassifier
from utils import write_results, get_training_data,\
                  get_evaluation_data, get_testing_data,\
                  calculate_accuracy
from slp import create_slp

def random_forest():
    """
    A random forest classifier
    """

    print "Random Forest Classifier"

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

    print "Training..."
    forest = RandomForestClassifier(n_estimators=1000,
                                    criterion="entropy")

    inputs = train_data[0::,1::]
    expected_outputs = train_data[0::,0]
    forest = forest.fit(inputs, expected_outputs)

    print "Predicting..."
    evaluation = forest.predict(eval_data)

    print "Accuracy: {:10.4f}".format(calculate_accuracy(evaluation))

    output = forest.predict(test_data)

    print "Writing results..."
    write_results("rand_forest.csv", ids, output)

    print "--- Done ---"

    return True

def slp():
    """
    A single layer perceptron
    """

    print "SLP"

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

    print "Training..."

    massaged_train_data = []
    for sample in train_data:
        massaged_train_data.append([[sample[1]], sample[0]])

    perceptron = create_slp(massaged_train_data)

    print "Predicting..."

    evaluation = []
    for sample in eval_data:
        evaluation.append(perceptron.recall(sample))

    print "Accuracy: {:10.4f}".format(calculate_accuracy(evaluation))

    output = []
    for sample in test_data:
        output.append(perceptron.recall(sample))

    print "Writing results..."
    write_results("slp.csv", ids, output)

    print "--- Done ---"

    return True
