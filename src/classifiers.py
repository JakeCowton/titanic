from sklearn.ensemble import RandomForestClassifier
from utils import write_results, get_training_data,\
                  get_evaluation_data, get_testing_data,\
                  calculate_accuracy


def random_forest():
    """
    A random forest classifier
    """

    print "Random Forest Classifier"

    train_df = get_training_data()
    eval_df = get_evaluation_data()
    test_df = get_testing_data()

    ids = test_df.PassengerId.values

    train_df = train_df.drop(["PassengerId",
                              "Name", "Sex", "Age", "SibSp", "Parch",
                              "Ticket", "Fare", "Cabin", "Embarked"],
                              axis=1)

    eval_df = eval_df.drop(["PassengerId", "Survived",
                            "Name", "Sex", "Age", "SibSp", "Parch",
                            "Ticket", "Fare", "Cabin", "Embarked"],
                            axis=1)

    test_df = test_df.drop(["PassengerId",
                             "Name", "Sex", "Age", "SibSp", "Parch",
                             "Ticket", "Fare", "Cabin", "Embarked"],
                             axis=1)

    train_data = train_df.values
    eval_data = eval_df.values
    test_data = test_df.values

    forest = RandomForestClassifier(n_estimators=100,
                                    criterion="entropy")
    print "Training..."
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

def bp_nn():
    """
    A back-propagating neural network
    """

    print "Back-propagating neural network"

    training_data = get_training_data()
    eval_data = get_evaluation_data()
    testing_data = get_testing_data()

    ids = test_df.PassengerId.values

