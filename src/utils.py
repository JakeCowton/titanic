"""
utils.py

Utility functions for titanic data analysis
"""
import pandas as pd


def write_results(filename, ids, output):
    root_path = "/home/jake/src/ncl/titanic/outputs/"
    with open(root_path + filename, "wb") as f:
        f.write("PassengerId,Survived\n")
        for i in range(len(output)):
            f.write("%d,%d\n" % (ids[i], output[i]))

    return True

def get_training_data():
    """
    Gets first 800 rows of train.csv
    """
    train_df = pd.read_csv("../data/train.csv", header=0)
    return train_df[0:800]

def get_evaluation_data():
    """
    Gets last 90 rows of train.csv for accuracy calculation
    """
    train_df = pd.read_csv("../data/train.csv", header=0)
    return train_df[801:]

def get_testing_data():
    """
    Gets the testing data
    """
    test_df = pd.read_csv("../data/test.csv", header=0)
    return test_df

def calculate_accuracy(output):
    """
    E.g. output[x] = 0
    Should be 90 in length
    """
    if len(output) != 90:
        raise IndexError("Output should be 90, not %d" % len(output))

    eval_df = get_evaluation_data()
    eval_df = eval_df.drop(["PassengerId", "Pclass",
                  "Name", "Sex", "Age", "SibSp", "Parch",
                  "Ticket", "Fare", "Cabin", "Embarked"],
                  axis=1)
    eval_data = eval_df.values[0::,0]

    correct = 0

    for i in range(len(output)):
        if output[i] == eval_data[i]:
            correct += 1

    print "%d of 90 were correctly predicted" % correct
    return correct / 90.0 # Number of eval samples
