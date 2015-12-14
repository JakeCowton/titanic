"""
utils.py

Utility functions for titanic data analysis
"""
import pandas as pd
import fuckit
from sklearn.svm import SVR, SVC


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

def get_all_training_data():
    train_df = pd.read_csv("../data/train.csv", header=0)
    return train_df

def get_testing_data():
    """
    Gets the testing data
    """
    test_df = pd.read_csv("../data/test.csv", header=0)
    return test_df

def normalise_data(data):

    try:
        # Replace missing ages with the median age
        data["Age"] = data["Age"].fillna(data["Age"].median())
    except KeyError:
        # This means it was dropped because of the GA
        pass

    try:
        # Replace NaNs with most frequented point of embarkment
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].max())
    except KeyError:
        # Removed by GA
        pass

    try:
        data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    except KeyError:
        # Removed by GA
        pass

    # Sex
    try:
        data.loc[data["Sex"] == "female", "Sex"] = 0.0
        data.loc[data["Sex"] == "male", "Sex"] = 1.0
    except KeyError:
        # This means it was dropped because of the GA
        pass

    # Embarked
    try:
        data.loc[data["Embarked"] == "C", "Embarked"] = 0.0
        data.loc[data["Embarked"] == "S", "Embarked"] = 1.0
        data.loc[data["Embarked"] == "Q", "Embarked"] = 2.0

    except KeyError:
        # This means it was dropped because of the GA
        pass

    normalised_data = (data - data.min()) / (data.max() - data.min())

    return normalised_data

def fill_nan(input_fields, output_field, to_predict, method="svr"):
    svm = None

    if method == "svr":
        svm = SVR()
    else:
        svm = SVC()

    train_df = get_all_training_data()

    expected_output = train_df[output_field].values
    train_inputs = train_df.drop(["PassengerId",
                                  "Survived", "Name",
                                  "Ticket", "Cabin"],
                                 axis=1)

    train_inputs = train_inputs.get(input_fields).values

    svm.fit(input_fields, outputs)

    return svm.predict(to_predict.get(input_fields).values())
