"""
utils.py

Utility functions for titanic data analysis
"""
import pandas as pd
import fuckit
from sklearn.svm import SVR, SVC


EVALUATION_FOLD_RANGE = {
    0: (0,    89),
    1: (90,  179),
    2: (180, 269),
    3: (270, 359),
    4: (360, 449),
    5: (450, 539),
    6: (540, 629),
    7: (630, 719),
    8: (720, 809),
    9: (810, 889)
}

def write_results(filename, ids, output):
    root_path = "/home/jake/src/ncl/titanic/outputs/"
    with open(root_path + filename, "wb") as f:
        f.write("PassengerId,Survived\n")
        for i in range(len(output)):
            f.write("%d,%d\n" % (ids[i], output[i]))

    return True

def get_training_data(fold=0):
    """
    Min fold = 0
    Max fold = 9
    """
    all_train_df = get_all_training_data()
    lower_limit, upper_limit = EVALUATION_FOLD_RANGE.get(fold)
    # Deletes the eval range
    train_df = all_train_df.drop(all_train_df.index[lower_limit:upper_limit])
    return train_df

def get_evaluation_data(fold=0):
    """
    Min fold = 0
    Max fold = 9
    """
    train_df = get_all_training_data()
    lower_limit, upper_limit = EVALUATION_FOLD_RANGE.get(fold)
    eval_df = train_df.ix[lower_limit:upper_limit]
    return eval_df

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

        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].median())

    except KeyError:
        # This means it was dropped because of the GA
        pass

    # Name (using title)
    try:
        data.loc[data["Name"].str.contains("Mr"), "Title"] = 0.0
        data.loc[data["Name"].str.contains("Mrs"), "Title"] = 1.0
        data.loc[data["Name"].str.contains("Miss"), "Title"] = 2.0
        data.loc[data["Name"].str.contains("Master"), "Title"] = 3.0
        data.loc[data["Name"].str.contains("Don"), "Title"] = 4.0
        data.loc[data["Name"].str.contains("Rev"), "Title"] = 5.0
        data.loc[data["Name"].str.contains("Dr"), "Title"] = 6.0
        data.loc[data["Name"].str.contains("Mme"), "Title"] = 7.0
        data.loc[data["Name"].str.contains("Ms"), "Title"] = 8.0
        data.loc[data["Name"].str.contains("Major"), "Title"] = 9.0
        data.loc[data["Name"].str.contains("Mlle"), "Title"] = 10.0
        data.loc[data["Name"].str.contains("Col"), "Title"] = 11.0
        data.loc[data["Name"].str.contains("Capt"), "Title"] = 12.0
        # Dutch male noble
        data.loc[data["Name"].str.contains("Jonkheer"), "Title"] = 13.0
        data.loc[data["Name"].str.contains("Countess"), "Title"] = 14.0

        data["Name"] = data["Name"].fillna(0.0) # Most common

        # Replaces names with title mappings
        data.Name = data.Title
        data = data.drop(["Title"], axis=1)

    except KeyError:
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

    expected_output = train_df.get([output_field]).values
    train_inputs = train_df.drop(["PassengerId",
                                  "Survived", "Name",
                                  "Ticket", "Cabin"],
                                 axis=1)

    train_inputs = normalise_data(train_inputs.get(input_fields)).values

    svm.fit(input_fields, outputs)

    predict_inputs = normalise_data(to_predict.get(input_fields)).values

    return svm.predict(predict_inputs)
