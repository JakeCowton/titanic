"""
utils.py

Utility functions for titanic data analysis
"""
import pandas as pd
import fuckit


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

    return correct / 90.0 # Number of eval samples

def normalise_data(data):
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

            # Ignores all errors as some values won't exist as the GA
            # will remove some
            with fuckit:
                # Convert everything to float
                sample[1].Survived = float(sample[1].Survived)
                sample[1].Pclass = float(sample[1].Pclass)
                sample[1].SibSp = float(sample[1].SibSp)
                sample[1].Parch = float(sample[1].Parch)

                # Normalise
                # Don't need survived as is already between 1 and 0
                sample[1].Pclass /= float(data.Pclass.max())
                sample[1].PclassSex /= float(data.Sex.max())
                sample[1].Age /= float(data.Age.max())
                sample[1].SibSp /= float(data.SibSp.max())
                sample[1].Parch /= float(data.Parch.max())
                sample[1].Fare /= float(data.Fare.max())
                sample[1].Embarked /= float(data.Pclass.max())

            out.append(sample[1].values)

        return out