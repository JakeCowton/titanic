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
    train_df = pd.read_csv("../data/train.csv", header=0)
    return train_df.values[0:89]

def get_evaluation_data():
    train_df = pd.read_csv("../data/train.csv", header=0)
    return train_df.values[90:]

def get_testing_data():
    test_df = pd.read_csv("../data/test.csv", header=0).values
    return test_df.values
