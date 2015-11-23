import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from utils import write_results, read_data


def random_forest():
    """
    A random forest classifier
    """
    print "Random Forest Classifier"

    train_df, test_df = read_data()

    ids = test_df.PassengerId.values

    train_df, test_df = get_pclass_data()

    train_data = train_df.values
    test_data = test_df.values

    forest = RandomForestClassifier(n_estimators=100,
                                    criterion="entropy")
    print "Training..."
    forest = forest.fit(train_data[0::,1::], train_data[0::,0])

    print "Predicting..."
    output = forest.predict(test_data).astype(int)

    print "Writing results..."
    write_results("rand_forest.csv", ids, output)

    print "Done!"

def bp_nn():
    """
    A back-propagating neural network
    """
    print "Back-propagating neural network"

    train_df, test_df = read_data()

    ids = test_df.PassengerId.values

    train_df, test_df = get_pclass_data()