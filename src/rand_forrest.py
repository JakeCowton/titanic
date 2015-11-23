import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from utils import write_results

train_df = pd.read_csv("../data/train.csv", header=0)
test_df = pd.read_csv("../data/test.csv", header=0)

train_df = train_df.drop(["PassengerId",
                          "Name", "Sex", "Age", "SibSp", "Parch",
                          "Ticket", "Fare", "Cabin", "Embarked"],
                          axis=1)

test_ids = test_df.PassengerId.values
test_df = test_df.drop(["PassengerId",
                        "Name", "Sex", "Age", "SibSp", "Parch",
                        "Ticket", "Fare", "Cabin", "Embarked"],
                          axis=1)

train_data = train_df.values
test_data = test_df.values

forest = RandomForestClassifier(n_estimators=100,
                                criterion="entropy")
print "Training..."
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

print "Predicting..."
output = forest.predict(test_data).astype(int)

print "Writing results..."
write_results("rand_forrest.csv", test_ids, output)

print "Done!"
