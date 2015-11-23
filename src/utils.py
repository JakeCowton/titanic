"""
utils.py

Utility functions for titanic data analysis
"""

def write_results(filename, ids, output):
    root_path = "/home/jake/src/ncl/titanic/outputs/"
    with open(root_path + filename, "wb") as f:
        f.write("PassengerId,Survived\n")
        for i in range(len(output)):
            f.write("%d,%d\n" % (ids[i], output[i]))

    return True

def read_data():
    train_df = pd.read_csv("../data/train.csv", header=0)
    test_df = pd.read_csv("../data/test.csv", header=0)

    return train_df, test_df

def get_pclass_data(train_df, test_df):
    train_df = train_df.drop(["PassengerId",
                              "Name", "Sex", "Age", "SibSp", "Parch",
                              "Ticket", "Fare", "Cabin", "Embarked"],
                              axis=1)

    test_df = test_df.drop(["PassengerId",
                            "Name", "Sex", "Age", "SibSp", "Parch",
                            "Ticket", "Fare", "Cabin", "Embarked"],
                              axis=1)

    return train_df, test_ids, test_df