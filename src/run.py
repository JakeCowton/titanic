from classifiers import random_forest, slp, mlp, sk_svm, ga_mlp, ga_rfc


def main():
    random_forest()
    slp()
    mlp()
    sk_svm()
    ga_mlp()
    ga_rfc()

if __name__ == "__main__":
    main()
