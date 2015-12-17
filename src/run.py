from classifiers import random_forest, slp, mlp, sk_svm, ga_mlp, ga_rfc,ga_svm
from numpy import asarray, matrix


def main():
    rfc_f_scores = asarray(random_forest())
    slp_f_scores = asarray(slp())
    mlp_f_scores = asarray(mlp())
    svm_f_scores = asarray(sk_svm())
    ga_mlp_f_scores = asarray(ga_mlp())
    ga_rfc_f_scores = asarray(ga_rfc())
    ga_svm_f_scores = asarray(ga_svm())

    print "---Random Forest---"
    print "F-Scores:" + str(rfc_f_scores)
    print "Mean: " + str(rfc_f_scores.mean())
    print "Std : " + str(rfc_f_scores.std())

    print "---SLP---"
    print "F-Scores:" + str(slp_f_scores)
    print "Mean: " + str(slp_f_scores.mean())
    print "Std : " + str(slp_f_scores.std())

    print "---MLP---"
    print "F-Scores:" + str(mlp_f_scores)
    print "Mean: " + str(mlp_f_scores.mean())
    print "Std : " + str(mlp_f_scores.std())

    print "---SVM---"
    print "F-Scores:" + str(svm_f_scores)
    print "Mean: " + str(svm_f_scores.mean())
    print "Std : " + str(svm_f_scores.std())

    print "---GA RFC---"
    print "F-Scores:" + str(ga_rfc_f_scores)
    print "Mean: " + str(ga_rfc_f_scores.mean())
    print "Std : " + str(ga_rfc_f_scores.std())

    print "---GA MLP---"
    print "F-Scores:" + str(ga_mlp_f_scores)
    print "Mean: " + str(ga_mlp_f_scores.mean())
    print "Std : " + str(ga_mlp_f_scores.std())

    print "---GA SVM---"
    print "F-Scores:" + str(ga_svm_f_scores)
    print "Mean: " + str(ga_svm_f_scores.mean())
    print "Std : " + str(ga_svm_f_scores.std())

    return matrix([rfc_f_scores,
                   slp_f_scores,
                   mlp_f_scores,
                   svm_f_scores,
                   ga_rfc_f_scores,
                   ga_mlp_f_scores,
                   ga_svm_f_scores])

if __name__ == "__main__":
    mat = main()
