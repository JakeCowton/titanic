from classifiers import random_forest, slp, mlp, sk_svm, ga_mlp, ga_rfc
from numpy import asarray


def main():
    rfc_f_scores = asarray(random_forest())
    slp_f_scores = asarray(slp())
    mlp_f_scores = asarray(mlp())
    svm_f_scores = asarray(sk_svm())
    ga_mlp_f_scores = asarray(ga_mlp())
    ga_rfc_f_scores = asarray(ga_rfc())

    print "---Random Forest---"
    print "Mean: " + str(rfc_f_scores.mean())
    print "Std : " + str(rfc_f_scores.std())

    print "---SLP---"
    print "Mean: " + str(slp_f_scores.mean())
    print "Std : " + str(slp_f_scores.std())

    print "---MLP---"
    print "Mean: " + str(mlp_f_scores.mean())
    print "Std : " + str(mlp_f_scores.std())

    print "---SVM---"
    print "Mean: " + str(svm_f_scores.mean())
    print "Std : " + str(svm_f_scores.std())

    print "---GA MLP---"
    print "Mean: " + str(ga_mlp_f_scores.mean())
    print "Std : " + str(ga_mlp_f_scores.std())

    print "---GA RFC---"
    print "Mean: " + str(ga_rfc_f_scores.mean())
    print "Std : " + str(ga_rfc_f_scores.std())

if __name__ == "__main__":
    main()
