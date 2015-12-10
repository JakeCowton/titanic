from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score


class EvaluationMetrics(object):

    def __init__(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual

    def calculate_accuracy(self):
        return accuracy_score(self.actual, self.predicted)

    def calculate_precision(self):
        return precision_score(self.actual, self.predicted)

    def calculate_recall(self):
        return recall_score(self.actual, self.predicted)

    def calculate_f1(self):
        return f1_score(self.actual, self.predicted)
