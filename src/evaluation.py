
def calculate_accuracy(classified_outs, actual_outs):
    """
    E.g. output[x] = 0
    Should be 90 in length
    """
    if len(classified_outs) != 90:
        raise IndexError("Output should be 90, not %d" % len(classified_outs))

    correct = 0

    for i in range(len(classified_outs)):
        if classified_outs[i] == actual_outs[i]:
            correct += 1

    accuracy = correct / float(len(classified_outs)) # Number of eval samples

    print "Accuracy: {:10.4f}".format(accuracy)

    return accuracy
