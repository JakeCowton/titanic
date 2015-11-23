def write_results(filename, ids, output):
    with open(filename, "wb") as f:
        f.write("PassengerId,Survived\n")
        for i in range(len(output)):
            f.write("%d, %d\n" % (ids[i], output[i]))

    return True
