def loadBias(filename):
    with open(filename) as f:
        res = f.read()

    return float(res)


def saveBias(bias, filename):
    with open(filename, "w") as f:
        f.write(str(bias))


def accuracy(testY, predictions):
    count = 0.0
    testY = testY.to_numpy()
    for i in range(len(predictions)):
        if predictions[i] > 0.5 and testY[i]:
            count += 1
        if predictions[i] < 0.5 and testY[i] == 0:
            count += 1
    return count / len(testY)


def kmeans_split(arr, prediction):
    a = []
    b = []
    for i in range(len(arr)):
        if prediction[i]:
            a.append(arr[i])
        else:
            b.append(arr[i])

    return a, b
