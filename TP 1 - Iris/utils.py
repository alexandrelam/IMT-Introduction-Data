def loadBias(filename):
    with open(filename) as f:
        res = f.read()

    return float(res)

def saveBias(bias, filename):
    with open(filename, "w") as f:
        f.write(str(bias))


