from perceptron import Perceptron
from sklearn.model_selection import train_test_split
from utils import * 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

t = [[1,2,3],[3,4,5]]

p = Perceptron(4, 50000, 0.01)

df = pd.read_csv("iris.csv")
df = df[df.species != "virginica"]

train, test = train_test_split(df,test_size=0.2)
trainX = train.drop(['species'], axis=1).to_numpy()
trainY = train['species'].astype('category')
trainY = trainY.cat.codes
trainY = trainY / trainY.max()

testX = test.drop(['species'], axis=1).to_numpy()
testY = test['species'].astype('category')
testY = testY.cat.codes
testY = testY / testY.max()

trainX_flatten = trainX.reshape(trainX.shape[0], -1).T
testX_flatten = testX.reshape(testX.shape[0], -1).T

trainX_flatten = (trainX_flatten/trainX_flatten.max())
testX_flatten = (testX_flatten/testX_flatten.max())

W, b, cost = p.fit(trainX_flatten, trainY, True)

x = np.arange(1,51)

np.savetxt("weights.txt", W)
np.savetxt("cost.txt", cost)
saveBias(b, "bias.txt")


W = np.loadtxt("weights.txt")
cost = np.loadtxt("cost.txt")
b = loadBias("bias.txt")

print(p.forward(testX_flatten, W, b))
print(testY)

print(cost)

plt.plot(x, cost)
plt.show()

