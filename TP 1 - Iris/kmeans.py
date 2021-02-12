from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import *

df = pd.read_csv("./data/iris.data")
df = df[df.species != "virginica"]

train, test = train_test_split(df, test_size=0.3)
trainX = train.drop(['species'], axis=1).to_numpy()
testX = test.drop(['species'], axis=1).to_numpy()

kmeans = KMeans(n_clusters=2, random_state=0).fit(trainX)

prediction = kmeans.predict(trainX)
print(prediction)
print(train['species'])

trainX_sepal_length = train['sepal_length']
trainX_petal_length = train['petal_length']
trainX_sepal_width = train['sepal_width']
trainX_petal_width = train['sepal_width']

print("shape", trainX_sepal_length.to_numpy().shape)

trainX_sepal_length_a, trainX_sepal_length_b = kmeans_split(
    trainX_sepal_length.to_numpy(), prediction)
trainX_petal_length_a, trainX_petal_length_b = kmeans_split(
    trainX_petal_length.to_numpy(), prediction)
trainX_sepal_width_a, trainX_sepal_width_b = kmeans_split(
    trainX_sepal_width.to_numpy(), prediction)
trainX_petal_width_a, trainX_petal_width_b = kmeans_split(
    trainX_petal_width.to_numpy(), prediction)

'''
fig, ax = plt.subplots(2)
ax[0].set_title("sepal")
ax[1].set_title("petal")

ax[0].scatter(trainX_sepal_length, trainX_sepal_width)
ax[1].scatter(trainX_petal_length, trainX_petal_width)
'''

plt.scatter(trainX_sepal_length_a, trainX_sepal_width_a)
plt.scatter(trainX_sepal_length_b, trainX_sepal_width_b)
plt.show()
