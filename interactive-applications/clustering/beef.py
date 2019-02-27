import khiva as kv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("Beef_TRAIN.txt", header=None, sep=' ')

data = data.values
classes = data[:,0]
data = np.delete(data, 0, 1)

print(data.shape)

plt.plot(data.transpose())
plt.grid(True)
plt.title("Todas las señales")
plt.show()

plt.plot(data[0, :])
plt.plot(data[1, :])
plt.plot(data[2, :])
plt.grid(True)
plt.title("Tres primeras señales")
plt.show()
k=5
(centroids, labels) = kv.k_shape(kv.Array(data), k)

cen = centroids.to_numpy()
lab = labels.to_numpy()

for i in range(k):
    cond = lab == i
    if np.any(cond):
        plt.plot(data[lab == i, :].transpose())
        plt.plot(cen[i, :], label="centroide", c='k')
        plt.grid(True)
        plt.show()

comparacion_kShape = np.vstack((classes, lab+1))

(centroids, labels) = kv.k_means(kv.Array(data), 5)

cen_km = centroids.to_numpy()
lab_km = labels.to_numpy()

km = KMeans(n_clusters=5).fit(data)
comparacion_KMeans = np.vstack((classes, km.labels_+1))



print("hola")
