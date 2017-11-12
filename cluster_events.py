import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
import pickle
from sklearn.externals import joblib

# feed it event attribute data
event_data = pd.read_csv('~/event_recommendation/events.csv').fillna(0)

# replace NaN with 0


attributes = list(event_data.columns.values)
numerical_attributes = attributes[7:]
# how to deal with non-numerical values, columns 1-9? ignore them, deal with them in decision tree

X = event_data[numerical_attributes]

batch_size = 32
mbk = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=batch_size,
                      n_init=5, max_iter=600, verbose=0)

print()
mbk.fit(X)

mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

s = pickle.dumps(mbk)
joblib.dump(mbk, 'mbk.pkl')

mbk = joblib.load('mbk.pkl')


evt_map = mbk.predict(X)
np.save('cluster_labels', evt_map)
