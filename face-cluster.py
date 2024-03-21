#!/usr/bin/env python3

from deepface.DeepFace import verification as DFv
import os
import sys
import time
import json
import numpy as np
import base64
from hdbscan import HDBSCAN

dbdir = '/data/family-photos/facedb/'
MODEL = 'DeepFace'
METHOD = 'euclidean_l2'

print("Loading from "+dbdir)
all = []
for f in sorted(os.listdir(dbdir)):
    if not f.endswith('.json'):
        continue
    #print(f)
    data = json.loads(open(dbdir + f, 'r').read().encode('ascii'))
    if "embedding" in data:
        data['embedding'] = np.frombuffer(base64.b64decode(data['embedding']), dtype='float64').tolist()
    else:
        for face in data["faces"]:
            face["file"] = data["path"]
            face["algo"] = data["algo"]
            face['embedding'] = np.frombuffer(base64.b64decode(face['embedding']), dtype='float64').tolist()
            all.append(face)
    #if len(all) > 500: break

print('... loaded ' + str(len(all)))
print('Clustering ')
start = time.time()

from sklearn.cluster import DBSCAN, HDBSCAN
#def face_distance(a, b):
#    return DFv.find_distance(all[int(a[0])]['embedding'], all[int(b[0])]['embedding'], 'euclidean_l2')
#cluster = DBSCAN(metric=face_distance, min_samples=2, eps=DFv.find_threshold(MODEL, 'euclidean_l2'))
#cluster = HDBSCAN(metric=face_distance, min_cluster_size=2, cluster_selection_epsilon=DFv.find_threshold(MODEL, 'euclidean_l2')*.9)
#cluster.fit(np.arange(len(all)).reshape(-1, 1))

# precompute euclidean_l2 for the whole dataset, this is a 4x speedup
norm = []

if METHOD == 'euclidean_l2':
    for a in all:
        norm.append(DFv.l2_normalize(a['embedding']))

    dist = np.ndarray((len(all),len(all)), dtype='float64')
    for a in range(len(all)):
        for b in range(a+1, len(all)):
            d = DFv.find_euclidean_distance(norm[a], norm[b])
            dist[a,b] = d
            dist[b,a] = d
elif METHOD == 'cosine':
    for a in all:
        norm.append(np.sqrt(np.sum(np.multiply(a['embedding'], a['embedding']))))

    dist = np.ndarray((len(all),len(all)), dtype='double')
    for a in range(len(all)):
        for b in range(a+1, len(all)):
            x = np.matmul(np.transpose(all[a]['embedding']), all[b]['embedding'])
            d = 1 - (x / (norm[a] * norm[b]))
            dist[a,b] = d
            dist[b,a] = d

cluster = HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1, cluster_selection_method='leaf', 
                        cluster_selection_epsilon=DFv.find_threshold(MODEL, METHOD))
#cluster = DBSCAN(metric='precomputed', min_samples=2, eps=DFv.find_threshold(MODEL, METHOD))
print('... preprocessed in ' + str(time.time() - start))
cluster.fit(dist)

labels = {}
n = 0
for x in cluster.labels_:
    n += 1
    if x not in labels:
        labels[x] = 1
    else:
        labels[x] += 1

print('... finished in ' + str(time.time() - start))
print(list(sorted(labels.items())))
#print(cluster.probabilities_)
