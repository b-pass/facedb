#!/usr/bin/env python3

from deepface.DeepFace import verification as DFv
import os
import sys
import time
import json
import numpy as np
import base64
import math
from hdbscan import HDBSCAN

dbdir = '/data/family-photos/facedb/'
MODEL = 'Facenet'
METHOD = 'euclidean'
conf_threshold = 0.4

distfile = f'{dbdir}/distance.{MODEL}.{METHOD}.{conf_threshold}.np'
if os.path.exists(distfile):
    dist = np.fromfile(distfile)
    dist = dist.reshape(int(math.sqrt(len(dist))), -1)
    print('Loaded from', distfile)
else:
    print("Reading from "+dbdir)
    all = []
    for f in sorted(os.listdir(dbdir)):
        if not f.endswith('.json'):
            continue
        #print(f)
        data = json.loads(open(dbdir + f, 'r').read().encode('ascii'))
        for face in data["faces"]:
            if face['confidence'] < conf_threshold:
                continue
            face["file"] = data["path"]
            face["algo"] = data["algo"]
            face['embedding'] = np.frombuffer(base64.b64decode(face['embedding']), dtype='float64').tolist()
            all.append(face)
        #if len(all) > 500: break

    print('... loaded ' + str(len(all)))
    print('Building distance matrix ')
    start = time.time()

    # precompute euclidean_l2 for the whole dataset, this is a 4x speedup
    norm = []

    if METHOD.startswith('euclidean'):
        if METHOD == 'euclidean_l2':
            for a in all:
                norm.append(DFv.l2_normalize(a['embedding']))
        else:
            for a in all:
                norm.append(a['embedding'])

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

    print('built in ' + str(time.time() - start))
    
    dist.tofile(distfile)
    print('Saved to', distfile)
    #def face_distance(a, b):
    #    return DFv.find_distance(all[int(a[0])]['embedding'], all[int(b[0])]['embedding'], 'euclidean_l2')
    #cluster = DBSCAN(metric=face_distance, min_samples=2, eps=DFv.find_threshold(MODEL, 'euclidean_l2'))
    #cluster = HDBSCAN(metric=face_distance, min_cluster_size=2, cluster_selection_epsilon=DFv.find_threshold(MODEL, 'euclidean_l2')*.9)
    #cluster.fit(np.arange(len(all)).reshape(-1, 1))

print('Clustering ')
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering

tries = [
    AgglomerativeClustering(metric='precomputed', linkage='complete', distance_threshold=DFv.find_threshold(MODEL, METHOD), n_clusters=None),
    AgglomerativeClustering(metric='precomputed', linkage='single', distance_threshold=DFv.find_threshold(MODEL, METHOD), n_clusters=None),
    AgglomerativeClustering(metric='precomputed', linkage='average', distance_threshold=DFv.find_threshold(MODEL, METHOD), n_clusters=None),
    HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1, cluster_selection_method='leaf', cluster_selection_epsilon=DFv.find_threshold(MODEL, METHOD)),
    HDBSCAN(metric='precomputed', min_cluster_size=3, cluster_selection_method='leaf', cluster_selection_epsilon=DFv.find_threshold(MODEL, METHOD)),
    DBSCAN(metric='precomputed', min_samples=2, eps=DFv.find_threshold(MODEL, METHOD)),
    DBSCAN(metric='precomputed', min_samples=3, eps=DFv.find_threshold(MODEL, METHOD)),
]

for algo in tries:
    start = time.time()
    print()
    print(algo)
    algo.fit(dist)

    labels = {}
    for x in algo.labels_:
        if x not in labels:
            labels[x] = 1
        else:
            labels[x] += 1
    #print(labels)
    average = 0
    highest = 0
    singles = 0
    if -1 in labels:
        singles = labels[-1]
        del labels[-1]
    for n in labels.values():
        average += n
        if n > highest:
            highest = n
        if n < 2:
            singles += 1
    useful = 0
    average_useful = 0
    for n in labels.values():
        if n > 1 and n < highest:
            useful += 1
            average_useful += n
    if highest < 750:
        average_useful += highest
        useful += 1
        highest = 0
    else:
        singles += highest
        highest = 0
    average_useful = average_useful / useful if useful else 'none'
    average = average / len(labels)

    #print('... finished in ' + str(time.time() - start))
    print("n_labels =", len(labels))
    print("highest =", highest)
    #print("2nd highest =", second_highest)
    print("useful =", useful)
    print("useless =", singles)
    print("average size =", average)
    print("average useful size =", average_useful)
    #print(list(sorted(labels.items())))
    #print(cluster.probabilities_)
