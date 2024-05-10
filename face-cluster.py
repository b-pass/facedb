#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow
tensorflow.keras.utils.disable_interactive_logging()

import cv2
from deepface.DeepFace import verification as DFv
import sys
import time
import json
import numpy as np
import base64
import math
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt

METHOD = 'euclidean_l2'
if len(sys.argv) > 1:
    dbdir = sys.argv[1]
    if len(sys.argv) > 2:
        METHOD = sys.argv[2]
MODEL = None
conf_threshold = 0.5

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
        #if face['facial_area']['w'] < 60 or face['facial_area']['h'] < 60:
        #    continue
        face["path"] = data["path"]
        face['basename'] = data['basename']
        face["algo"] = data["algo"]
        if not MODEL:
            MODEL = face['algo']
        face['embedding'] = np.frombuffer(base64.b64decode(face['embedding']), dtype='float64').tolist()
        all.append(face)
    #if len(all) > 500: break
print('... loaded ' + str(len(all)))

distfile = f'{dbdir}/distance.{MODEL}.{METHOD}.{conf_threshold}.np'
if os.path.exists(distfile):
    dist = np.fromfile(distfile)
    dist = dist.reshape(int(math.sqrt(len(dist))), -1)
    print('Loaded from', distfile)
else:
    print('Building distance matrix ')
    start = time.time()

    norm = []

    if METHOD.startswith('euclidean'):
        # precompute euclidean_l2 for the whole dataset, this is a 4x speedup
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

        dist = np.ndarray((len(all),len(all)), dtype='float64')
        for a in range(len(all)):
            for b in range(len(all)):
                if b <= a: continue
                x = np.matmul(np.transpose(all[a]['embedding']), all[b]['embedding'])
                d = 1 - (x / (norm[a] * norm[b]))
                dist[a,b] = d
                dist[b,a] = d
    else:
        print('FAIL, method unknown')
        sys.exit(1)

    print('built in ' + str(time.time() - start))
    
    dist.tofile(distfile)
    print('Saved to', distfile)
    #def face_distance(a, b):
    #    return DFv.find_distance(all[int(a[0])]['embedding'], all[int(b[0])]['embedding'], 'euclidean_l2')
    #cluster = DBSCAN(metric=face_distance, min_samples=2, eps=DFv.find_threshold(MODEL, 'euclidean_l2'))
    #cluster = HDBSCAN(metric=face_distance, min_cluster_size=2, cluster_selection_epsilon=DFv.find_threshold(MODEL, 'euclidean_l2')*.9)
    #cluster.fit(np.arange(len(all)).reshape(-1, 1))

print('Clustering ')

import networkx as nx
g = nx.Graph()
thresh = DFv.find_threshold(MODEL, METHOD)
y = 0
n = 0
for a in range(len(all)):
    g.add_node(n, idx=n)
for a in range(len(all)):
    for b in range(a+1,len(all)):
        if dist[a,b] < thresh:
            g.add_edge(a, b)#, weight=1.0-dist[a,b])
            y += 1
        else:
            n += 1
print(f"\nGot {y} edges and skipped {n} ({n/(y+n)*100.0}%)")

print(f'Connected components =',len([x for x in nx.connected_components(g)]))

count = 0
while True:
    done = True
    for c in nx.connected_components(g):
        if len(c) < 200: continue
        print(f"Component len={len(c)}")

        worst = (-1,1)
        best = (-1,0)
        for n in c:
            d = all[n]['confidence']
            if d < worst[1]:
                worst = (n,d)
            if d > best[1]:
                best = (n,d)
        print(f'Worst confidence in {len(c)} is on {worst[0]}, it is {worst[1]*100}')
        print(all[worst[0]])
        print(f'Best confidence in {len(c)} is on {best[0]}, it is {best[1]*100}')
        print(f'Shortest path is {nx.shortest_path(g,best[0], worst[0])}')
        #g.remove_node(worst[0])
        done = False
        count += 1
        break
    #if count > 10:
    sys.exit(1)
    print('\n\n\nmore??\n\n')

#print(f"Bad components = {len(bad)}, Good components = {len(good)}")
#sys.exit(1)

print(f'thresh {thresh}')
while True:
    count = 0
    for (n,d) in sorted(g.degree, key=lambda x: x[1], reverse=True):
        count += 1
        print("degree?",n,'=',d)
        #print(all[n]['filename'])
        worst = (-1,0)
        for x in range(len(all)):
            if dist[n,x] > worst[1] and dist[n,x] < thresh and n!=x and g.has_edge(n,x):
                worst = (x,dist[n,x])
        print(f'worst edge ({n},{worst[0]}) = {worst[1]}')
        g.remove_edge(n,worst[0])
        if d < 500:
            break
    if count < 5:
        sys.exit(99)
    print('\n\nMORE??\n')
    d
sys.exit(1)

pos = nx.spring_layout(g)

elab = {}
for i,j,w in g.edges.data('weight'):
    elab[(i,j)] = w

degree_sequence = sorted((d for n, d in g.degree()), reverse=True)
dmax = max(degree_sequence)

fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# Create a gridspec for adding subplots of different sizes
axgrid = fig.add_gridspec(5, 4)

ax0 = fig.add_subplot(axgrid[0:3, :])
nx.draw(g, pos, ax=ax0)
#nx.draw_networkx_edge_labels(g, pos, edge_labels=elab) 
ax0.set_title("G")
ax0.set_axis_off()

ax1 = fig.add_subplot(axgrid[3:, :2])
ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Degree Rank Plot")
ax1.set_ylabel("Degree")
ax1.set_xlabel("Rank")

ax2 = fig.add_subplot(axgrid[3:, 2:])
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")

fig.tight_layout()

plt.show()
sys.exit(1)

#tries = [
#    AgglomerativeClustering(metric='precomputed', linkage='complete', distance_threshold=DFv.find_threshold(MODEL, METHOD), n_clusters=None),
#    AgglomerativeClustering(metric='precomputed', linkage='single', distance_threshold=DFv.find_threshold(MODEL, METHOD), n_clusters=None),
#    AgglomerativeClustering(metric='precomputed', linkage='average', distance_threshold=DFv.find_threshold(MODEL, METHOD), n_clusters=None),
#    HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1, cluster_selection_method='leaf', cluster_selection_epsilon=DFv.find_threshold(MODEL, METHOD)),
#    HDBSCAN(metric='precomputed', min_cluster_size=3, cluster_selection_method='leaf', cluster_selection_epsilon=DFv.find_threshold(MODEL, METHOD)),
#    DBSCAN(metric='precomputed', min_samples=2, eps=DFv.find_threshold(MODEL, METHOD)),
#    DBSCAN(metric='precomputed', min_samples=3, eps=DFv.find_threshold(MODEL, METHOD)),
#]
start = time.time()
#algo = AgglomerativeClustering(metric='precomputed', linkage='complete', distance_threshold=DFv.find_threshold(MODEL, METHOD), n_clusters=None)
#algo = HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1, cluster_selection_method='leaf', cluster_selection_epsilon=DFv.find_threshold(MODEL, METHOD))
algo = AgglomerativeClustering(metric='precomputed', linkage='average', distance_threshold=DFv.find_threshold(MODEL, METHOD), n_clusters=None)
print()
print(algo)
algo.fit(dist)

clusters = {}
i = 0
for (f,label) in zip(all,algo.labels_):
    f['idx'] = i
    i += 1
    if label not in clusters:
        clusters[label] = [f]
    else:
        clusters[label].append(f)

dedupe = []
for (label,cluster) in clusters.items():
    if len(cluster) < 500:
        done = {}
        for x in cluster:
            if x['basename'] not in done or done[x['basename']]['confidence'] < x['confidence']:
                done[x['basename']] = x
        for f in done.values():
            dedupe.append(f)
    else:
        dedupe += cluster
print(f"Cluster-dedupe from {len(all)} to {len(dedupe)}")
dist2 = np.ndarray((len(dedupe),len(dedupe)), dtype='float64')
for a in range(len(dedupe)):
    for b in range(a+1, len(dedupe)):
        dist2[b,a] = dist2[a,b] = dist[dedupe[a]['idx'], dedupe[b]['idx']]

dist = dist2
all = dedupe

algo = HDBSCAN(metric='precomputed', min_cluster_size=3, min_samples=1, cluster_selection_method='leaf', cluster_selection_epsilon=DFv.find_threshold(MODEL, METHOD)/2)
#algo = DBSCAN(metric='precomputed', min_samples=2, eps=DFv.find_threshold(MODEL, METHOD))
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

print('... finished in ' + str(time.time() - start))
print("n_labels =", len(labels))
print("highest =", highest)
#print("2nd highest =", second_highest)
print("useful =", useful)
print("useless =", singles)
print("average size =", average)
print("average useful size =", average_useful)
#print(list(sorted(labels.items())))
#print(cluster.probabilities_)

labeled = {}
for (f,x) in zip(all,algo.labels_):
    if x not in labeled:
        labeled[x] = []
    labeled[x].append(f)
labeled.pop(-1, [])

#try: os.mkdir(f'{dbdir}/clusters/')
#except: pass

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        
        if i < len(images):
            img = images[i]#img = cv2.resize(images[i], (h, w))
            plt.imshow(img, cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
        
        plt.xticks(())
        plt.yticks(())

embeddings = []
imgs = []
titles = []

print('Loading image previews')
for face in all:
    im = cv2.imread(face['path'])
    a = face['facial_area']
    imgs.append(cv2.resize(im[ a['y'] : a['y'] + a['h'], a['x'] : a['x']+a['w'] ], (160,160)))
    titles.append(face['basename'])
    embeddings.append(face['embedding'])
print('Done')
'''
from sklearn.decomposition import PCA
def scatter_thumbnails(data, images, zoom=0.12, colors=None):
    assert len(data) == len(images)

    # reduce embedding dimentions to 2
    x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data

    # create a scatter plot.
    f = plt.figure(figsize=(22, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], s=4)
    _ = ax.axis('off')
    _ = ax.axis('tight')

    # add thumbnails :)
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    for i in range(len(images)):
        image = images[i]
        im = OffsetImage(image, zoom=zoom)
        bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
        ab = AnnotationBbox(im, x[i], xycoords='data',
                            frameon=(bboxprops is not None),
                            pad=0.02,
                            bboxprops=bboxprops)
        ax.add_artist(ab)
    return ax

def plot_clusters(x, algorithm, *args, **kwds):
    import seaborn as sns
    labels = algorithm(*args, **kwds).fit_predict(x)
    palette = sns.color_palette('deep', np.max(labels) + 1)
    colors = [palette[x] if x >= 0 else (0,0,0) for x in labels]
    ax = scatter_thumbnails(x, imgs, 0.06, colors)
    plt.title(f'Clusters found by {algorithm.__name__}')
    return labels

from sklearn.manifold import TSNE
# PCA first to speed it up
x = PCA(n_components=50).fit_transform(embeddings)
x = TSNE(perplexity=50, n_components=3).fit_transform(x)

# clusters = plot_clusters(x, hdbscan.HDBSCAN, alpha=1.0, min_cluster_size=2, min_samples=1)
clusters = plot_clusters(x, DBSCAN, n_jobs=-1, eps=1.0, min_samples=1)
plt.show()
'''

for k,faces in labeled.items():
    if len(faces) < 2:
        continue

    if len(faces) > 500:
        print('Label',k,'has',len(faces),'faces, that is ridiculous')
        continue

    images = []
    titles = []
    for face in faces:
        im = cv2.imread(face['path'])
        a = face['facial_area']
        images.append(im[ a['y'] : a['y'] + a['h'], a['x'] : a['x']+a['w'] ])
        titles.append(face['basename'])
    
    c = int(math.sqrt(len(images)))
    r = c
    if r*c < len(images):
        r += 1
    
    plot_gallery(images, titles, 121, 121, r, c)
    plt.show()

sys.exit(1)

def scatter_thumbnails(zoom=0.12, colors=None):
    #assert len(data) == len(images)

    # reduce embedding dimentions to 2
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    # PCA first to speed it up
    x = PCA(n_components=50).fit_transform(embeddings)
    x = TSNE(perplexity=50, n_components=3).fit_transform(x)
    
    #from sklearn.manifold import MDS
    #x = MDS(n_components=2).fit_transform(dist)

    # create a scatter plot.
    f = plt.figure(figsize=(22, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], s=4)
    _ = ax.axis('off')
    _ = ax.axis('tight')

    # add thumbnails :)
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    for i in range(len(imgs)):
        image = imgs[i]
        im = OffsetImage(image, zoom=zoom)
        bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
        ab = AnnotationBbox(im, x[i], xycoords='data',
                            frameon=(bboxprops is not None),
                            pad=0.02,
                            bboxprops=bboxprops)
        ax.add_artist(ab)
    return ax

import seaborn as sns
labels = algo.labels_
palette = sns.color_palette('deep', np.max(labels) + 1)
colors = [palette[x] if x >= 0 else (0,0,0) for x in labels]
ax = scatter_thumbnails(0.06, colors)
plt.title(f'Clusters')
plt.show()#plt.savefig('/tmp/clusters.png')
sys.exit(0)


# clusters = plot_clusters(x, hdbscan.HDBSCAN, alpha=1.0, min_cluster_size=2, min_samples=1)
#plt.show()

for k,faces in labeled.items():
    if len(faces) < 2:
        continue

    if len(faces) > 500:
        print('Label',k,'has',len(faces),'faces, that is ridiculous')
        continue

    images = []
    titles = []
    for face in faces:
        im = cv2.imread(face['path'])
        a = face['facial_area']
        images.append(im[ a['y'] : a['y'] + a['h'], a['x'] : a['x']+a['w'] ])
        titles.append(face['basename'])
    
    c = int(math.sqrt(len(images)))
    r = c
    if r*c < len(images):
        r += 1
    
    plot_gallery(images, titles, 121, 121, r, c)
    plt.show()


