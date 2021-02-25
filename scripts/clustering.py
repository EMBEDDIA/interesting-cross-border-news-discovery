from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import numpy as np
import pickle


def affprop_clustering(doc_embeddings):
    print("Affinity propagation clustering")
    aff_prop = AffinityPropagation().fit(doc_embeddings)
    labels = aff_prop.labels_
    centers = aff_prop.cluster_centers_
    print("Clusters:", len(centers))
    return labels, centers


def kmeans_clustering(doc_embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters).fit(doc_embeddings)
    labels = kmeans.predict(doc_embeddings)
    centers = np.array(kmeans.cluster_centers_)
    return labels, centers


languages = ['latvian', 'estonian']

filepaths = {'estonian': 'estonian_sbert_embeddings.pkl',
             'latvian': 'latvian_sbert_embeddings.pkl'}


doc_embeddings = []
doc_ids = []
for lang in languages:

    data = pickle.load(open(filepaths[lang], 'rb'))
    ids = [k for k in data]
    embeddings = [data[i] for i in ids]
    doc_embeddings.extend(embeddings)
    doc_ids.extend(ids)

cluster_labels, cluster_centers = affprop_clustering(doc_embeddings)