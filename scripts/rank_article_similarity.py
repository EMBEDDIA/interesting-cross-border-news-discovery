import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


languages = ['latvian', 'estonian']
#lang_codes = ['et', 'lv']

filepaths = {'estonian': 'estonian_sbert_embeddings.pkl',
             'latvian': 'latvian_sbert_embeddings.pkl'}


# load SBERT embeddings of Estonian and Latvian articles
doc_embeddings = {lang: [] for lang in languages}
doc_ids = {lang: [] for lang in languages}
for lang in languages:
    print("Lang:", lang.upper())
    data = pickle.load(open(filepaths[lang], 'rb'))
    doc_embeddings[lang] = [data[doc_id] for doc_id in data]
    doc_ids[lang] = [doc_id for doc_id in data]
    print("Doc embeddings:", len(doc_embeddings[lang]))
    print("Doc ids:", len(doc_ids[lang]))
    print(doc_ids[lang][:10])


print("Query articles: Latvian - Candidate articles: Estonian")
# for each Latvian article, rank Estonian articles according to their embedding similarity
art_rankings = {}
for i, doc_id in enumerate(doc_ids['latvian']):
    cos_sims = cosine_similarity([doc_embeddings['latvian'][i]], doc_embeddings['estonian'])[0]
    index_ranking = list(np.argsort(-cos_sims))
    id_ranking = [doc_ids['estonian'][doc_index] for doc_index in index_ranking][:100]
    art_rankings[doc_id] = id_ranking


# save article rankings
dump_file = "delfi_to_ee_sbert_rankings.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(art_rankings, f)
    f.close()
    print("Saved articles rankings as", dump_file, "!")


print("Query articles: Estonian - Candidate articles: Latvian")
# for each Estonian article, rank Latvian articles according to their embedding similarity
art_rankings = {}
for i, doc_id in enumerate(doc_ids['estonian']):
    cos_sims = cosine_similarity([doc_embeddings['estonian'][i]], doc_embeddings['latvian'])[0]
    index_ranking = list(np.argsort(-cos_sims))
    id_ranking = [doc_ids['latvian'][doc_index] for doc_index in index_ranking][:100]
    art_rankings[doc_id] = id_ranking


# save article rankings
dump_file = "ee_to_delfi_sbert_rankings.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(art_rankings, f)
    f.close()
    print("Saved articles rankings as", dump_file, "!")


