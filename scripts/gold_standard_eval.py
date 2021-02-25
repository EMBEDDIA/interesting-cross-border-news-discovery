import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

lang_codes = ['et', 'lv']

paired_articles = json.load(open("articles_21.json", "r"))
paired_embeddings = {lang: [] for lang in lang_codes}

# encode articles using SBERT
for lang in lang_codes:
    print("\nLang:", lang.upper())
    embeddings = []
    # encode latvian articles
    for i, art in enumerate(paired_articles):
        art_text = art[lang]['title'].lower() + " " + art[lang]['body'].lower()
        encoded_text = model.encode(art_text)
        embeddings.append(encoded_text)
    paired_embeddings[lang] = embeddings
    print("Doc emb:", len(embeddings))


# rank articles according to similarity
cos_sims = cosine_similarity(paired_embeddings['et'], paired_embeddings['lv'])
ranks = [np.argsort(-cos_sims[i]) for i in range(cos_sims.shape[0])]

# compute precision@1:
prec_1 = np.mean([1 if ranks[i][0] == i else 0 for i in range(len(ranks))])
print("Prec@1:", prec_1)

# compute MRR
MRR = np.mean([1/(np.where(ranks[i]==i)[0][0]+1) for i in range(len(ranks))])
print("MRR", MRR)