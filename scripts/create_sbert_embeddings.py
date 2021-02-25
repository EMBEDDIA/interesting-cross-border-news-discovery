import os
import json
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

languages = ['estonian', 'latvian']
#lang_codes = ['et', 'lv']


filepaths = {'estonian': 'DELFI_subcorpus/subcorpus.json',
             'latvian': 'lv_articles_entire_collection/'}

# encode articles using SBERT
for lang in languages:
    print("\nLang:", lang.upper())

    if lang is 'latvian':
        # encode latvian articles
        json_files = sorted([f for f in os.listdir(filepaths[lang]) if '.json' in f])
        for jf in json_files:
            encoded_articles = {}
            print("JSON file:", jf)
            articles = json.load(open(filepaths[lang] + jf, 'r'))
            print("Articles:", len(articles))
            for art in articles:
                art_text = art['title'].lower() + ' ' + art['bodyText'].lower()
                encoded_text = model.encode(art_text)
                art_id = art['id']
                encoded_articles[art_id] = encoded_text
            print("Doc embeddings:", len(encoded_articles))
            # save article embeddings
            dump_file = lang + "_" + jf[:-5] + "_sbert_embeddings.pkl"
            with open(dump_file, 'wb') as f:
                pickle.dump(encoded_articles, f)
                f.close()
                print("Saved SBERT doc embeddings as", dump_file, "!")
    else:
        encoded_articles = {}
        # encode estonian articles
        articles = json.load(open(filepaths[lang], 'r'))['list']
        for i, art in enumerate(articles):
            art_id = i
            art_text = art.lower()
            encoded_text = model.encode(art_text)
            encoded_articles[art_id] = encoded_text

        print("Doc embeddings:", len(encoded_articles))
        # save article embeddings
        dump_file = lang + "_sbert_embeddings.pkl"
        with open(dump_file, 'wb') as f:
            pickle.dump(encoded_articles, f)
            f.close()
            print("Saved SBERT doc embeddings as", dump_file, "!")


