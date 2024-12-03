import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from .utils import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../train/')))
from M2Bert import M2BertReranker

tqdm.pandas()

class EMatcher:
    def __init__(self, dataset, metadata):
        # load the right table for the mentioned dataset
        dataset_path = metadata[dataset]
        self.right_table = pd.read_csv(dataset_path + '/tableB.csv')
        self.right_table = self.right_table.reset_index(drop=True)
        #self.right_table = self.right_table.drop('id', axis=1)
        self.max_length=512

        # Generate first stage embeddings
        self.biencoder = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        
        self.crossencoder = M2BertReranker()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.crossencoder.load_state_dict(torch.load("/home/nkatyal/EMatcher/train/trained_model/M2BertStateDict.pth"))
        self.crossencoder.eval()
        # Collate the columns into a single document
        preprocess_row(self.right_table)
        
        self.right_table['embeddings'] = self.right_table.progress_apply(encode_document, axis=1, args=(self.biencoder,)) 
        
        # Construct the first stage index using FAISS
        dimensions = 768
        M = 32
        self.hnsw_index = faiss.IndexHNSWFlat(dimensions, M)
        embeddings = np.array(self.right_table['embeddings'].to_numpy().tolist(), dtype='float32')

        self.hnsw_index.add(embeddings)
    
    def match(self, row):
        query = add_column_tags(row, row.head())
        # First stage: candidate generation
        query_embedding = np.array(self.biencoder.encode(query)).reshape(-1, 1)
        topk = 3
        _, indices = self.hnsw_index.search(query_embedding.T, topk)
        topk_rows = self.right_table.iloc[indices.flatten(), :]

        # Second stage: re-ranking
        # Concatenate the query and all passages and predict the scores for the pairs [query, passage]
        model_inputs = [[query, self.right_table.loc[candidate, 'processed']] for candidate in range(topk)]
        scores = []
        for row1, row2 in enumerate(model_inputs):
            input_text = prepare_for_sequence_classification(row1, row2)
            # Tokenize and pad the sequences
            encoded = self.tokenizer(
                    input_text,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
            with torch.no_grad():
                score = self.crossencoder(encoded['input_ids'], encoded['attention_mask'])
                scores.append(score)

        best_example = scores.index(max(scores))
        best_row = topk_rows.iloc[best_example]

        return best_row
