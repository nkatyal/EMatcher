# Original Ditto Model defined in https://github.com/megagonlabs/ditto/blob/master/ditto_light/ditto.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

from torch.utils import data
from transformers import AutoModel

#from tensorboardX import SummaryWriter
#from apex import amp
from tqdm import tqdm

from .utils import *

tqdm.pandas()

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)


    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc) # .squeeze() # .sigmoid()



class Ditto:
    def __init__(self, dataset, metadata, dk=None, lm='roberta'):
        # load the right table for the mentioned dataset
        dataset_path = metadata[dataset]
        self.right_table = pd.read_csv(dataset_path + '/tableB.csv')
        self.right_table = self.right_table.reset_index(drop=True)
        # Collate the columns into a single document
        preprocess_row(self.right_table)
        self.max_length=512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crossencoder = DittoModel(device=device, lm=lm)
        self.crossencoder.eval()
        self.tokenizer = get_tokenizer(lm)
        self.injector = None
        if dk is not None:
            if dk == 'product':
                self.injector = ProductDKInjector(config, dk)
            else:
                self.injector = GeneralDKInjector(config, dk)
        
        self.da = None
        if self.da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None

    def match(self, left_row):
        scores = []
        left = add_column_tags(left_row, left_row.head())
        for index, right in self.right_table.iterrows():
            # left + right
            x = self.tokenizer.encode(text=left,
                                      text_pair=right.loc['processed'],
                                      max_length=self.max_length,
                                      truncation=True,
                                      return_tensors='pt')

            # augment if da is set
            if self.da is not None:
                combined = self.augmenter.augment_sent(left + ' [SEP] ' + right.loc['processed'], self.da)
                left, right = combined.split(' [SEP] ')
                x_aug = self.tokenizer.encode(text=left,
                                          text_pair=right,
                                          max_length=self.max_length,
                                          truncation=True,
                                          return_tensors='pt')

                with torch.no_grad():
                    score = self.crossencoder(x, x_aug)
            else:
                with torch.no_grad():   
                    score = self.crossencoder(x)
            scores.append(score)

        # Get the tuple with the best score
        best_example = scores.index(max(scores))
        best_row = self.right_table[best_example]

        return best_row