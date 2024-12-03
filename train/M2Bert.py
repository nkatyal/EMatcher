import torch
from torch import nn
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR

# Defining the m2bert based re-ranker
class M2BertReranker(nn.Module):
    
    def __init__(self):
        super(M2BertReranker, self).__init__()

        # Using M2BERT as the base model
        self.bert = AutoModel.from_pretrained('togethercomputer/m2-bert-80M-2k-retrieval', trust_remote_code=True)
        self.bert.use_flashfft = True
         # input dimensionality: 768, output dimensionality: 1
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Take the [CLS] token's output (first token in the sequence)
        last_hidden_states = outputs[0]
        cls_output = last_hidden_states[:, 0, :]
        
        # Pass through the linear layer to get relevance scores
        logits = self.classifier(cls_output)
        return logits

class MSMarcoDataset(Dataset):
    def __init__(self, query_dict, passage_dict, training_set, negatives_per_query):
        self.query_dict = query_dict
        self.passage_dict = passage_dict
        self.training_set = training_set
        self.negatives_per_query = negatives_per_query
        self.num_training_pairs = len(self.training_set)
        self.query_batch_size = (1 + self.negatives_per_query)

    def __len__(self):
        return self.num_training_pairs * self.query_batch_size

    def  __getitem__(self, idx):
        query_idx = int(idx // self.query_batch_size)
        doc_idx = idx % self.query_batch_size

        qid = self.training_set[query_idx]['query']

        if doc_idx == 0:
            label = 1
            pid = self.training_set[query_idx]['positive']
        else:
            label = 0
            pid = self.training_set[query_idx][f'negative_{doc_idx}']

        return self.query_dict[qid], self.passage_dict[pid], label


class MSMarcoCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for query_text, passage_text, label in batch:
            # Concatenate query and document with [CLS] and [SEP]
            input_text = f"[CLS] {query_text} [SEP] {passage_text} [SEP]"
            
            # Tokenize and pad the sequences
            encoded = self.tokenizer(
                input_text,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids.append(encoded['input_ids'].squeeze(0))  # Remove batch dimension
            attention_masks.append(encoded['attention_mask'].squeeze(0))
            labels.append(label)

        # Stack the input_ids and attention_masks to create a batch
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': torch.tensor(labels)
        }





def train_reranker(qid_to_query, pid_to_passage, msmarco_training_pairs, num_epochs=1, bert_lr=2e-5, linear_lr=1e-3):
    # Create train-validation split prepare for training
    train_split = 0.7
    num_samples_train = int(len(msmarco_training_pairs) * 0.7)

    msmarco_ds_train = MSMarcoDataset(qid_to_query, 
                                    pid_to_passage, 
                                    msmarco_training_pairs.select(range(num_samples_train)), 
                                    negatives_per_query=3)
    
    msmarco_ds_valid = MSMarcoDataset(qid_to_query, 
                                    pid_to_passage, 
                                    msmarco_training_pairs.select(range(num_samples_train, len(msmarco_training_pairs))), 
                                    negatives_per_query=3)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Create a DataLoader with the collator
    train_dataloader = DataLoader(msmarco_ds_train, batch_size=64, collate_fn=MSMarcoCollator(tokenizer))
    valid_dataloader = DataLoader(msmarco_ds_valid, batch_size=64, collate_fn=MSMarcoCollator(tokenizer))

    model = M2BertReranker()

    # Set up optimizer with different learning rates for BERT and linear layer
    optimizer = Adam([
        {'params': model.bert.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': linear_lr}
    ])

    scheduler = StepLR(optimizer, step_size=10000, gamma=0.1) 
    
    # Loss criterion (binary cross entropy)
    criterion = torch.nn.BCEWithLogitsLoss()
    accuracy = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    curr_loss = 0
    for epoch in range(num_epochs):
        model.train()

        # Progress bar for training
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        curr_batch = 0
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)

            loss = criterion(logits.squeeze(),labels.float())
            curr_loss += loss.item()

            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            scheduler.step()
            # Update progress bar
            pbar.set_postfix(loss=curr_loss / (pbar.n + 1))
        
        # Print average loss for the epoch
        avg_loss = curr_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

        # Validate
        model.eval()
        correct_classified = 0
        total_validation_samples = 0

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validation set"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)
                predictions = torch.sigmoid(logits).squeeze()

                predicted_labels = (predictions > 0.5).int()
                correct_classified += (predicted_labels == labels).sum().item()
                total_validation_samples += labels.size(0)

            curr_accuracy = correct_classified / total_validation_samples

            if curr_accuracy >= accuracy:
                # Saving the best model
                print(f"New best model found with accuracy: {accuracy}")
                accuracy = curr_accuracy
                torch.save(model.state_dict(), "./trained_model/M2BertStateDict.pth")

def main():
    # load the msmarco dataset
    # This dataset contains the queries from the msmarco passage ranking dataset
    query_dataset = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")
    qid_to_query = dict(zip(query_dataset["qid"], query_dataset["text"]))

    # This dataset contains the passages that can be used to answer the queries from above
    passage_dataset = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")
    pid_to_passage = dict(zip(passage_dataset["pid"], passage_dataset["text"]))

    # This document contains the hard negatives mined for each query from the dataset above.
    msmarco_training_pairs = load_dataset('sentence-transformers/msmarco-bm25', 'triplet-50-ids', split='train')

    train_reranker(qid_to_query, pid_to_passage, msmarco_training_pairs)

if __name__ == "__main__":
    main()