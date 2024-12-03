import pandas as pd

def add_column_tags(row, header):
        str_row = [ f"COL {h} VALUE {row[h]}" for h in header]
        return " ".join(str_row)

def preprocess_row(df):
    df["processed"] = df.apply(add_column_tags, args=(df.head(),), axis=1)


# Encode each document
def encode_document(df, biencoder):
    return biencoder.encode(df['processed'])


def prepare_for_sequence_classification(row1, row2):
    return f'[CLS] {row1} [SEP] {row2} [SEP]'
