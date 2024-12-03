import sys
import json
import argparse
import pandas as pd

from EMModels.EMatcher import EMatcher
from EMModels.utils import preprocess_row
from EMDataset import EMDataset

def main_cli():
    
    # Loading the config file that contains the datasets and saved model locations relative to this source file
    with open("../datasets_config.json", 'r') as config_file:
        metadata = json.load(config_file)
    
    # Parsing command line arguments
    parser = argparse.ArgumentParser(
        description="The main routine that performs entity matching on the dataset using a model parsed through the command line"
    )

    parser.add_argument("--dataset", type=str, required=True, 
        help=f"The tag of the data set to be used for entity matching, can be one of: {metadata.keys}")
    
    parser.add_argument("--model", type=str, required=False)

    args = parser.parse_args()

    # Verifying the correctness of arguments
    if args.dataset not in metadata.keys():
        raise ValueError(f"dataset can be one of: {metadata.keys()}")

    em = EMatcher(args.dataset, metadata)

    left_table = pd.read_csv(metadata[args.dataset] + "tableA.csv")

    # All pairs are used only for inference and not training
    training_pairs = pd.read_csv(metadata[args.dataset] + 'train.csv')
    validation_pairs = pd.read_csv(metadata[args.dataset] + 'valid.csv')
    testing_pairs = pd.read_csv(metadata[args.dataset] + 'test.csv')
    stacked_pairs = pd.concat([training_pairs, validation_pairs, testing_pairs], ignore_index=True)
    matching_pairs = stacked_pairs[stacked_pairs['label'] == 1]
    
    left_table = left_table.reset_index(drop=True)

    emd = EMDataset(left_table, matching_pairs, em)

    emd.profile()

if __name__ == "__main__":
    main_cli()