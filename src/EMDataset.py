import time
from tqdm import tqdm

class EMDataset:
    def __init__(self, left_table, matching_pairs, matcher):
        self.left_table = left_table
        self.matching_pairs = matching_pairs
        self.matcher = matcher
    
    def profile(self):
        samples_processed = 0
        correct_matched = 0
        start_time = time.time()
        pbar = tqdm(self.matching_pairs.iterrows(), desc=f"Processing example: ", total=len(self.matching_pairs))
        for index, pair in pbar:
            left_id = pair['ltable_id']
            right_id = pair['rtable_id']

            left_row = self.left_table[self.left_table['id'] == left_id]

            predicted_row = self.matcher.match(left_row)

            if right_id == predicted_row['id']:
                correct_matched += 1

            samples_processed += 1
        end_time = time.time()
        print(f"Accuracy: {correct_matched / samples_processed} in time: {end_time-start_time} seconds")
