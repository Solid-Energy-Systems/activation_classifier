import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
from collections import defaultdict
from arrangement_puzzle.seating_puzzle import load_puzzle_from_disk
from arrangement_puzzle.llm_parser import analyze_llm_output
import h5py
import json

class ActivationDataset(Dataset):
    def __init__(self, 
                 preprocessed_data_dir, 
                 layers_to_load,
                 sequence_length=200,
                 verbose=True):
        self.preprocessed_data_dir = preprocessed_data_dir
        self.layers_to_load = layers_to_load
        self.sequence_length = sequence_length
        self.verbose = verbose

        self.activations = []
        self.labels = []

        # Traverse the top-level directory, looking for subdirectories
        for puzzle_dir in os.listdir(self.preprocessed_data_dir):
            puzzle_path = os.path.join(self.preprocessed_data_dir, puzzle_dir)
            if not os.path.isdir(puzzle_path):
                continue
            
            h5_filepath = os.path.join(puzzle_path, 'hidden_states.h5')
            json_filepath = os.path.join(puzzle_path, 'parsed_data.json')

            # If either file is missing, skip this puzzle
            if not os.path.exists(h5_filepath):
                if self.verbose:
                    print(f"Warning: Missing hidden_states.h5 in {puzzle_dir}, skipping.")
                continue
            if not os.path.exists(json_filepath):
                if self.verbose:
                    print(f"Warning: Missing parsed_data.json in {puzzle_dir}, skipping.")
                continue

            # -- Load JSON for token categories --
            with open(json_filepath, 'r') as jf:
                data = json.load(jf)
                input_string = data['input_string']
                input_ids = data['input_ids']
                token_dict = {int(k): v for k, v in data['token_dict'].items()}

            # -- Load shape from H5 and read the needed layers --
            with h5py.File(h5_filepath, 'r') as hf:
                ds = hf['tensor_data']  # shape: (batch_size, num_layers, num_tokens, hidden_size)
                
                # We expect batch_size=1, so shape looks like:
                # (1, n_layers, n_tokens, hidden_size)
                b_size, n_layers, n_tokens, hidden_size = ds.shape
                if b_size != 1:
                    if self.verbose:
                        print(f"Warning: batch_size != 1 for {puzzle_dir} (got {b_size}), skipping.")
                    continue

                # Check that requested layers are valid
                if max(self.layers_to_load) >= n_layers:
                    if self.verbose:
                        print(f"Warning: Requested layer index out of range for {puzzle_dir}, skipping.")
                    continue
                
                # Only load the selected layers from disk (one slice at a time).
                layer_data_list = []
                for lidx in self.layers_to_load:
                    # ds[0, lidx, :, :] => shape [num_tokens, hidden_size]
                    layer_data_np = ds[0, lidx, :, :]
                    layer_data_list.append(torch.tensor(layer_data_np))

                # Stack them along dim=0 -> shape: [len(layers_to_load), num_tokens, hidden_size]
                activation_data = torch.stack(layer_data_list, dim=0)

            # -- Calculate prompt length --
            num_prompt_tokens = len(input_ids) - n_tokens
            if num_prompt_tokens < 0:
                if self.verbose:
                    print(f"Warning: Negative prompt tokens for {puzzle_dir}, skipping.")
                continue

            # -- Build category array for the relevant tokens --
            token_categories = []
            for idx in range(num_prompt_tokens, len(input_ids)):
                cat = token_dict.get(idx, 'unknown')
                token_categories.append(cat)

            # -- Find sequences of consecutive tokens with the same category --
            idx_token = len(token_categories) - 1
            while idx_token >= 0:
                current_category = token_categories[idx_token]
                if current_category in ['solution_correct', 'solution_incorrect', 'clue_hallucination']:
                    end_idx = idx_token
                    # Assign label: 'solution_correct' -> 1, otherwise 0
                    category_label = 1 if current_category == 'solution_correct' else 0
                    # Move backwards until category changes
                    while idx_token >= 0 and token_categories[idx_token] == current_category:
                        idx_token -= 1
                    start_idx = idx_token + 1
                    seq_len = end_idx - start_idx + 1
                    if seq_len >= self.sequence_length:
                        seq_start = end_idx - self.sequence_length + 1
                        seq_slice = activation_data[:, seq_start:end_idx+1, :]
                        self.activations.append(seq_slice)
                        self.labels.append(category_label)
                else:
                    idx_token -= 1

        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        if self.verbose:
            print(f"Loaded {len(self.activations)} activation sequences from: {self.preprocessed_data_dir}")
            if len(self.labels) > 0:
                num_correct = int(self.labels.sum().item())
                num_incorrect = len(self.labels) - num_correct
                print(f"Number of correct activations: {num_correct}")
                print(f"Number of incorrect activations: {num_incorrect}")
            else:
                print("No activation sequences loaded.")

    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return {'activation': self.activations[idx], 'label': self.labels[idx]}


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    A PyTorch sampler for creating balanced batches of samples across two classes.

    Args:
        labels (torch.Tensor): Tensor of binary labels (0 or 1) for the dataset.
        batch_size (int): Number of samples per batch. Must be an even number.

    Attributes:
        labels (np.ndarray): Numpy array of dataset labels.
        label_to_indices (dict): Dictionary mapping each label to a list of corresponding sample indices.
        num_samples (int): Total number of samples to draw per epoch (twice the size of the smallest class).
        batch_size (int): Configured batch size.
    """
    def __init__(self, labels, batch_size):
        self.labels = labels.numpy()
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        # Find the minimum class size
        min_class_size = min(len(indices) for indices in self.label_to_indices.values())
        self.num_samples = min_class_size * 2  # Total samples per epoch

        self.batch_size = batch_size
        assert self.batch_size % 2 == 0, "Batch size should be even for balanced classes."

    def __iter__(self):
        indices = []
        for _ in range(self.num_samples // (self.batch_size // 2)):
            batch_indices = []
            for label in [0, 1]:
                label_indices = np.random.choice(
                    self.label_to_indices[label],
                    size=self.batch_size // 2,
                    replace=True
                )
                batch_indices.extend(label_indices)
            np.random.shuffle(batch_indices)
            indices.extend(batch_indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples



# Function to get the input size from a sample
def get_input_size(dataset):
    """
    Computes the input size for a dataset sample based on its activation tensor.

    Args:
        dataset (Dataset): A PyTorch dataset containing activation tensors.

    Returns:
        int: The input size, calculated as the product of sequence length and hidden size.
    """
    sample = dataset[0]
    seq_len, hidden_size = sample['activation'].shape
    return seq_len * hidden_size  # Input size is 100 * hidden_size

def get_hidden_size(dataset):
    """
    Retrieves the hidden size from the activation tensor of a dataset sample.

    Args:
        dataset (Dataset): A PyTorch dataset containing activation tensors.

    Returns:
        int: The hidden size of the activation tensors.
    """
    sample = dataset[0]
    sequence_length, hidden_size = sample['activation'].shape
    return hidden_size  # Return the hidden_size