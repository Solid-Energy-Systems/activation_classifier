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
    """
    A PyTorch dataset for loading and processing LLM activations and corresponding labels.

    Args:
        root_dirs (list of str): Directories containing activation and input data (e.g., ['train/activations']).
        layers_to_load (list of int): Indices of model layers to load activations from (e.g., [0, 1, 2]).
        sequence_length (int): Number of tokens to consider in the activation sequences.
        verbose (bool): If True, prints informational messages and warnings (default is True).

    Attributes:
        activations (list of torch.Tensor): Processed activation sequences for each sample.
        labels (torch.Tensor): Labels corresponding to the activation sequences.
        sequence_length (int): Configured sequence length for each activation sequence.
        layers_to_load (list of int): Layers selected for loading activations.
        verbose (bool): Whether to print informational messages and warnings.
    """
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

        # Iterate over preprocessed h5/json pairs
        for file_name in os.listdir(self.preprocessed_data_dir):
            if file_name.endswith('.h5'):
                puzzle_id = file_name.replace('.h5', '')
                h5_filepath = os.path.join(self.preprocessed_data_dir, file_name)
                json_filepath = os.path.join(self.preprocessed_data_dir, f"{puzzle_id}.json")

                # Check for corresponding JSON
                if not os.path.exists(json_filepath):
                    if self.verbose:
                        print(f"Warning: Missing JSON file for puzzle_id={puzzle_id}, skipping.")
                    continue

                # Load activation metadata
                with h5py.File(h5_filepath, 'r') as hf:
                    num_layers = hf.attrs['num_layers']
                    num_tokens = hf.attrs['num_tokens']

                # Load precomputed tokenization and category info
                with open(json_filepath, 'r') as jf:
                    data = json.load(jf)
                    input_string = data['input_string']
                    input_ids = data['input_ids']
                    token_dict = {int(k): v for k, v in data['token_dict'].items()}  # Convert keys to integers

                # Determine prompt length:
                # You know from preprocessing how activations align with tokens.
                # The original code excluded prompt tokens by comparing lengths.
                # If we stored that information differently, we could just load it here.
                # Assuming `num_tokens` is the number of tokens in activations:
                num_prompt_tokens = len(input_ids) - num_tokens
                if num_prompt_tokens < 0:
                    # Inconsistent data, skip
                    if self.verbose:
                        print(f"Warning: negative prompt tokens for puzzle_id={puzzle_id}, skipping.")
                    continue

                # Construct category array for tokens corresponding to activations
                # token_dict is keyed by token index in the full input sequence
                # We need to map them to categories after the prompt.
                token_categories = []
                for idx in range(num_prompt_tokens, len(input_ids)):
                    cat = token_dict.get(idx, 'unknown')
                    token_categories.append(cat)

                # Now, load the required layers from the h5 file
                # We'll load once and slice sequences.
                with h5py.File(h5_filepath, 'r') as hf:
                    # Stack the requested layers into a single tensor
                    layer_data = []
                    for lidx in self.layers_to_load:
                        layer_data.append(hf[f"layer_{lidx}"][:])  # shape: [tokens, hidden_size]
                    # layer_data: list of arrays, each [tokens, hidden_size]
                    # Stack into a single array [num_layers, tokens, hidden_size]
                    activation_data = torch.tensor(np.array(layer_data))

                # Find sequences of consecutive tokens with same category
                idx_token = len(token_categories) - 1
                while idx_token >= 0:
                    current_category = token_categories[idx_token]
                    if current_category in ['solution_correct', 'solution_incorrect', 'clue_hallucination']:
                        end_idx = idx_token
                        category_label = 1 if current_category == 'solution_correct' else 0
                        # Move backwards until category changes
                        while idx_token >= 0 and token_categories[idx_token] == current_category:
                            idx_token -= 1
                        start_idx = idx_token + 1
                        seq_length = end_idx - start_idx + 1
                        if seq_length >= self.sequence_length:
                            seq_start = end_idx - self.sequence_length + 1
                            activation_seq = activation_data[:, seq_start:end_idx+1, :]
                            self.activations.append(activation_seq)
                            self.labels.append(category_label)
                        else:
                            # skip sequence if shorter than required
                            pass
                    else:
                        idx_token -= 1

        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        if self.verbose:
            print(f"Loaded {len(self.activations)} activation sequences from: {self.preprocessed_data_dir}")
            if len(self.labels) > 0:
                num_correct = self.labels.sum().item()
                num_incorrect = len(self.labels) - num_correct
                print(f"Number of correct activations: {int(num_correct)}")
                print(f"Number of incorrect activations: {int(num_incorrect)}")
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