import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
from collections import defaultdict
from arrangement_puzzle.seating_puzzle import load_puzzle_from_disk
from arrangement_puzzle.llm_parser import analyze_llm_output

class ActivationDataset(Dataset):
    """
    A PyTorch dataset for loading and processing LLM activations and corresponding labels.

    Args:
        root_dirs (list of str): Directories containing activation and input data (e.g., ['train/activations']).
        layers_to_load (list of int): Indices of model layers to load activations from (e.g., [0, 1, 2]).
        sequence_length (int): Number of tokens to consider in the activation sequences.
        puzzles_json_dir (str): Directory containing puzzle JSON files, used to load puzzle metadata.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing input text.

    Attributes:
        activations (list of torch.Tensor): Processed activation sequences for each sample.
        labels (torch.Tensor): Labels corresponding to the activation sequences.
        sequence_length (int): Configured sequence length for each activation sequence.
        puzzles_json_dir (str): Path to the directory containing puzzle metadata.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance used for text processing.
        layers_to_load (list of int): Layers selected for loading activations.
    """
    def __init__(self, root_dirs, layers_to_load, sequence_length=200, puzzles_json_dir=None, tokenizer=None):
        """
        Args:
            root_dirs (list of strings): Directories with all the data (e.g., ['train/activations']).
            layers_to_load (list of ints): List of layer indices to load (e.g., [0, 1, 2]).
            sequence_length (int): Number of tokens to consider from the end of the sequence.
            puzzles_json_dir (string): Directory containing the puzzle JSON files (e.g., 'path/to/puzzles').
            tokenizer: The tokenizer to use (needed for analyze_llm_output).
        """
        self.activations = []
        self.labels = []
        self.sequence_length = sequence_length
        self.puzzles_json_dir = puzzles_json_dir
        self.tokenizer = tokenizer
        self.layers_to_load = layers_to_load

        if not self.tokenizer:
            raise ValueError("Tokenizer must be provided.")

        for root_dir in root_dirs:
            for subdir in os.listdir(root_dir):
                subdir_path = os.path.join(root_dir, subdir)
                if os.path.isdir(subdir_path):
                    # Assume subdir name corresponds to the puzzle ID, e.g., '6000'
                    puzzle_id = subdir
                    puzzle_json_file = os.path.join(self.puzzles_json_dir, f"{puzzle_id}.json")

                    # Load the puzzle data
                    if not os.path.exists(puzzle_json_file):
                        print(f"Warning: Puzzle JSON file '{puzzle_json_file}' not found for puzzle ID '{puzzle_id}', skipping.")
                        continue

                    try:
                        clues, correct_arrangement, name_color_mapping = load_puzzle_from_disk(puzzle_json_file)
                    except Exception as e:
                        print(f"Error loading puzzle JSON file '{puzzle_json_file}': {e}")
                        continue

                    # Find the activation file and input string file
                    activation_file = None
                    input_string_file = None
                    for file_name in os.listdir(subdir_path):
                        if file_name == 'hidden_states.pt':
                            activation_file = os.path.join(subdir_path, file_name)
                        elif file_name == 'input.txt':
                            input_string_file = os.path.join(subdir_path, file_name)
                    if activation_file and input_string_file:
                        # Load the activation data
                        activation_data = torch.load(activation_file)  # Shape: [layers, tokens, hidden_size]

                        # Select specified layers
                        activation_data = activation_data[self.layers_to_load, :, :]  # Shape: [num_layers, tokens, hidden_size]

                        num_layers, num_tokens_activation, hidden_size = activation_data.shape

                        # Load the input string
                        with open(input_string_file, 'r') as f:
                            input_string = f.read()

                        # Generate the token_dict using analyze_llm_output
                        try:
                            token_dict, _ = analyze_llm_output(input_string, correct_arrangement, name_color_mapping, self.tokenizer)
                        except Exception as e:
                            print(f"Error analyzing LLM output for '{input_string_file}': {e}")
                            continue

                        # Tokenize the input string with offsets
                        encoded = self.tokenizer(input_string, return_offsets_mapping=True, add_special_tokens=False)
                        token_indices = list(range(len(encoded['input_ids'])))
                        token_categories = [token_dict.get(idx, 'unknown') for idx in token_indices]

                        num_tokens_tokenizer = len(encoded['input_ids'])

                        # Exclude prompt tokens
                        num_prompt_tokens = num_tokens_tokenizer - num_tokens_activation
                        if num_prompt_tokens < 0:
                            print(f"Warning: Number of prompt tokens is negative in '{activation_file}', skipping.")
                            continue

                        token_categories = token_categories[num_prompt_tokens:]

                        # Ensure activations and token_categories are aligned
                        if len(token_categories) != num_tokens_activation:
                            print(f"Warning: After excluding prompt tokens, activation length ({num_tokens_activation}) does not match token length ({len(token_categories)}), skipping.")
                            continue

                        # At this point, len(token_categories) == num_tokens_activation
                        # Proceed to find sequences of consecutive tokens with the same category
                        idx_token = len(token_categories) - 1
                        while idx_token >= 0:
                            current_category = token_categories[idx_token]
                            if current_category in ['solution_correct', 'solution_incorrect', 'clue_hallucination']:
                                # Start collecting tokens
                                end_idx = idx_token
                                category_label = 1 if current_category == 'solution_correct' else 0
                                while idx_token >= 0 and token_categories[idx_token] == current_category:
                                    idx_token -= 1
                                start_idx = idx_token + 1  # Index of the first token in this consecutive sequence
                                seq_length = end_idx - start_idx + 1
                                if seq_length >= self.sequence_length:
                                    # Take the last 'sequence_length' tokens
                                    seq_start = end_idx - self.sequence_length + 1
                                    activation_seq = activation_data[:, seq_start:end_idx+1, :]  # Shape: [num_layers, sequence_length, hidden_size]
                                    self.activations.append(activation_seq)
                                    self.labels.append(category_label)
                                else:
                                    print(f"Warning: Consecutive token sequence length ({seq_length}) is less than {self.sequence_length} in '{activation_file}', category '{current_category}', skipping.")
                            else:
                                idx_token -= 1
                    else:
                        if not activation_file:
                            print(f"Warning: Activation file 'hidden_states.pt' not found in '{subdir_path}'.")
                        if not input_string_file:
                            print(f"Warning: Input string file not found in '{subdir_path}'.")
        print(f"Loaded {len(self.activations)} activation sequences from directories: {root_dirs}.")

        # Convert labels to tensor for efficient indexing
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        # Calculate and print label counts
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
        activations = self.activations[idx]  # Shape: [num_layers, sequence_length, hidden_size]
        label = self.labels[idx]
        return {'activation': activations, 'label': label}


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