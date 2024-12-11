import torch
import torch.nn as nn

class ActivationClassifier(nn.Module):
    """
    A neural network for classifying activation patterns in transformer layers.

    This model processes activation tensors from multiple layers, sequences, and hidden dimensions,
    and predicts a binary classification output using a combination of convolutional and fully connected layers.

    Attributes:
        num_layers (int): Number of transformer layers.
        sequence_length (int): Length of the sequence to process.
        hidden_size (int): Size of the hidden dimension in transformer activations.
        conv1 (nn.Conv1d): Convolutional layer operating over the sequence dimension.
        relu (nn.ReLU): ReLU activation function.
        pool (nn.MaxPool1d): Max pooling layer over the sequence dimension.
        fc1 (nn.Linear): Fully connected layer that processes the flattened convolutional output.
        bn1 (nn.BatchNorm1d): Batch normalization layer applied after `fc1`.
        fc2 (nn.Linear): Fully connected layer reducing the feature space further.
        bn2 (nn.BatchNorm1d): Batch normalization layer applied after `fc2`.
        fc3 (nn.Linear): Fully connected layer producing a single output.
        sigmoid (nn.Sigmoid): Sigmoid activation function for binary classification.
    """

    def __init__(self, num_layers, sequence_length, hidden_size):
        """
        Initializes the ActivationClassifier.

        Args:
            num_layers (int): Number of transformer layers.
            sequence_length (int): Length of the sequence to process.
            hidden_size (int): Size of the hidden dimension in transformer activations.
        """
        super(ActivationClassifier, self).__init__()
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        # Total number of channels after reshaping
        total_channels = num_layers * hidden_size

        # Convolutional layers over the sequence dimension
        self.conv1 = nn.Conv1d(
            in_channels=total_channels,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after convolution and pooling
        conv_output_size = self._get_conv_output_size(sequence_length)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size * 128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _get_conv_output_size(self, seq_length):
        """
        Computes the output sequence length after applying convolution and pooling layers.

        Args:
            seq_length (int): Input sequence length.

        Returns:
            int: The output sequence length after convolution and pooling.
        """
        # After Conv1d with padding=1 and kernel_size=3, output length remains the same
        conv_length = seq_length
        # After MaxPool1d with kernel_size=2, output length is halved
        pooled_length = conv_length // 2
        return pooled_length

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_layers, sequence_length, hidden_size].

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, 1], representing classification scores.
        """
        # x shape: [batch_size, num_layers, sequence_length, hidden_size]
        batch_size = x.size(0)
        # Reshape x to merge num_layers and hidden_size into channels
        x = x.view(batch_size, self.num_layers * self.hidden_size, self.sequence_length)  # Shape: [batch_size, total_channels, sequence_length]
        x = self.conv1(x)  # Conv1d over sequence dimension
        x = self.relu(x)
        x = self.pool(x)  # Pooling over sequence dimension
        x = x.view(batch_size, -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x