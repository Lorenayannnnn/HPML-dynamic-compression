import torch

import torch.nn as nn


class CompressionClassifier(nn.Module):
    """
    Binary classifier that predicts whether the next token should be compressed.
    Takes hidden states from a language model and outputs binary predictions.
    """
    
    def __init__(self, hidden_size, dropout):
        """
        Args:
            hidden_size: Dimension of the input hidden states
            dropout: Dropout probability for regularization
        """
        super(CompressionClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            logits: Tensor of shape (batch_size, seq_len)
        """
        # Pass through classifier
        logits = self.classifier(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        return logits
    
    def predict(self, hidden_states, threshold=0.5):
        """
        Generate binary predictions.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            threshold: Decision threshold for binary classification
        
        Returns:
            predictions: Boolean tensor of shape (batch_size, seq_len)
        """
        logits, _ = self.forward(hidden_states)
        probs = torch.sigmoid(logits)
        return probs > threshold