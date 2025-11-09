import torch
import torch.nn as nn
from compression_classifier import CompressionClassifier
from transformers.modeling_outputs import SequenceClassifierOutput


class CompressionProbeModel(nn.Module):
    """
    A wrapper that wraps a language model + trainable compression classifier.
    The forward pass will extract hidden states from the LM and pass them through a binary classifier.
    """
    
    def __init__(self, language_model, compression_classifier: CompressionClassifier):
        """
        Args:
            language_model
            compression_classifier: CompressionClassifier
        """
        super(CompressionProbeModel, self).__init__()
        
        self.language_model = language_model
        self.compression_classifier = compression_classifier
        
        for param in self.language_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        """
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            next_is_COMP_label: Binary labels for compression (batch_size, seq_len)
            labels: Alias for next_is_COMP_label (for HuggingFace compatibility)
                    Values: 1 (next token is COMP), 0 (next token is not COMP), -100 (ignore when computing loss)
        
        Returns:
            SequenceClassifierOutput with loss and logits and (optional) hidden states
        """
        # Use labels as fallback if next_is_COMP_label is not provided
        if next_is_COMP_label is None and labels is not None:
            next_is_COMP_label = labels
        # Get hidden states from the language model
        with torch.no_grad():
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
        
        # Pass the hidden states from the last layer to the classifier
        hidden_states = lm_outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        logits = self.compression_classifier(hidden_states)  # (batch_size, seq_len)
        
        loss = None
        if next_is_COMP_label is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            flat_logits = logits.view(-1)
            flat_labels = next_is_COMP_label.view(-1)
            
            valid_mask = (flat_labels != -100)
            assert valid_mask.sum() >= 0, "No valid positions to compute loss."
            
            losses = loss_fct(flat_logits[valid_mask], flat_labels[valid_mask].float())
            loss = losses.mean()
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
    
    def save_pretrained(self, save_directory):
        """Save only the classifier (LM is frozen)"""
        torch.save(
            self.compression_classifier.state_dict(),
            f"{save_directory}/compression_classifier.pt"
        )
    
    def load_classifier(self, load_path):
        """Load classifier weights"""
        self.compression_classifier.load_state_dict(
            torch.load(f"{load_path}/compression_classifier.pt")
        )
