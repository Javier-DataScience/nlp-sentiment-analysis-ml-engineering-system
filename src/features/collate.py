"""
collate.py

This module defines how individual samples from the dataset are combined into batches
for PyTorch training.

Responsibilities:
- Pad variable-length sequences to the same length
- Convert lists/tensors into a uniform tensor batch
- Prepare data for DataLoader

This version is robust to both:
- Python lists
- PyTorch tensors
"""

import torch


def pad_sequence(sequence, max_len, pad_value=0):
    """
    Pads or truncates a sequence to a fixed length.

    Handles both list and torch.Tensor inputs safely.
    """

    # Convert tensor → list if needed
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()

    # Pad
    if len(sequence) < max_len:
        sequence = sequence + [pad_value] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]

    return sequence


def collate_batch(batch):
    """
    Converts a list of dataset samples into a padded batch.

    Each item in batch is expected to be:
    {
        "text": List[int] or Tensor[int],
        "label": int or Tensor[int]
    }
    """

    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]

    # Ensure we work with lists internally
    texts = [t.tolist() if isinstance(t, torch.Tensor) else t for t in texts]

    max_len = max(len(t) for t in texts)

    padded_texts = [pad_sequence(t, max_len) for t in texts]

    # Convert to tensors
    text_tensor = torch.tensor(padded_texts, dtype=torch.long)

    # Labels → tensor (handle both int and tensor cases)
    if isinstance(labels[0], torch.Tensor):
        label_tensor = torch.stack(labels)
    else:
        label_tensor = torch.tensor(labels, dtype=torch.long)

    return {"text": text_tensor, "labels": label_tensor}
