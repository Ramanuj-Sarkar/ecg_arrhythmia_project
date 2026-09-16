"""
model.py
========
A 1D-CNN + BiLSTM architecture for ECG arrhythmia classification.

Design rationale (worth saying out loud in an interview):
  - 1D convolutions first: learn local morphological features (QRS complex
    shape, P-wave/T-wave patterns) that are translation-invariant in time.
  - BiLSTM on top: captures longer-range temporal dependencies and rhythm
    irregularities that convolutions alone tend to miss.
  - MC Dropout (dropout left active at inference time, see uncertainty.py):
    gives a cheap, principled uncertainty estimate without a full Bayesian
    reformulation -- appropriate for a first-pass clinical decision-support
    tool where "I don't know" needs to be a valid model output.

Swap-in options if you want to extend this for the resume project:
  - Replace the CNN+LSTM encoder with a small Transformer encoder
    (nn.TransformerEncoder) over patch-embedded ECG segments.
  - For imaging modalities (echo, CT), see the JD's mention of MONAI /
    vision transformers -- out of scope here but worth name-dropping
    as "the next modality I'd extend this to."
"""

import torch
import torch.nn as nn


class ECGClassifier(nn.Module):
    def __init__(self, n_leads=1, n_classes=4, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=64, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_leads, timesteps)
        z = self.conv(x)                      # (batch, channels, T')
        z = self.dropout(z)
        z = z.permute(0, 2, 1)                # (batch, T', channels) for LSTM
        z, _ = self.lstm(z)                   # (batch, T', 2*hidden)
        z = z.mean(dim=1)                     # temporal average pooling
        logits = self.head(z)                 # (batch, n_classes)
        return logits


if __name__ == "__main__":
    m = ECGClassifier(n_leads=1, n_classes=4)
    x = torch.randn(8, 1, 500)
    out = m(x)
    print("output shape:", out.shape)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {n_params:,}")
