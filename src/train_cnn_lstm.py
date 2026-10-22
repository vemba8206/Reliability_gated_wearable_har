"""Sensor-wise CNN-LSTM hybrid deep-learning classifier in PyTorch."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


class CNNLSTMModel(nn.Module):
    """
    Compact hybrid CNN-LSTM network for 6-channel wearable IMU window classification.
    1D-CNN extracts local temporal-spatial feature representations.
    LSTM models temporal sequence dynamics across the receptive field.
    """
    def __init__(
        self,
        in_channels: int = 6,
        n_classes: int = 6,
        conv1_filters: int = 32,
        conv1_kernel: int = 7,
        conv2_filters: int = 64,
        conv2_kernel: int = 5,
        lstm_hidden: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        # Conv block 1: (batch, 6, 256) -> (batch, 32, 128)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1_filters, kernel_size=conv1_kernel, padding=conv1_kernel // 2),
            nn.BatchNorm1d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Conv block 2: (batch, 32, 128) -> (batch, 64, 64)
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv1_filters, conv2_filters, kernel_size=conv2_kernel, padding=conv2_kernel // 2),
            nn.BatchNorm1d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # LSTM: input size is conv2_filters, hidden size is lstm_hidden
        self.lstm = nn.LSTM(
            input_size=conv2_filters,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 256, 6) -> transpose to (batch, 6, 256) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        # Transpose to (batch, seq_len, channels) for LSTM
        x = x.transpose(1, 2)
        out, (hn, cn) = self.lstm(x)
        # Use last LSTM hidden state
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits


class SensorCNNLSTM:
    """Wrapper for training, evaluating, and checkpointing Sensor-Wise CNN-LSTM."""

    def __init__(
        self,
        sensor_name: str,
        class_list: List[int],
        in_channels: int = 6,
        conv1_filters: int = 32,
        conv1_kernel: int = 7,
        conv2_filters: int = 64,
        conv2_kernel: int = 5,
        lstm_hidden: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 15,
        patience: int = 3,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        self.sensor_name = sensor_name
        self.canonical_classes = sorted(list(class_list))
        self.n_classes = len(self.canonical_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.canonical_classes)}

        self.in_channels = in_channels
        self.conv1_filters = conv1_filters
        self.conv1_kernel = conv1_kernel
        self.conv2_filters = conv2_filters
        self.conv2_kernel = conv2_kernel
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        self.net = CNNLSTMModel(
            in_channels=in_channels,
            n_classes=self.n_classes,
            conv1_filters=conv1_filters,
            conv1_kernel=conv1_kernel,
            conv2_filters=conv2_filters,
            conv2_kernel=conv2_kernel,
            lstm_hidden=lstm_hidden,
            dropout=dropout
        ).to(self.device)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.is_trained = False
        self.train_duration_sec: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "SensorCNNLSTM":
        """
        Fit model using PyTorch with early stopping on validation loss.
        X_train: (N, 256, 6)
        y_train: (N,) activity IDs
        """
        logger.info(f"Training {self.sensor_name} CNN-LSTM on {len(y_train)} windows ({self.device})...")
        t0 = time.time()

        # Map activity IDs to consecutive 0..K-1 indices
        y_train_idx = np.array([self.class_to_idx[c] for c in y_train], dtype=np.int64)

        # Handle NaNs safely
        X_tr = np.nan_to_num(X_train, nan=0.0)
        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_train_idx, dtype=torch.long)
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if has_val:
            y_val_idx = np.array([self.class_to_idx[c] for c in y_val], dtype=np.int64)
            X_v = np.nan_to_num(X_val, nan=0.0)
            val_ds = TensorDataset(
                torch.tensor(X_v, dtype=torch.float32),
                torch.tensor(y_val_idx, dtype=torch.long)
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Class weights for loss
        counts = np.bincount(y_train_idx, minlength=self.n_classes)
        weights = len(y_train_idx) / (self.n_classes * np.maximum(counts, 1).astype(np.float32))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            total_batches = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_batches += 1

            avg_train_loss = epoch_loss / max(1, total_batches)
            self.history["train_loss"].append(avg_train_loss)

            if has_val:
                self.net.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        v_logits = self.net(vx)
                        v_loss += criterion(v_logits, vy).item()
                        preds = torch.argmax(v_logits, dim=1)
                        v_correct += (preds == vy).sum().item()
                        v_total += len(vy)

                avg_val_loss = v_loss / max(1, len(val_loader))
                val_acc = v_correct / max(1, v_total)
                self.history["val_loss"].append(avg_val_loss)
                self.history["val_acc"].append(val_acc)
                self.net.train()

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.train_duration_sec = time.time() - t0
        self.is_trained = True
        logger.info(f"{self.sensor_name} CNN-LSTM trained in {self.train_duration_sec:.2f}s.")
        return self

    def predict_proba_aligned(self, X_windows: np.ndarray) -> np.ndarray:
        """Predict softmax probabilities aligned to canonical_classes."""
        if not self.is_trained:
            raise RuntimeError(f"CNN-LSTM for {self.sensor_name} is not trained.")

        N = len(X_windows)
        if N == 0:
            return np.empty((0, self.n_classes), dtype=np.float32)

        X_clean = np.nan_to_num(X_windows, nan=0.0)
        tensor_x = torch.tensor(X_clean, dtype=torch.float32).to(self.device)

        self.net.eval()
        probs_list = []
        batch_size = max(64, self.batch_size)
        with torch.no_grad():
            for i in range(0, N, batch_size):
                bx = tensor_x[i:i + batch_size]
                logits = self.net(bx)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        all_probs = np.concatenate(probs_list, axis=0)
        return all_probs

    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        """Predict canonical activity IDs directly."""
        probs = self.predict_proba_aligned(X_windows)
        best_indices = np.argmax(probs, axis=1)
        return np.array([self.canonical_classes[idx] for idx in best_indices])

    def save(self, filepath: Path) -> None:
        """Save model state dict and architecture parameters."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "sensor_name": self.sensor_name,
            "canonical_classes": self.canonical_classes,
            "in_channels": self.in_channels,
            "conv1_filters": self.conv1_filters,
            "conv1_kernel": self.conv1_kernel,
            "conv2_filters": self.conv2_filters,
            "conv2_kernel": self.conv2_kernel,
            "lstm_hidden": self.lstm_hidden,
            "dropout": self.dropout,
            "history": self.history,
            "train_duration_sec": self.train_duration_sec
        }, filepath)

    @classmethod
    def load(cls, filepath: Path, device: Optional[str] = None) -> "SensorCNNLSTM":
        """Load saved checkpoint."""
        ckpt = torch.load(filepath, map_location=device or "cpu")
        inst = cls(
            sensor_name=ckpt["sensor_name"],
            class_list=ckpt["canonical_classes"],
            in_channels=ckpt["in_channels"],
            conv1_filters=ckpt["conv1_filters"],
            conv1_kernel=ckpt["conv1_kernel"],
            conv2_filters=ckpt["conv2_filters"],
            conv2_kernel=ckpt["conv2_kernel"],
            lstm_hidden=ckpt["lstm_hidden"],
            dropout=ckpt["dropout"],
            device=device
        )
        inst.net.load_state_dict(ckpt["state_dict"])
        inst.history = ckpt.get("history", {})
        inst.train_duration_sec = ckpt.get("train_duration_sec", 0.0)
        inst.is_trained = True
        return inst
