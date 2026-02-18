import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocess import prepare_data, WINDOW_SIZE, FEATURE_COLS

# Intra-op parallelism: use all cores for matrix ops (LSTM, attention, etc.)
torch.set_num_threads(os.cpu_count() or 4)
# Data loading workers: keep low on Windows to avoid spawning too many torch processes
NUM_WORKERS = 2


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class StockDataset(Dataset):
    """Wraps numpy arrays from preprocess.prepare_data into a PyTorch Dataset."""

    def __init__(self, X, y):
        """
        Args:
            X: np.ndarray (num_samples, window_size, num_features)
            y: np.ndarray (num_samples, 2) — [6mo_close_return, 6mo_dividend_yield]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────
# Positional Encoding for Transformer
# ──────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the transformer layers."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


# ──────────────────────────────────────────────
# LSTM + Transformer Hybrid Model
# ──────────────────────────────────────────────

class StockPredictor(nn.Module):
    """
    Hybrid LSTM → Transformer → Regression head.

    Pipeline:
        1. LSTM extracts sequential patterns from the input window
        2. Transformer attends over LSTM outputs for global context
        3. Linear head predicts close_return and dividend_yield
    """

    def __init__(
        self,
        num_features=len(FEATURE_COLS),
        d_model=64,
        lstm_layers=2,
        transformer_heads=4,
        transformer_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        # Project input features to d_model dimensions
        self.input_proj = nn.Linear(num_features, d_model)

        # LSTM for sequential pattern extraction
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(d_model, max_len=WINDOW_SIZE)

        # Transformer encoder for global attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Regression head: 2 outputs (close_return, dividend_yield)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, window_size, num_features)
        Returns:
            (batch, 2) — predictions for [close_return, dividend_yield]
        """
        # Project to d_model
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Add positional encoding + transformer
        x = self.pos_encoder(lstm_out)
        x = self.transformer(x)
        x = self.layer_norm(x)

        # Use last timestep as summary
        x = x[:, -1, :]
        x = self.dropout(x)

        return self.head(x)


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────

class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"  Early stopping after {self.patience} epochs without improvement")
            return True
        return False

    def restore_best(self, model):
        """Load the best model weights back."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_model(
    splits,
    batch_size=64,
    epochs=100,
    lr=1e-3,
    patience=10,
    device=None,
):
    """Full training loop with validation monitoring and early stopping."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaders
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]

    train_loader = DataLoader(
        StockDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=True,
    )
    val_loader = DataLoader(
        StockDataset(X_val, y_val), batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=True,
    )

    # Model, loss, optimizer, scheduler
    model = StockPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    stopper = EarlyStopping(patience=patience)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Batch: {batch_size}")
    print(f"{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(X_train)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(X_val)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {train_loss:>12.8f}  {val_loss:>12.8f}  {current_lr:>10.6f}")

        scheduler.step(val_loss)

        if stopper.step(val_loss, model):
            break

    # Restore best model
    stopper.restore_best(model)
    print(f"\nBest validation loss: {stopper.best_loss:.8f}")

    return model


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(model, splits, device=None):
    """Evaluate on test set. Reports MSE and directional accuracy for close returns."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test, y_test, meta = splits["test"]
    test_loader = DataLoader(
        StockDataset(X_test, y_test), batch_size=64, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=True,
    )

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # MSE per target
    mse_close = ((preds[:, 0] - targets[:, 0]) ** 2).mean()
    mse_div = ((preds[:, 1] - targets[:, 1]) ** 2).mean()

    # Directional accuracy: did we predict the sign of close return correctly?
    pred_dir = (preds[:, 0] > 0).astype(int)
    true_dir = (targets[:, 0] > 0).astype(int)
    dir_accuracy = (pred_dir == true_dir).mean()

    date_range = f"{meta[0]['date']} to {meta[-1]['date']}"
    print(f"\nTest Results ({len(X_test):,} samples, {date_range}):")
    print(f"  Close return MSE:       {mse_close:.8f}")
    print(f"  Dividend yield MSE:     {mse_div:.10f}")
    print(f"  Direction accuracy:     {dir_accuracy:.2%}")
    print(f"  (Baseline ~50% for random, higher for always-up over 6mo horizon)")

    return {"mse_close": mse_close, "mse_div": mse_div, "dir_accuracy": dir_accuracy}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Stock Prediction Model")
    print("=" * 50)

    print("\n[1/3] Preparing data...")
    splits, scalers = prepare_data()

    print("\n[2/3] Training model...")
    model = train_model(splits, batch_size=32, epochs=100, patience=10)

    print("\n[3/3] Evaluating on test set...")
    results = evaluate(model, splits)

    # Save model weights
    save_path = "model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
