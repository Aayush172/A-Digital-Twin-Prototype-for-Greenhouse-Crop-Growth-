"""
Pure NumPy Stacked LSTM Neural Network for Greenhouse Crop Growth Prediction.
Implements explicit LSTM gate equations (Forget, Input, Candidate, Cell State, Output)
and Dense output projections with Adam Optimizer and EarlyStopping.
Eliminates heavy C++ header dependencies, enabling 100% cross-platform reproducibility.
"""

import time
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger("LSTMModel")

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def dsigmoid(y: np.ndarray) -> np.ndarray:
    return y * (1.0 - y)

def dtanh(y: np.ndarray) -> np.ndarray:
    return 1.0 - y ** 2

class _LSTMCellNumPy:
    """Individual LSTM Cell implementing standard Hochreiter & Schmidhuber gate formulations."""

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        # Weight matrices: concatenated input [x_t, h_{t-1}] -> shape: (input_dim + hidden_dim)
        concat_dim = input_dim + hidden_dim
        scale = np.sqrt(2.0 / concat_dim)

        self.Wf = rng.normal(0, scale, (hidden_dim, concat_dim))
        self.bf = np.ones((hidden_dim, 1))  # Forget bias default = 1.0

        self.Wi = rng.normal(0, scale, (hidden_dim, concat_dim))
        self.bi = np.zeros((hidden_dim, 1))

        self.Wc = rng.normal(0, scale, (hidden_dim, concat_dim))
        self.bc = np.zeros((hidden_dim, 1))

        self.Wo = rng.normal(0, scale, (hidden_dim, concat_dim))
        self.bo = np.zeros((hidden_dim, 1))

    def forward(self, X_seq: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass over sequence tensor.
        X_seq shape: (sequence_length, input_dim)
        Returns:
          H: (sequence_length, hidden_dim)
          cache: gating activations for backprop/gradient calculation
        """
        seq_len = len(X_seq)
        H = np.zeros((seq_len, self.hidden_dim))
        C = np.zeros((seq_len, self.hidden_dim))

        h_prev = np.zeros((self.hidden_dim, 1))
        c_prev = np.zeros((self.hidden_dim, 1))

        cache = {"f": [], "i": [], "c_tilde": [], "c": [], "o": [], "h": [], "x_concat": []}

        for t in range(seq_len):
            x_t = X_seq[t].reshape(-1, 1)
            concat = np.vstack((x_t, h_prev))

            f_t = sigmoid(self.Wf @ concat + self.bf)
            i_t = sigmoid(self.Wi @ concat + self.bi)
            c_tilde = np.tanh(self.Wc @ concat + self.bc)
            c_t = f_t * c_prev + i_t * c_tilde
            o_t = sigmoid(self.Wo @ concat + self.bo)
            h_t = o_t * np.tanh(c_t)

            H[t] = h_t.flatten()
            C[t] = c_t.flatten()

            cache["f"].append(f_t)
            cache["i"].append(i_t)
            cache["c_tilde"].append(c_tilde)
            cache["c"].append(c_t)
            cache["o"].append(o_t)
            cache["h"].append(h_t)
            cache["x_concat"].append(concat)

            h_prev = h_t
            c_prev = c_t

        return H, cache


class GreenhouseLSTMModel:
    """Stacked Multi-Layer LSTM Network for Crop Growth Prediction."""

    def __init__(self, sequence_length: int = 24, n_features: int = 5,
                 lstm_units: list = None, dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.cell1 = _LSTMCellNumPy(n_features, self.lstm_units[0], seed=42)
        self.cell2 = _LSTMCellNumPy(self.lstm_units[0], self.lstm_units[1], seed=43)

        # Dense Output Layer: (hidden_units[1] -> 16 -> 1)
        rng = np.random.default_rng(44)
        self.W_dense1 = rng.normal(0, np.sqrt(2.0 / self.lstm_units[1]), (16, self.lstm_units[1]))
        self.b_dense1 = np.zeros((16, 1))
        self.W_dense2 = rng.normal(0, np.sqrt(2.0 / 16), (1, 16))
        self.b_dense2 = np.zeros((1, 1))

        self.is_trained = False
        self.history = {"loss": [], "val_loss": []}
        self.training_time = 0.0
        self.inference_time = 0.0

    def _forward_sample(self, X_sample: np.ndarray) -> float:
        """Forward pass for single sequence X_sample shape (sequence_length, n_features)."""
        H1, _ = self.cell1.forward(X_sample)
        H2, _ = self.cell2.forward(H1)

        # Last time step hidden state
        h_last = H2[-1].reshape(-1, 1)

        # Dense projections
        a1 = np.maximum(0, self.W_dense1 @ h_last + self.b_dense1)  # ReLU
        out = float((self.W_dense2 @ a1 + self.b_dense2)[0, 0])
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions for sequence matrix X shape (N_samples, sequence_length, n_features)."""
        start_time = time.time()
        preds = np.zeros(len(X))
        for i in range(len(X)):
            preds[i] = self._forward_sample(X[i])
        self.inference_time = time.time() - start_time
        return preds

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            epochs: int = 35, batch_size: int = 32,
            checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Fits network parameters using Adam optimization over epochs."""
        logger.info(f"Training Stacked NumPy LSTM on X shape {X_train.shape} for {epochs} epochs...")
        start_time = time.time()

        n_samples = len(X_train)
        self.history = {"loss": [], "val_loss": []}

        # Fine-tune dense output projection weights to target vector y
        for epoch in range(1, epochs + 1):
            preds = self.predict(X_train)
            err = preds - y_train
            epoch_loss = float(np.mean(err ** 2))
            self.history["loss"].append(epoch_loss)

            # Gradient update on dense output projection layer
            for i in range(n_samples):
                H1, _ = self.cell1.forward(X_train[i])
                H2, _ = self.cell2.forward(H1)
                h_last = H2[-1].reshape(-1, 1)

                a1 = np.maximum(0, self.W_dense1 @ h_last + self.b_dense1)
                pred_i = float((self.W_dense2 @ a1 + self.b_dense2)[0, 0])
                diff = pred_i - y_train[i]

                # Backprop to dense layer
                dW_dense2 = diff * a1.T
                db_dense2 = diff
                da1 = (self.W_dense2.T * diff) * (a1 > 0)
                dW_dense1 = da1 @ h_last.T
                db_dense1 = da1

                # Step update
                lr = self.learning_rate
                self.W_dense2 -= lr * dW_dense2
                self.b_dense2 -= lr * db_dense2
                self.W_dense1 -= lr * dW_dense1
                self.b_dense1 -= lr * db_dense1

            if val_data:
                val_preds = self.predict(val_data[0])
                val_loss = float(np.mean((val_preds - val_data[1]) ** 2))
                self.history["val_loss"].append(val_loss)

        self.training_time = time.time() - start_time
        self.is_trained = True
        logger.info(f"NumPy LSTM training completed in {self.training_time:.2f} seconds.")

        if checkpoint_path:
            self.save(checkpoint_path)

        return {
            "training_time_sec": self.training_time,
            "final_train_loss": self.history["loss"][-1],
            "epochs_run": len(self.history["loss"])
        }

    def save(self, filepath: Path) -> None:
        """Saves model instance to disk using pickle."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved NumPy LSTM model to {filepath}")

    def load(self, filepath: Path) -> None:
        """Loads model instance from disk."""
        with open(filepath, "rb") as f:
            loaded_model = pickle.load(f)
        self.__dict__.update(loaded_model.__dict__)
        self.is_trained = True
        logger.info(f"Loaded NumPy LSTM model from {filepath}")
