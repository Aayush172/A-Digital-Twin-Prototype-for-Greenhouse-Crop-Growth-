"""
Unit tests for Linear Regression and Stacked LSTM Models.
"""

import pytest
import numpy as np
from models.linear_regression_model import BaselineLinearRegressionModel
from models.lstm_model import GreenhouseLSTMModel

def test_linear_regression_fit_predict():
    X_train = np.random.randn(50, 12 * 5)
    y_train = np.random.randn(50)

    model = BaselineLinearRegressionModel()
    model.fit(X_train, y_train)

    preds = model.predict(X_train)
    assert len(preds) == 50

def test_lstm_model_fit_predict():
    X_train = np.random.randn(40, 12, 5)
    y_train = np.random.randn(40)

    model = GreenhouseLSTMModel(sequence_length=12, n_features=5, lstm_units=[16, 8])
    res = model.fit(X_train, y_train, epochs=2, batch_size=16)

    assert "training_time_sec" in res
    preds = model.predict(X_train)
    assert len(preds) == 40
