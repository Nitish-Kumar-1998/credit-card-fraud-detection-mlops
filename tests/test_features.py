import pandas as pd
import numpy as np
import pytest

def test_log1p_transform():
    amount = pd.Series([0, 10, 100, 1000])
    transformed = np.log1p(amount)
    
    # log1p(0) should be 0
    assert transformed[0] == 0
    # values should be compressed
    assert transformed.max() < amount.max()
    # no negative values
    assert (transformed >= 0).all()

def test_time_normalization():
    time = pd.Series([0, 50000, 100000, 172792])
    normalized = (time - time.min()) / (time.max() - time.min())
    
    # min should be 0
    assert normalized.min() == 0
    # max should be 1
    assert normalized.max() == 1
    # all values between 0 and 1
    assert (normalized >= 0).all()
    assert (normalized <= 1).all()

def test_no_missing_values():
    df = pd.DataFrame({
        'Amount': [10, 20, 30],
        'Time': [100, 200, 300],
        'Class': [0, 0, 1]
    })
    assert df.isnull().sum().sum() == 0

def test_class_imbalance():
    y = pd.Series([0] * 99 + [1] * 1)
    fraud_pct = y.mean() * 100
    assert fraud_pct < 5  # less than 5% fraud