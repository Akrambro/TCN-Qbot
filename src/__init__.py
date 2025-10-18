"""
TCN Forex Trading Bot - Source Module

PyTorch-based Temporal Convolutional Network for Forex direction prediction
"""

__version__ = '1.0.0'

from .tcn_model import TCNForex, TCNTrainer
from .data_preprocessing import ForexDataPreprocessor

__all__ = [
    'TCNForex',
    'TCNTrainer',
    'ForexDataPreprocessor'
]
