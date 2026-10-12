"""
snn_robust_tau.loss
===================
Loss functions for training robust SNNs.
"""

from .asar import (
    ASARLoss,
    ASARLossV2,
    OnlineASAR,
    OnlinePredictionError,
    OnlineASARHardState,
    OnlineASARSoftState,
    prepare_mem_for_asar,
    get_asar_loss,
)

from .hajr import (
    HAJRLoss,
    OnlineHAJR,
    get_hajr_loss,
)

__all__ = [
    # ASAR
    'ASARLoss',
    'ASARLossV2',
    'OnlineASAR',
    'OnlinePredictionError',
    'OnlineASARHardState',
    'OnlineASARSoftState',
    'prepare_mem_for_asar',
    'get_asar_loss',
    # HAJR
    'HAJRLoss',
    'OnlineHAJR',
    'get_hajr_loss',
]
