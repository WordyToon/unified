"""
Core hardware configuration and model definition modules for c_engine.
"""

from .hw_config import SYS_BITS, HW_CONFIG
from .xmodel import (
    XModel,
    XBundle,
    XConvBN,
    XDense,
    XPool,
    XActivation,
    keras
)

__all__ = [
    'SYS_BITS',
    'HW_CONFIG',
    'XModel',
    'XBundle',
    'XConvBN',
    'XDense',
    'XPool',
    'XActivation',
    'keras'
]

