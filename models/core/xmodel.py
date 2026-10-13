"""
================================================================================
Standardized Deep Learning Model & Hardware-Aware Layer Specification (XModel)
================================================================================
Provides standard architectural components:
  - XModel: Base model class supporting modular bundles & hardware co-design
  - XBundle: Modular execution bundle combining Core (Conv/Dense) + Pool + Residual Add
  - XConvBN: Quantized Convolutional Layer with integrated Batch Normalization
  - XDense: Quantized Fully-Connected / GEMV Layer
  - XPool: 2D Spatial Pooling Layer (Max / Average)
  - XActivation: Quantized Activation Function (ReLU, LeakyReLU, None)
================================================================================
"""

import sys
import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple
from .hw_config import SYS_BITS, HW_CONFIG

# Standalone / Keras compatibility wrapper
try:
    import keras
    register_serializable = keras.saving.register_keras_serializable
except Exception:
    class _DummyKerasSaving:
        @staticmethod
        def register_keras_serializable(*args, **kwargs):
            def decorator(cls):
                return cls
            return decorator

    class _DummyKeras:
        saving = _DummyKerasSaving()

    keras = _DummyKeras()


class XActivation:
    """Quantized Activation Unit specification for Accelerator PPU."""
    def __init__(self, sys_bits: SYS_BITS, o_int_bits: int = 0, type: Optional[str] = 'relu', slope: float = 0.0):
        self.sys_bits = sys_bits
        self.o_int_bits = o_int_bits
        self.type = type.lower() if type is not None else None
        self.slope = slope

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.type == 'relu':
            return np.maximum(0, x)
        elif self.type == 'leaky_relu':
            return np.where(x >= 0, x, (x * self.slope).astype(x.dtype))
        return x

    def __repr__(self) -> str:
        return f"XActivation(type={self.type}, slope={self.slope}, o_int_bits={self.o_int_bits})"


class XPool:
    """2D Spatial Pooling Layer (Max / Average)."""
    def __init__(self, type: str = 'max', pool_size: int = 2, strides: int = 2, 
                 padding: str = 'same', act: Optional[XActivation] = None):
        self.type = type.lower()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.lower()
        self.act = act

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Expected shape: (H, W, C) or (N, H, W, C)
        is_4d = (x.ndim == 4)
        if not is_4d:
            x = np.expand_dims(x, axis=0)
        
        N, H, W, C = x.shape
        out_h = (H + self.strides - 1) // self.strides if self.padding == 'same' else (H - self.pool_size) // self.strides + 1
        out_w = (W + self.strides - 1) // self.strides if self.padding == 'same' else (W - self.pool_size) // self.strides + 1
        
        out = np.zeros((N, out_h, out_w, C), dtype=x.dtype)
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.strides
                h_end = min(h_start + self.pool_size, H)
                w_start = j * self.strides
                w_end = min(w_start + self.pool_size, W)
                
                patch = x[:, h_start:h_end, w_start:w_end, :]
                if self.type == 'max':
                    out[:, i, j, :] = np.max(patch, axis=(1, 2))
                elif self.type == 'avg':
                    out[:, i, j, :] = np.mean(patch, axis=(1, 2)).astype(x.dtype)

        if self.act:
            out = self.act(out)

        return out[0] if not is_4d else out

    def __repr__(self) -> str:
        return f"XPool(type='{self.type}', pool_size={self.pool_size}, strides={self.strides}, padding='{self.padding}')"


class XConvBN:
    """Quantized 2D Convolution Layer with integrated BatchNorm and Activation."""
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]], 
                 strides: int = 1, padding: str = 'same', 
                 k_int_bits: int = 0, b_int_bits: int = 0, 
                 use_bias: bool = True, act: Optional[XActivation] = None):
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = strides
        self.padding = padding.lower()
        self.k_int_bits = k_int_bits
        self.b_int_bits = b_int_bits
        self.use_bias = use_bias
        self.act = act

        self.weights = None # Shape: (filters, in_channels, kh, kw)
        self.biases = None  # Shape: (filters,)

    def init_weights(self, in_channels: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        kh, kw = self.kernel_size
        # INT8 weights [-128, 127]
        self.weights = np.random.randint(-15, 15, size=(self.filters, in_channels, kh, kw), dtype=np.int8)
        if self.use_bias:
            self.biases = np.random.randint(-10, 10, size=(self.filters,), dtype=np.int16)
        else:
            self.biases = np.zeros((self.filters,), dtype=np.int16)

    def __call__(self, x: np.ndarray, shift_bits: int = 7) -> np.ndarray:
        # Bit-accurate convolution
        is_4d = (x.ndim == 4)
        if not is_4d:
            x = np.expand_dims(x, axis=0)
        
        N, H, W, C = x.shape
        if self.weights is None:
            self.init_weights(C)
            
        kh, kw = self.kernel_size
        out_h = H // self.strides
        out_w = W // self.strides
        
        out = np.zeros((N, out_h, out_w, self.filters), dtype=np.int32)
        pad_h = (kh - 1) // 2 if self.padding == 'same' else 0
        pad_w = (kw - 1) // 2 if self.padding == 'same' else 0

        # Perform MAC accumulation
        for n in range(N):
            for i in range(out_h):
                for j in range(out_w):
                    for f in range(self.filters):
                        val = int(self.biases[f]) if self.use_bias else 0
                        for di in range(kh):
                            for dj in range(kw):
                                ni = i * self.strides + di - pad_h
                                nj = j * self.strides + dj - pad_w
                                if 0 <= ni < H and 0 <= nj < W:
                                    for ci in range(C):
                                        val += int(x[n, ni, nj, ci]) * int(self.weights[f, ci, di, dj])
                        out[n, i, j, f] = val

        # Requantization and scaling
        scaled = (out + (1 << (shift_bits - 1))) >> shift_bits
        clipped = np.clip(scaled, -128, 127).astype(np.int8)
        
        if self.act:
            clipped = self.act(clipped)

        return clipped[0] if not is_4d else clipped

    def __repr__(self) -> str:
        return f"XConvBN(filters={self.filters}, kernel_size={self.kernel_size}, strides={self.strides}, padding='{self.padding}')"


class XDense:
    """Quantized Fully-Connected / GEMV Layer."""
    def __init__(self, units: int, k_int_bits: int = 0, b_int_bits: int = 0,
                 use_bias: bool = True, act: Optional[XActivation] = None):
        self.units = units
        self.k_int_bits = k_int_bits
        self.b_int_bits = b_int_bits
        self.use_bias = use_bias
        self.act = act

        self.weights = None # Shape: (in_features, units)
        self.biases = None  # Shape: (units,)

    def init_weights(self, in_features: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.randint(-15, 15, size=(in_features, self.units), dtype=np.int8)
        if self.use_bias:
            self.biases = np.random.randint(-10, 10, size=(self.units,), dtype=np.int16)
        else:
            self.biases = np.zeros((self.units,), dtype=np.int16)

    def __call__(self, x: np.ndarray, shift_bits: int = 7) -> np.ndarray:
        if x.ndim > 2:
            x = x.reshape(x.shape[0] if x.ndim == 4 else 1, -1)
        elif x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        N, in_features = x.shape
        if self.weights is None:
            self.init_weights(in_features)

        out = np.matmul(x.astype(np.int32), self.weights.astype(np.int32))
        if self.use_bias:
            out += self.biases.astype(np.int32)

        scaled = (out + (1 << (shift_bits - 1))) >> shift_bits
        clipped = np.clip(scaled, -128, 127).astype(np.int8)

        if self.act:
            clipped = self.act(clipped)

        return clipped[0] if clipped.shape[0] == 1 else clipped

    def __repr__(self) -> str:
        return f"XDense(units={self.units}, use_bias={self.use_bias})"


class XBundle:
    """
    Standard Execution Bundle:
      Encapsulates a computational stage containing Core (Conv/Dense), 
      optional Pooling, optional Residual Addition with Activation, 
      Flattening, and Softmax.
    """
    def __init__(self, core: Optional[Union[XConvBN, XDense]] = None, 
                 pool: Optional[XPool] = None, 
                 add_act: Optional[XActivation] = None, 
                 flatten: bool = False, 
                 softmax: bool = False):
        self.core = core
        self.pool = pool
        self.add_act = add_act
        self.flatten = flatten
        self.softmax = softmax

    def __call__(self, x: np.ndarray, skip: Optional[np.ndarray] = None, shift_bits: int = 7) -> np.ndarray:
        out = x
        if self.core is not None:
            out = self.core(out, shift_bits=shift_bits)

        if skip is not None:
            # Fused Residual Saturating Addition: Y = act(clamp(Z + skip, -128, 127))
            res_sum = np.clip(out.astype(np.int32) + skip.astype(np.int32), -128, 127).astype(np.int8)
            if self.add_act is not None:
                out = self.add_act(res_sum)
            else:
                out = res_sum

        if self.pool is not None:
            out = self.pool(out)

        if self.flatten:
            out = out.flatten()

        if self.softmax:
            # Softmax calculation
            e_x = np.exp(out.astype(np.float32) - np.max(out.astype(np.float32)))
            out = e_x / np.sum(e_x)

        return out

    def __repr__(self) -> str:
        components = []
        if self.core: components.append(f"core={self.core}")
        if self.pool: components.append(f"pool={self.pool}")
        if self.add_act: components.append(f"add_act={self.add_act}")
        if self.flatten: components.append("flatten=True")
        if self.softmax: components.append("softmax=True")
        return f"XBundle({', '.join(components)})"


class XModel:
    """
    Standard Base Neural Network Model for Hardware-Software Co-Design.
    Provides standard layer registry, summary, quantization, and verification.
    """
    def __init__(self, sys_bits: SYS_BITS, x_int_bits: int = 0, hw_config: Optional[HW_CONFIG] = None, *args, **kwargs):
        self.sys_bits = sys_bits
        self.x_int_bits = x_int_bits
        self.hw_config = hw_config if hw_config is not None else HW_CONFIG(sys_bits=sys_bits)

    def input_quant_layer(self, x: np.ndarray) -> np.ndarray:
        """Standard input quantizer mapping continuous/float inputs to INT8 [-128, 127]."""
        if x.dtype == np.uint8:
            # Map [0, 255] -> [-128, 127]
            return (x.astype(np.int16) - 128).astype(np.int8)
        elif np.issubdtype(x.dtype, np.floating):
            # Scale float [-1.0, 1.0] -> [-128, 127]
            return np.clip(np.round(x * 127.0), -128, 127).astype(np.int8)
        return x.astype(np.int8)

    def call(self, x: np.ndarray) -> np.ndarray:
        """Executes model computational forward pass."""
        raise NotImplementedError("Subclasses must implement call(x)")

    def summary(self):
        """Prints formatted summary of the model bundles and hardware specs."""
        print("=" * 80)
        print(f" MODEL ARCHITECTURE SUMMARY: {self.__class__.__name__}")
        print("=" * 80)
        print(f" Hardware Configuration: R={self.hw_config.r}, C={self.hw_config.c} ({self.hw_config.total_pes} PEs)")
        print(f" Data Precision: X={self.sys_bits.x}-bit, W={self.sys_bits.k}-bit, B={self.sys_bits.b}-bit, ACC={self.sys_bits.y}-bit")
        print(f" Memory: {self.hw_config.num_banks} Banks x {self.hw_config.word_size} bits ({self.hw_config.ram_depth} Rows)")
        print("-" * 80)
        
        total_params = 0
        bundle_idx = 0
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, XBundle):
                bundle_idx += 1
                core_type = attr_val.core.__class__.__name__ if attr_val.core else "None"
                print(f" [{bundle_idx:02d}] {attr_name:<8} : {core_type:<10} | {attr_val}")
                if attr_val.core and hasattr(attr_val.core, 'weights') and attr_val.core.weights is not None:
                    params = attr_val.core.weights.size + (attr_val.core.biases.size if attr_val.core.use_bias else 0)
                    total_params += params

        print("-" * 80)
        print(f" Total Tracked Parameters: {total_params:,}")
        print("=" * 80)

