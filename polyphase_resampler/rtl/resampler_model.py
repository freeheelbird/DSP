"""Python model of polyphase_resampler.sv.

Floating point by default; pass fixed=FixedPoint(...) and it reproduces the RTL bit for bit
(quantised coefficients, integer accumulate, round half up, saturate).

    import polyphase_lab as pl
    from resampler_model import PolyphaseResampler, FixedPoint, to_fixed, to_float

    h = pl.design_prototype(3, 2, 48)

    ref = PolyphaseResampler(h, L=3, M=2)                    # floating point
    y   = ref.process(x)                                     # x: float or complex array

    fx  = FixedPoint(data_w=16, coef_w=18, coef_frac=16)     # same numbers as the RTL parameters
    hw  = PolyphaseResampler(h, L=3, M=2, fixed=fx)          # bit-exact with the Verilog
    yq  = hw.process(to_fixed(x, fx.data_w))                 # integers in, integers out
    hw.write_mem("coeffs.mem")                               # ROM file for the RTL

Both modes compute   y[m] = sum_j c[p][j] * x[n - j],   p = (m*M) % L,   n = (m*M) // L,
with c[p][j] = L*h[p + j*L]. The model is streaming: state carries over between process()
calls, so feeding the data in chunks gives the same result as one call. Output m is produced
as soon as input n has arrived, exactly as in the RTL.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FixedPoint:
    """Word lengths, named as the RTL parameters DATA_W, COEF_W, COEF_FRAC."""
    data_w: int = 16        # signed sample width, input and output
    coef_w: int = 18        # signed coefficient width
    coef_frac: int = 16     # fractional bits in the coefficients


def to_fixed(x, data_w):
    """Float in [-1, 1) -> signed integers of data_w bits (round to nearest, saturate)."""
    lim = 1 << (data_w - 1)
    q = lambda v: np.clip(np.round(np.asarray(v, float) * lim), -lim, lim - 1).astype(np.int64)
    return q(np.real(x)) + 1j * q(np.imag(x)) if np.iscomplexobj(x) else q(x)


def to_float(y, data_w):
    """Signed integers of data_w bits -> float in [-1, 1)."""
    return np.asarray(y) / float(1 << (data_w - 1))


class PolyphaseResampler:
    def __init__(self, h, L, M, fixed=None):
        """h: prototype taps at unity DC gain (as from polyphase_lab.design_prototype).
        fixed: None for floating point, or a FixedPoint for the bit-exact RTL model."""
        self.L, self.M, self.fixed = int(L), int(M), fixed
        h = np.asarray(h, float)
        self.K = -(-len(h) // self.L)                              # taps per leg
        hp = np.zeros(self.K * self.L)
        hp[:len(h)] = h
        E = hp.reshape(self.K, self.L).T * self.L                  # E[p, j] = L*h[p + j*L]
        self._set_coeffs(E)
        self.reset()

    @classmethod
    def from_mem(cls, path, L, M, fixed):
        """Build the fixed-point model straight from the RTL's coefficient file."""
        vals = [int(t, 16) for t in open(path).read().split()]
        K, r = divmod(len(vals), L)
        if r:
            raise ValueError(f"{path} has {len(vals)} words, not a multiple of L={L}")
        q = np.array([v - (1 << fixed.coef_w) if v >> (fixed.coef_w - 1) else v for v in vals], np.int64).reshape(L, K)
        self = cls.__new__(cls)
        self.L, self.M, self.fixed, self.K = int(L), int(M), fixed, K
        self.c = q
        self.reset()
        return self

    def _set_coeffs(self, E):
        fx = self.fixed
        if fx is None:
            self.c = E
            return
        q = np.round(E * (1 << fx.coef_frac)).astype(np.int64)
        lim = 1 << (fx.coef_w - 1)
        if q.max() >= lim or q.min() < -lim:
            raise ValueError(f"coefficients need more than {fx.coef_w - fx.coef_frac} integer bits; reduce coef_frac")
        if fx.data_w + fx.coef_w + int(np.ceil(np.log2(self.K))) + 1 > 63:
            raise ValueError("accumulator would exceed 63 bits")
        self.c = q

    # ------------------------------------------------------------------ properties
    @property
    def coeffs(self):
        """Leg coefficients as real numbers, shape (L, K). In fixed mode these are the quantised values."""
        return self.c / float(1 << self.fixed.coef_frac) if self.fixed else self.c

    @property
    def prototype(self):
        """The (quantised) prototype h at unity DC gain, length L*K."""
        return self.coeffs.T.reshape(-1) / self.L

    # ------------------------------------------------------------------ streaming
    def reset(self):
        self._hist = np.zeros(self.K - 1, float if self.fixed is None else np.int64)   # last K-1 inputs
        self._n_in = 0                          # inputs received so far
        self._m = 0                             # next output index

    def process(self, x):
        """Feed input samples, get back every output that can now be computed.

        Float mode: real or complex floats. Fixed mode: integers (real, or complex with integer
        parts, handled as two independent real channels the way two RTL instances would)."""
        x = np.asarray(x)
        if self.fixed is not None:
            if np.iscomplexobj(x):
                parts = [x.real, x.imag]
            else:
                parts = [x]
            lim = 1 << (self.fixed.data_w - 1)
            for p in parts:
                if np.any(p != np.round(p)) or np.any(p < -lim) or np.any(p >= lim):
                    raise ValueError(f"fixed mode needs integers in [{-lim}, {lim - 1}]; use to_fixed()")
            x = x.astype(np.complex128) if np.iscomplexobj(x) else x.astype(np.int64)
        buf = np.concatenate([self._hist.astype(np.result_type(self._hist, x)), x])   # buf[i] = x[n_in - (K-1) + i]
        base = self._n_in - (self.K - 1)
        n_in = self._n_in + len(x)
        m_end = -(-(n_in * self.L) // self.M)                     # outputs m with floor(m*M/L) <= n_in - 1
        m = np.arange(self._m, m_end)
        p, n = (m * self.M) % self.L, (m * self.M) // self.L
        win = buf[(n - base)[:, None] - np.arange(self.K)[None, :]]            # win[i, j] = x[n_i - j]
        c = self.c[p]
        if self.fixed is None:
            y = np.sum(c * win, axis=1)
        elif np.iscomplexobj(win):
            y = self._finish(np.sum(c * win.real.astype(np.int64), axis=1)) + 1j * self._finish(np.sum(c * win.imag.astype(np.int64), axis=1))
        else:
            y = self._finish(np.sum(c * win, axis=1))
        self._hist = buf[len(buf) - (self.K - 1):] if self.K > 1 else buf[:0]
        self._n_in, self._m = n_in, m_end
        return y

    def _finish(self, acc):
        """Round half up, drop the fractional bits, saturate to data_w. Same as RTL stage 3."""
        fx = self.fixed
        half = (1 << (fx.coef_frac - 1)) if fx.coef_frac else 0
        lim = 1 << (fx.data_w - 1)
        return np.clip((acc + half) >> fx.coef_frac, -lim, lim - 1)

    # ------------------------------------------------------------------ files
    def write_mem(self, path):
        """Write the coefficient ROM for the RTL ($readmemh, L*K words, leg-major)."""
        if self.fixed is None:
            raise ValueError("write_mem needs a fixed-point model")
        write_mem(path, self.c.reshape(-1), self.fixed.coef_w)


def write_mem(path, vals, width):
    """Signed integers -> two's complement hex, one word per line."""
    digits = (width + 3) // 4
    with open(path, "w") as f:
        for v in np.asarray(vals).reshape(-1):
            f.write(f"{int(v) & ((1 << width) - 1):0{digits}x}\n")


if __name__ == "__main__":
    import os, sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [_here, os.path.join(_here, "..")]          # polyphase_lab.py beside this file or one level up
    import polyphase_lab as pl
    from scipy import signal

    L, M, fx = 3, 2, FixedPoint(16, 18, 16)
    h = pl.design_prototype(L, M, 48)
    n = np.arange(4000)
    x = 0.5 * np.sin(2 * np.pi * 0.031 * n) + 0.3 * np.sin(2 * np.pi * 0.173 * n + 1)

    ref = PolyphaseResampler(h, L, M)
    y = ref.process(x)
    print("float vs scipy.signal.upfirdn : max diff", np.abs(y - signal.upfirdn(L * h, x, L, M)[:len(y)]).max())

    hw = PolyphaseResampler(h, L, M, fixed=fx)
    yq = hw.process(to_fixed(x, fx.data_w))
    err = to_float(yq, fx.data_w) - y
    print(f"fixed ({fx.data_w}-bit data, {fx.coef_w}-bit coefficients) vs float: "
          f"SNR {10 * np.log10(np.mean(y ** 2) / np.mean(err ** 2)):.1f} dB, max error {np.abs(err).max() * (1 << (fx.data_w - 1)):.2f} LSB")

    hw.reset()
    chunks = np.concatenate([hw.process(c) for c in np.array_split(to_fixed(x, fx.data_w), 37)])
    print("chunked vs one call           : identical" if np.array_equal(chunks, yq) else "chunked vs one call: DIFFERENT")
