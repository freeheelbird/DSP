#!/usr/bin/env python3
"""Generate the coefficient ROM (and optional test vectors) for polyphase_resampler.sv.

    python gen_coeffs.py --L 3 --M 2 --taps 48 --method equiripple --test 400

Writes coeffs.mem (L*K lines, leg-major: line p*K + j = round(L*h[p + j*L] * 2^frac)).
With --test N it also writes stim.mem and expected.mem (bit-exact model of the RTL)
and params.vh for the testbench. Expected outputs come from resampler_model.py in fixed-point mode.
Needs polyphase_lab.py and resampler_model.py beside this file or one level up.
"""
import argparse, os, sys
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..")]      # polyphase_lab.py beside this file or one level up
import polyphase_lab as pl
from resampler_model import PolyphaseResampler, FixedPoint, write_mem


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--L", type=int, default=3); a.add_argument("--M", type=int, default=2)
    a.add_argument("--taps", type=int, default=48)
    a.add_argument("--method", default="equiripple", choices=pl.METHODS)
    a.add_argument("--pass-edge", type=float, default=0.8); a.add_argument("--stop-edge", type=float, default=1.2)
    a.add_argument("--weight", type=float, default=1.0)
    a.add_argument("--data-w", type=int, default=16); a.add_argument("--coef-w", type=int, default=18)
    a.add_argument("--frac", type=int, default=16)
    a.add_argument("--test", type=int, default=0, metavar="N_IN", help="also write test vectors for N_IN input samples")
    a.add_argument("--outdir", default=".")
    o = a.parse_args()

    h = pl.design_prototype(o.L, o.M, o.taps, o.method, o.pass_edge, o.stop_edge, o.weight)
    fx = FixedPoint(o.data_w, o.coef_w, o.frac)
    try:
        hw = PolyphaseResampler(h, o.L, o.M, fixed=fx)
    except ValueError as e:
        raise SystemExit(str(e))
    K = hw.K
    hw.write_mem(os.path.join(o.outdir, "coeffs.mem"))
    hq = hw.prototype
    st = pl.filter_stats(hq[:len(h)], o.L, o.M, o.pass_edge, o.stop_edge, o.method == "halfband")
    print(f"L={o.L} M={o.M}  {len(h)} taps -> K={K} per leg, {o.L * K} ROM words")
    print(f"quantised to {o.coef_w} bits: ripple {st['ripple_db']:.4f} dB p-p, rejection {st['rejection_db']:.1f} dB")
    print(f"instantiate with  .L({o.L}), .M({o.M}), .K({K}), .DATA_W({o.data_w}), .COEF_W({o.coef_w}), .COEF_FRAC({o.frac})")

    if o.test:
        rng = np.random.default_rng(1)
        lim = (1 << (o.data_w - 1)) - 1
        n = np.arange(o.test)
        x = 0.45 * lim * np.sin(2 * np.pi * 0.031 * n) + 0.3 * lim * np.sin(2 * np.pi * 0.173 * n + 1) + 0.2 * lim * rng.uniform(-1, 1, o.test)
        x[: o.test // 8] = rng.choice([-lim - 1, lim], o.test // 8)          # full-scale burst to exercise saturation
        x = np.round(x).astype(np.int64)
        y = hw.process(x)
        write_mem(os.path.join(o.outdir, "stim.mem"), x, o.data_w)
        write_mem(os.path.join(o.outdir, "expected.mem"), y, o.data_w)
        with open(os.path.join(o.outdir, "params.vh"), "w") as f:
            f.write(f"localparam int L = {o.L}, M = {o.M}, K = {K}, DATA_W = {o.data_w}, COEF_W = {o.coef_w}, COEF_FRAC = {o.frac};\n")
            f.write(f"localparam int N_IN = {len(x)}, N_OUT = {len(y)};\n")
        print(f"test vectors: {len(x)} inputs, {len(y)} expected outputs, {(np.abs(y) >= lim).sum()} saturated")
