#!/usr/bin/env python3
"""RTL vs fixed-point Python verification for polyphase_resampler.sv.

For each configuration this script
  1. designs the prototype and builds the bit-exact model (resampler_model.py, fixed mode),
  2. writes coeffs.mem, stim.mem, expected.mem and params.vh into build/<case>/,
  3. compiles the RTL and testbench with Verilator and runs the simulation once per stall setting
     (random throttling of input valid and output ready),
  4. reads back what the RTL actually produced (rtl_out.txt) and compares it sample for sample
     with the Python model.

Examples
    python verify_rtl.py                                   # one case: L=3, M=2, 48 taps
    python verify_rtl.py --L 5 --M 8 --taps 96 --n 5000
    python verify_rtl.py --data-w 12 --coef-w 16 --frac 14 --stall 0 50 90
    python verify_rtl.py --sweep                           # a spread of rates, designs and word lengths
    python verify_rtl.py --no-sim                          # only write the vectors, for another simulator

Needs numpy, scipy, Verilator 5 (`pip install verilator` works), and polyphase_lab.py and
resampler_model.py beside this file or one level up. Exit status is 0 only if every run matches.
"""
import argparse, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import polyphase_lab as pl
from resampler_model import PolyphaseResampler, FixedPoint, write_mem

RTL = [os.path.join(HERE, "tb_polyphase_resampler.sv"), os.path.join(HERE, "polyphase_resampler.sv")]
SWEEP = [  # L, M, taps, method, data_w, coef_w, frac
    (3, 2, 48, "equiripple", 16, 18, 16), (2, 5, 60, "equiripple", 16, 18, 16), (1, 4, 33, "equiripple", 16, 18, 16),
    (5, 1, 61, "equiripple", 16, 18, 16), (7, 4, 84, "ls", 16, 18, 16), (2, 1, 47, "halfband", 16, 18, 16),
    (5, 8, 96, "equiripple", 12, 16, 14), (4, 9, 37, "ls", 14, 18, 16), (1, 1, 9, "equiripple", 8, 10, 8),
    (5, 4, 80, "equiripple", 16, 25, 23),
]


def stimulus(n, data_w, seed):
    """Two tones plus noise, with a full-scale random burst at the start to exercise saturation."""
    rng = np.random.default_rng(seed)
    lim = (1 << (data_w - 1)) - 1
    t = np.arange(n)
    x = 0.45 * lim * np.sin(2 * np.pi * 0.031 * t) + 0.3 * lim * np.sin(2 * np.pi * 0.173 * t + 1) + 0.2 * lim * rng.uniform(-1, 1, n)
    x[: n // 8] = rng.choice([-lim - 1, lim], n // 8)
    return np.round(x).astype(np.int64)


def find_verilator():
    for name in ("verilator", "verilator-cli"):
        if shutil.which(name):
            return name
    sys.exit("Verilator not found. Install it (for example `pip install verilator`) or use --no-sim.")


def run_case(L, M, taps, method, data_w, coef_w, frac, n, stalls, seed, build_root, simulate, keep):
    name = f"L{L}_M{M}_N{taps}_{method}_d{data_w}_c{coef_w}_f{frac}"
    work = os.path.join(build_root, name)
    os.makedirs(work, exist_ok=True)
    fx = FixedPoint(data_w, coef_w, frac)
    h = pl.design_prototype(L, M, taps, method)
    model = PolyphaseResampler(h, L, M, fixed=fx)
    x = stimulus(n, data_w, seed)
    expected = model.process(x)
    model.write_mem(os.path.join(work, "coeffs.mem"))
    write_mem(os.path.join(work, "stim.mem"), x, data_w)
    write_mem(os.path.join(work, "expected.mem"), expected, data_w)
    with open(os.path.join(work, "params.vh"), "w") as f:
        f.write(f"localparam int L = {L}, M = {M}, K = {model.K}, DATA_W = {data_w}, COEF_W = {coef_w}, COEF_FRAC = {frac};\n")
        f.write(f"localparam int N_IN = {len(x)}, N_OUT = {len(expected)};\n")
    lim = 1 << (data_w - 1)
    info = f"{name:<40} K={model.K:<3} {len(x)} in, {len(expected)} out, {int(((expected == lim - 1) | (expected == -lim)).sum())} saturated"
    if not simulate:
        print(f"{info}\n    vectors written to {work}")
        return True

    cmd = [find_verilator(), "--binary", "--timing", "-j", "0", "-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSEDSIGNAL",
           "-Wno-UNUSEDPARAM", "-Wno-PROCASSINIT", f"-I{work}", "--Mdir", os.path.join(work, "obj_dir"),
           "--top", "tb_polyphase_resampler", "-o", "sim"] + RTL
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"{info}\n    BUILD FAILED\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
        return False
    print(f"{info}   (built in {time.time() - t0:.0f} s)")

    ok = True
    for stall in stalls:
        out_file = os.path.join(work, "rtl_out.txt")
        if os.path.exists(out_file):
            os.remove(out_file)
        r = subprocess.run([os.path.join(work, "obj_dir", "sim"), f"+stall={stall}", f"+verilator+seed+{seed + stall + 1}"],
                           cwd=work, capture_output=True, text=True)
        tb_line = next((l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL"))), "no PASS/FAIL line from testbench")
        got = np.loadtxt(out_file, dtype=np.int64, ndmin=1) if os.path.exists(out_file) and os.path.getsize(out_file) else np.zeros(0, np.int64)
        k = min(len(got), len(expected))
        bad = np.flatnonzero(got[:k] != expected[:k])
        if len(got) == len(expected) and not len(bad) and tb_line.startswith("PASS"):
            print(f"    stall {stall:>2}%: MATCH   {len(got)} outputs identical")
        else:
            ok = False
            print(f"    stall {stall:>2}%: MISMATCH   RTL gave {len(got)} outputs, model {len(expected)}; {len(bad)} differ; testbench: {tb_line}")
            for i in bad[:5]:
                print(f"        y[{i}]: rtl {got[i]}, model {expected[i]}  (leg {(i * M) % L}, newest input x[{(i * M) // L}])")
    if ok and not keep:
        shutil.rmtree(os.path.join(work, "obj_dir"), ignore_errors=True)
    return ok


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--L", type=int, default=3); a.add_argument("--M", type=int, default=2)
    a.add_argument("--taps", type=int, default=48)
    a.add_argument("--method", default="equiripple", choices=pl.METHODS)
    a.add_argument("--data-w", type=int, default=16); a.add_argument("--coef-w", type=int, default=18)
    a.add_argument("--frac", type=int, default=16, help="fractional bits in the coefficients (COEF_FRAC)")
    a.add_argument("--n", type=int, default=2000, help="number of input samples")
    a.add_argument("--stall", type=int, nargs="+", default=[0, 30, 70], help="stall probabilities in percent, one simulation each")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--sweep", action="store_true", help="run the built-in list of configurations instead of one case")
    a.add_argument("--no-sim", action="store_true", help="write vectors only; do not build or run Verilator")
    a.add_argument("--keep", action="store_true", help="keep the Verilator build directories of passing cases")
    a.add_argument("--build", default=os.path.join(HERE, "build"), help="output directory")
    o = a.parse_args()

    cases = SWEEP if o.sweep else [(o.L, o.M, o.taps, o.method, o.data_w, o.coef_w, o.frac)]
    results = [run_case(*c, o.n, o.stall, o.seed, o.build, not o.no_sim, o.keep) for c in cases]
    if not o.no_sim:
        print(f"\n{sum(results)} of {len(results)} configurations match the fixed-point model" + ("" if all(results) else "  <-- FAILURES"))
    sys.exit(0 if all(results) else 1)
