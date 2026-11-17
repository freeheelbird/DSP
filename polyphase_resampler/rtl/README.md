# RTL, fixed-point model and verification

| File | Purpose |
|---|---|
| `polyphase_resampler.sv` | The resampler. |
| `tb_polyphase_resampler.sv` | Self-checking testbench with randomly throttled valid and ready. |
| `resampler_model.py` | Python model: floating point, or fixed point matching the RTL bit for bit. |
| `gen_coeffs.py` | Designs a prototype, quantises it and writes the coefficient ROM file. |
| `verify_rtl.py` | Runs RTL-versus-model verification with Verilator. |

The Python scripts import `polyphase_lab.py` from the parent folder.

## The module

```
y[m] = sum_{j=0}^{K-1} c[p][j] * x[n - j]
p = (m*M) mod L          leg (coefficient set) used for output m
n = floor(m*M / L)       newest input sample used for output m
c[p][j] = L * h[p + j*L]
```

This is the "L legs, output commutator" structure: one shared K-sample delay line, and the leg index selects a row
of the coefficient ROM. Zero-stuffed samples are never stored or multiplied.

| Parameter | Default | Meaning |
|---|---|---|
| `L` | 3 | Interpolation factor (number of legs) |
| `M` | 2 | Decimation factor |
| `K` | 16 | Taps per leg, `ceil(N_taps / L)` |
| `DATA_W` | 16 | Signed sample width, input and output |
| `COEF_W` | 18 | Signed coefficient width |
| `COEF_FRAC` | 16 | Fractional bits in the coefficients |
| `COEF_FILE` | `"coeffs.mem"` | `$readmemh` file, `L*K` words, word `p*K + j` = `round(L*h[p + j*L] * 2^COEF_FRAC)` |

| Port | Direction | Meaning |
|---|---|---|
| `clk`, `rst` | in | Clock; synchronous active-high reset |
| `s_data`, `s_valid`, `s_ready` | in, in, out | Input samples at `f_in` |
| `m_data`, `m_valid`, `m_ready` | out, out, in | Output samples at `f_in * L / M` |

Behaviour:

- **Control.** The phase advances by M per output; each wrap past L shifts one more input into the delay line.
  The last input an output needs can arrive in the same clock that launches it, so the clock only has to be as
  fast as the higher of the input and output sample rates.
- **Datapath.** K parallel multipliers, a sum, then round half up and saturate to `DATA_W`. Latency is 4 clocks.
- **Handshake.** Valid/ready on both sides. The whole pipeline stalls while `m_valid` is high and `m_ready` is low.
- **Gain.** The interpolation gain of L is folded into the coefficients. With `COEF_W - COEF_FRAC = 2` they span
  the range -2 to +2.

Limits:

- Real samples only. For I/Q use two instances sharing one coefficient file.
- At most one output per clock, and the K-term sum is a single combinational stage. High sample rates need several
  outputs computed in parallel per clock and a pipelined adder tree.
- Coefficient symmetry and zero taps (for example in half-band designs) are not exploited.
- Verified in simulation only. It has not been through synthesis or timing closure.

## Generating coefficients

```
python gen_coeffs.py --L 3 --M 2 --taps 48 --method equiripple
```

Writes `coeffs.mem`, reports ripple and rejection after quantisation, and prints the parameters to instantiate
with. Options: `--method {equiripple,ls,halfband}`, `--pass-edge`, `--stop-edge`, `--weight`, `--data-w`,
`--coef-w`, `--frac`. `--test N` also writes `stim.mem`, `expected.mem` and `params.vh` for the testbench.

## Python model

```python
import polyphase_lab as pl
from resampler_model import PolyphaseResampler, FixedPoint, to_fixed, to_float

h = pl.design_prototype(3, 2, 48)

ref = PolyphaseResampler(h, L=3, M=2)                 # floating point
y   = ref.process(x)                                  # real or complex floats

fx  = FixedPoint(data_w=16, coef_w=18, coef_frac=16)  # same numbers as the RTL parameters
hw  = PolyphaseResampler(h, L=3, M=2, fixed=fx)       # bit-exact with the RTL
yq  = hw.process(to_fixed(x, fx.data_w))              # integers in, integers out
```

- Float mode matches `scipy.signal.upfirdn(L*h, x, L, M)`.
- Fixed mode quantises the coefficients, accumulates in integers, rounds half up and saturates, as the RTL does.
  Complex integer input is processed as two independent real channels.
- The model is streaming: state carries across `process()` calls, so chunked input gives the same output as one call.
- `write_mem()` writes the ROM file; `PolyphaseResampler.from_mem()` builds the model from an existing one.
- `.coeffs` and `.prototype` return the quantised taps as real numbers, for example to plot with `pl.plot_filter`.

Running `python resampler_model.py` prints a self-check: float against SciPy, fixed against float, chunked against
one call.

## Verifying the RTL against the model

Needs Verilator 5 (`pip install verilator` is enough).

```
python verify_rtl.py                                 # one case: L=3, M=2, 48 taps
python verify_rtl.py --L 5 --M 8 --taps 96 --n 5000
python verify_rtl.py --data-w 12 --coef-w 16 --frac 14 --stall 0 50 90
python verify_rtl.py --sweep                         # ten rates, designs and word lengths
python verify_rtl.py --no-sim                        # write vectors only
```

For each configuration the script builds the fixed-point model, writes the vectors into `build/<case>/`, compiles
and runs the testbench once per stall setting, then reads back the RTL's outputs (`rtl_out.txt`) and compares them
sample for sample with the model. The stimulus is two tones plus noise with a full-scale burst, so saturation is
exercised. Exit status is 0 only if every run matches.

```
L3_M2_N48_equiripple_d16_c18_f16         K=16  2000 in, 3000 out, 151 saturated   (built in 5 s)
    stall  0%: MATCH   3000 outputs identical
    stall 30%: MATCH   3000 outputs identical
    stall 70%: MATCH   3000 outputs identical
```

Configurations in `--sweep`, all matching as of the last run:

| L | M | Taps | Design | Data | Coefficient | Fraction |
|---|---|---|---|---|---|---|
| 3 | 2 | 48 | equiripple | 16 | 18 | 16 |
| 2 | 5 | 60 | equiripple | 16 | 18 | 16 |
| 1 | 4 | 33 | equiripple | 16 | 18 | 16 |
| 5 | 1 | 61 | equiripple | 16 | 18 | 16 |
| 7 | 4 | 84 | least-squares | 16 | 18 | 16 |
| 2 | 1 | 47 | half-band | 16 | 18 | 16 |
| 5 | 8 | 96 | equiripple | 12 | 16 | 14 |
| 4 | 9 | 37 | least-squares | 14 | 18 | 16 |
| 1 | 1 | 9 | equiripple | 8 | 10 | 8 |
| 5 | 4 | 80 | equiripple | 16 | 25 | 23 |

**Other simulators.** Run `verify_rtl.py --no-sim`, then compile the two `.sv` files with `build/<case>/` as the
include path and working directory. The testbench prints `PASS` or `FAIL` itself; `+stall=<percent>` sets the
throttling. Only Verilator has been tried.
