# Filter Designer

A lowpass filter designer for comparing filter types against one specification. You give the sample rate,
passband edge, stopband edge, passband ripple and stopband attenuation; each selected type is designed at the
smallest size that meets it, so tap counts and multiplier counts are comparable.

| File | Purpose |
|---|---|
| `filter_designer.py` | The module: design, quantisation, multiplier counting, signal metrics, plots, export, GUI. |
| `filter_designer.ipynb` | Notebook with the GUI and scripted examples. |

## Use

```
pip install -r requirements.txt
```

Open `filter_designer.ipynb` and run the cells, or in your own notebook:

```python
%matplotlib inline
import filter_designer as fd
fd.designer()
```

Scripted:

```python
spec = fd.Spec(fs=100, fpass=10, fstop=15, ripple_db=0.1, atten_db=60)   # ripple is peak to peak
filters = fd.design_all(spec, ["equiripple", "ls", "kaiser", "elliptic"])
rows = fd.summary(filters, spec, bits=16)                                # list of dicts, one per filter
fd.plot_responses(filters, spec, bits=16)
fd.plot_signal(filters, spec, "rrc_adj", bw=14, alpha=0.35, adj_db=10)
fd.export(filters, spec, "filter_export", bits=16)
```

## What the GUI offers

- **Filter types (checkboxes, any combination).**
  FIR: equiripple (Parks-McClellan), least-squares, Kaiser window, half-band.
  IIR: Butterworth, Chebyshev I, Chebyshev II, elliptic, as second-order sections.
- **Response plots.** Magnitude with the spec's forbidden regions shaded, passband detail, group delay and
  impulse response.
- **Quantisation.** Tick "Quantise coefficients" and set the word length. The quantised response is drawn dashed
  in the same colour and the table switches to the quantised figures.
- **Input spectra.** Root raised cosine and raised cosine (selectable roll-off and bandwidth), ideal rectangular,
  Gaussian, flat, and RRC plus an adjacent channel. The input is plotted before and after each filter.
  Bandwidth is two-sided: the symbol rate for RRC and raised cosine, the -3 dB width for Gaussian.
- **Export.** Writes coefficient files for every ticked filter.

## The summary table

| Column | Meaning |
|---|---|
| Taps / order | Smallest FIR length or IIR order that meets the spec. |
| Multipliers: every non-zero coefficient | One multiplier per non-zero coefficient, no sharing. |
| after symmetry sharing | FIR: symmetric pairs share one multiplier through a pre-adder. IIR: the unit `b0` is free and `b0 = b2` is shared. |
| **non-trivial** | The previous column minus coefficients that are plus or minus a power of two (shifts). The real multiplier count. |
| distinct constants | Distinct values up to sign and power-of-two scaling: a lower bound when identical constants are shared (transposed form). |
| CSD adders | Adders to build the non-trivial set from shifts and adds, with no multipliers. Quantised only. |
| Ripple, Stopband | What the design (or its quantised version) actually achieves. |
| Delay | Group delay in samples; for IIR, the value near DC and how much it varies across the passband. |
| Min coefficient bits | Smallest word length that stays within 1 dB of the stopband spec and 10% of the ripple spec. |
| Signal loss | In-band power lost by the selected input. |
| Adjacent rejection | How much the adjacent channel's power is reduced (RRC + adjacent input). |
| EVM from ISI | For an RRC input followed by a matched RRC receiver: the ISI this filter adds, as a percentage. |

Example, default spec (fs 100, passband 10, stopband 15, 0.1 dB, 60 dB) at 16 bits:

| Filter | Size | Non-trivial multipliers | Min bits | EVM on RRC (alpha 0.35, BW 14) |
|---|---|---|---|---|
| Equiripple | 57 taps | 29 | 14 | 0.37% |
| Least-squares | 73 taps | 36 | 12 | 0.02% |
| Kaiser window | 75 taps | 28 | 13 | 0.02% |
| Elliptic | order 6 | 10 | 14 | 6.1% |

The Kaiser design has the most taps but the fewest FIR multipliers here: its cutoff lands on fs/8, so 18 taps
are exactly zero.

## Exported files

| File | Contents |
|---|---|
| `<type>_float.txt` | FIR taps, one per line. |
| `<type>_q<bits>.mem` | FIR taps as two's complement hex for `$readmemh`. |
| `<type>_q<bits>.coe` | FIR taps as a Xilinx COE file. |
| `<type>_sos.txt` | IIR second-order sections (`b0 b1 b2 a0 a1 a2`), gain folded into the first section. |
| `<type>_sos_q<bits>.txt` | IIR integers `b1 b2 a1 a2` per section with `b0 = a0 = 1`, plus the overall gain. |
| `filters.json` | The spec, every coefficient set, fractional-bit counts and multiplier counts. |

FIR taps are scaled so the largest just fits the word length; the number of fractional bits is in `filters.json`
and on the quantised object (`filt.quantized(bits).frac`).

## Limits

- Lowpass only.
- FIR designs are capped at 1,500 taps; a tighter spec reports that it needs more.
- Half-band needs the passband edge below fs/4 and the stopband edge above it.
- IIR quantisation rounds the section coefficients and the gain and checks stability. It does not model internal
  word growth, overflow or limit cycles.
- In a quantised IIR section, `b1 = 2` (a zero at z = -1) is stored as exactly `2^(bits-1)`: treat it as a
  hard-wired shift, not a stored coefficient word.
