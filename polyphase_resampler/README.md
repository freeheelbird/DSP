# Polyphase Resampler

A rational resampler, `x[n] -> up L -> H(z) -> down M -> y[m]`, taken from exploration to hardware:

| Part | Files | Purpose |
|---|---|---|
| Lab | `polyphase_lab.py`, `polyphase_lab.ipynb` | Design the prototype filter, see what sits in every storage unit of every leg, and view the spectra at each stage. |
| Web app | [`web/index.html`](web) | The same lab as a single page that runs in a browser. |
| RTL | [`rtl/polyphase_resampler.sv`](rtl) | Synthesizable SystemVerilog implementation of the L-leg structure. |
| Model | [`rtl/resampler_model.py`](rtl) | Floating-point reference with an optional fixed-point mode that matches the RTL bit for bit. |
| Verification | [`rtl/verify_rtl.py`](rtl) | Simulates the RTL in Verilator and compares its output with the fixed-point model. |

The RTL, model and verification flow are documented in [`rtl/README.md`](rtl/README.md). The web app is described in [`web/README.md`](web/README.md).

## Lab

```
pip install -r requirements.txt
```

Open `polyphase_lab.ipynb` in JupyterLab, VS Code or Colab and run the cells, or in your own notebook:

```python
%matplotlib inline
import polyphase_lab as pl
pl.lab()                                   # interactive controls (ipywidgets)
```

The same pieces are available as plain functions:

```python
h = pl.design_prototype(L=3, M=2, N=48, method="equiripple")   # or "ls", "halfband"
pl.filter_stats(h, 3, 2)                   # ripple, rejection, band edges
pl.plot_filter(h, 3, 2)                    # magnitude and impulse response
pl.plot_structure(h, 3, 2, m=23, form="L") # which x[n] is in each storage unit for output y[23]
pl.plot_legs(h, 3, 2)                      # magnitude and group delay of every leg
pl.plot_spectra(h, 3, 2, signal="band", bw=0.75)
y = pl.polyphase_resample(x, h, 3, 2)      # sample-by-sample, as the diagram computes it
```

What it covers:

- **Prototype filter designer.** Equiripple (Parks-McClellan), least-squares and half-band designs, with passband
  and stopband edges given as multiples of the ideal cutoff `fc = 0.5 / max(L, M)` at the prototype rate `L * f_in`.
- **Structure.** Two views of the same arithmetic: L legs with an output commutator (one shared delay line, the leg
  index picks the coefficient set) and M legs with an input commutator (each leg has its own delay line).
  `leg_contents()` returns the raw table of sample and tap indices.
- **Spectra.** Exact steady-state spectra of a periodic test signal at the input, after upsampling, after filtering
  and at the output, with image and alias content drawn separately.

Conventions: frequencies in plots are in units of the input sample rate. The prototype is designed at unity DC
gain; the interpolation gain of L is applied where the filter is used.
