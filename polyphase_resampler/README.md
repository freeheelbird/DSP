# Polyphase Resampler Lab

A small tool for exploring polyphase rational resamplers: `x[n] -> up L -> H(z) -> down M -> y[m]`.

- **Prototype filter designer**: equiripple (Parks-McClellan), least-squares and half-band designs.
- **Structure diagram**: which `x[n]` sits in every storage unit of every leg for a given output sample,
  in both the L-leg (output commutator) and M-leg (input commutator) forms.
- **Spectra**: exact steady-state spectra at the input, after upsampling, after filtering and at the output,
  with image and alias content shown separately.
- **Reference implementation**: `polyphase_resample` computes each output the way the L-leg diagram shows it
  and matches `scipy.signal.upfirdn`.

## Use

```
pip install -r requirements.txt
```

Open `polyphase_lab.ipynb` in JupyterLab, VS Code or Colab and run the cells, or in your own notebook:

```python
%matplotlib inline
import polyphase_lab as pl
pl.lab()                                   # interactive controls

h = pl.design_prototype(L=3, M=2, N=48, method="equiripple")
pl.plot_filter(h, 3, 2)
pl.plot_structure(h, 3, 2, m=23, form="L")
pl.plot_spectra(h, 3, 2, signal="band", bw=0.75)
```

Frequencies in the plots are in units of the input sample rate. The prototype is designed at unity DC gain;
multiply by L when interpolating.
