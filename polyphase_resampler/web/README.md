# Web app

`index.html` is the browser version of the lab: one self-contained page of HTML and plain JavaScript.
There is no build step, no server and no Python.

**To use it:** download `index.html` and open it in a browser. (Viewing the file on GitHub shows its source,
not the running page.) It needs no network access apart from two web fonts, and falls back to system fonts
without them.

What it does:

- **Rates.** L and M from 1 to 32.
- **Prototype filter designer.** Equiripple, least-squares and half-band designs, with ripple and rejection
  read-outs and a coefficient list.
- **Structure diagram.** Every storage unit of every leg labelled with the `x[n]` it holds and the `h[i]` that
  weights it. Step or play through output samples. Two views: L legs with an output commutator, M legs with an
  input commutator.
- **Leg responses.** Magnitude and group delay of each leg.
- **Spectra.** Input, after upsampling, after filtering and output, with image and alias content drawn separately.

The filter designs are implemented in JavaScript (a Parks-McClellan exchange and a least-squares solver) and
were checked against SciPy's `remez` and `firls`. The Python lab in the parent folder covers the same ground
and allows larger L, M and tap counts.
