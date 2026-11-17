"""Lowpass filter designer and comparison tool.

Give one specification (sample rate, passband edge, stopband edge, passband ripple, stopband
attenuation). Each selected filter type is designed at the smallest size that meets it, so the
types can be compared like for like.

In a notebook:

    %matplotlib inline
    import filter_designer as fd
    fd.designer()                      # GUI (needs ipywidgets)

Scripted:

    spec = fd.Spec(fs=100, fpass=10, fstop=15, ripple_db=0.1, atten_db=60)
    filters = fd.design_all(spec, ["equiripple", "kaiser", "elliptic"])
    fd.summary(filters, spec, bits=16)             # list of dict rows
    fd.plot_responses(filters, spec, bits=16)
    fd.export(filters, spec, "filter_export", bits=16)

Ripple is peak to peak in dB. FIR types: equiripple (Parks-McClellan), least-squares, Kaiser
window, half-band. IIR types: Butterworth, Chebyshev I, Chebyshev II, elliptic (as second-order
sections). Requires numpy, scipy, matplotlib; ipywidgets for designer().
"""
from dataclasses import dataclass, field
import json, os
import numpy as np
from scipy import signal as sps
import matplotlib.pyplot as plt

MAX_TAPS = 1500

# key: (label, kind, colour). Colour follows the filter type, whatever else is selected.
TYPES = {
    "equiripple": ("Equiripple (Parks-McClellan)", "fir", "#2a78d6"),
    "ls":         ("Least-squares",                "fir", "#eb6834"),
    "kaiser":     ("Kaiser window",                "fir", "#1baf7a"),
    "halfband":   ("Half-band",                    "fir", "#eda100"),
    "butter":     ("Butterworth (IIR)",            "iir", "#e87ba4"),
    "cheby1":     ("Chebyshev I (IIR)",            "iir", "#008300"),
    "cheby2":     ("Chebyshev II (IIR)",           "iir", "#6250d6"),
    "elliptic":   ("Elliptic (IIR)",               "iir", "#e34948"),
}
INPUTS = {
    "none":    "None",
    "rrc":     "Root raised cosine (RRC)",
    "rc":      "Raised cosine",
    "rect":    "Ideal rectangular",
    "gauss":   "Gaussian",
    "white":   "Flat (white) to Nyquist",
    "rrc_adj": "RRC + adjacent channel",
}


@dataclass(frozen=True)
class Spec:
    fs: float = 100.0          # sample rate (any unit; edges use the same unit)
    fpass: float = 10.0        # passband edge
    fstop: float = 15.0        # stopband edge
    ripple_db: float = 0.1     # passband ripple, peak to peak
    atten_db: float = 60.0     # stopband attenuation

    def check(self):
        if not (0 < self.fpass < self.fstop < self.fs / 2):
            raise ValueError("need 0 < passband edge < stopband edge < fs/2")
        if self.ripple_db <= 0 or self.atten_db <= 0:
            raise ValueError("ripple and attenuation must be positive (dB)")

    @property
    def dp(self):              # passband deviation +/- for a peak-to-peak ripple
        r = 10 ** (self.ripple_db / 20)
        return (r - 1) / (r + 1)

    @property
    def ds(self):
        return 10 ** (-self.atten_db / 20)


@dataclass
class Filter:
    key: str
    h: np.ndarray = None           # FIR taps
    sos: np.ndarray = None         # IIR sections, normalised: rows [1, b1, b2, 1, a1, a2]
    gain: float = 1.0              # IIR overall gain
    order: int = 0                 # IIR order
    note: str = ""                 # why it could not be designed, or a warning
    q: np.ndarray = None           # quantised integers (FIR taps, or IIR [b1, b2, a1, a2] per section)
    frac: int = None               # fractional bits of q
    qgain: tuple = None            # IIR gain as (integer, fractional bits)
    bits: int = None

    label = property(lambda s: TYPES[s.key][0])
    kind = property(lambda s: TYPES[s.key][1])
    color = property(lambda s: TYPES[s.key][2])
    ok = property(lambda s: s.h is not None or s.sos is not None)

    @property
    def size(self):
        return len(self.h) if self.kind == "fir" else self.order

    @property
    def size_text(self):
        if not self.ok:
            return "n/a"
        return f"{len(self.h)} taps" if self.kind == "fir" else f"order {self.order}"

    def response(self, f, fs):
        """Complex frequency response at frequencies f (same unit as fs)."""
        if self.kind == "fir":
            return sps.freqz(self.h, worN=f, fs=fs)[1]
        full = np.column_stack([np.ones(len(self.sos)), self.sos[:, 1:]])
        return self.gain * sps.sosfreqz(full, worN=f, fs=fs)[1]

    def impulse(self, n=None):
        if self.kind == "fir":
            return self.h
        x = np.zeros(n or 200)
        x[0] = 1
        return self.gain * sps.sosfilt(np.column_stack([np.ones(len(self.sos)), self.sos[:, 1:]]), x)

    def stable(self):
        return self.kind == "fir" or all(np.all(np.abs(np.roots(s[3:])) < 1) for s in self.sos)

    def quantized(self, bits):
        """Copy of this filter with coefficients rounded to signed `bits`-bit integers."""
        if not self.ok:
            return self
        if self.kind == "fir":
            e = int(np.floor(np.log2(np.abs(self.h).max())))
            frac = bits - 2 - e                                    # largest tap just fits
            q = np.round(self.h * 2.0 ** frac).astype(np.int64)
            return Filter(self.key, h=q / 2.0 ** frac, q=q, frac=frac, bits=bits)
        frac = bits - 2                                            # section coefficients lie in [-2, 2]
        lim = 2 ** (bits - 1)
        # +2.0 (a zero at z = -1) rounds to exactly 2^(bits-1): kept, since it is a hard-wired shift, not a stored word
        q = np.clip(np.round(self.sos[:, [1, 2, 4, 5]] * 2.0 ** frac), -lim, lim).astype(np.int64)
        s = self.sos.copy()
        s[:, [1, 2, 4, 5]] = q / 2.0 ** frac
        ge = int(np.floor(np.log2(self.gain)))
        gf = bits - 2 - ge
        qg = int(np.round(self.gain * 2.0 ** gf))
        out = Filter(self.key, sos=s, gain=qg / 2.0 ** gf, order=self.order, q=q, frac=frac, qgain=(qg, gf), bits=bits)
        if not out.stable():
            out.note = f"unstable at {bits} bits"
        return out


# --------------------------------------------------------------------------- design
def _grid(spec, n_taps=0):
    return np.linspace(0, spec.fs / 2, int(min(32768, max(4096, 8 * n_taps))) + 1)


def measure(filt, spec, f=None):
    """Passband ripple (dB peak to peak) and stopband attenuation (dB) actually achieved."""
    f = _grid(spec, filt.size if filt.kind == "fir" else 0) if f is None else f
    d = 20 * np.log10(np.abs(filt.response(f, spec.fs)) + 1e-300)
    return float(np.ptp(d[f <= spec.fpass])), float(-d[f >= spec.fstop].max())


def _meets(filt, spec, tol_db=0.02):
    r, a = measure(filt, spec)
    return r <= spec.ripple_db * 1.01 + 1e-9 and a >= spec.atten_db - tol_db


def _ls_fit(nu, bands):
    nu = np.asarray(nu, float)
    d, s = nu[:, None] - nu[None, :], nu[:, None] + nu[None, :]

    def I(al, w1, w2):
        safe = np.where(np.abs(al) < 1e-12, 1.0, al)
        return np.where(np.abs(al) < 1e-12, w2 - w1, (np.sin(safe * w2) - np.sin(safe * w1)) / safe)

    Q, b = np.zeros((len(nu), len(nu))), np.zeros(len(nu))
    for w1, w2, D, W in bands:
        Q += 0.5 * W * (I(d, w1, w2) + I(s, w1, w2))
        b += W * D * I(nu, w1, w2)
    Q += np.eye(len(nu)) * 1e-14 * np.trace(Q) / len(nu)
    return np.linalg.solve(Q, b)


def _fir_ls(N, spec):
    w = 2 * np.pi / spec.fs
    bands = [(0, w * spec.fpass, 1.0, 1.0), (w * spec.fstop, np.pi, 0.0, (spec.dp / spec.ds) ** 2)]
    h = np.zeros(N)
    if N % 2:
        n = (N - 1) // 2
        a = _ls_fit(np.arange(n + 1), bands)
        h[n] = a[0]
        k = np.arange(1, n + 1)
        h[n - k] = h[n + k] = a[1:] / 2
    else:
        n = N // 2
        a = _ls_fit(np.arange(n) + 0.5, bands)
        k = np.arange(n)
        h[n - 1 - k] = h[n + k] = a / 2
    return h


def _fir_remez(N, spec):
    return sps.remez(N, [0, spec.fpass, spec.fstop, spec.fs / 2], [1, 0], weight=[1, spec.dp / spec.ds], fs=spec.fs, maxiter=100)


def _fir_kaiser(N, spec, beta):
    return sps.firwin(N, (spec.fpass + spec.fstop) / 2, window=("kaiser", beta), fs=spec.fs)


def _estimate_taps(spec):
    df = (spec.fstop - spec.fpass) / spec.fs
    return int(max(3, np.ceil((-20 * np.log10(np.sqrt(spec.dp * spec.ds)) - 13) / (14.6 * df) + 1)))


def _min_taps(make, spec, start, key):
    """Smallest N for which make(N) meets the spec, searching around `start`."""
    def ok(N):
        try:
            h = make(N)
        except Exception:
            return None
        return h if np.all(np.isfinite(h)) and _meets(Filter(key, h=h), spec) else None

    N, h = int(max(3, start)), None
    step = 1
    while N <= MAX_TAPS:                                   # go up until it meets
        h = ok(N)
        if h is not None:
            break
        step = max(1, N // 40)
        N += step
    if h is None:
        return Filter(key, note=f"needs more than {MAX_TAPS} taps")
    misses = 0
    M = N - 1
    while M >= 3 and misses < max(3, 2 * step):            # then come back down
        g = ok(M)
        if g is not None:
            N, h, misses = M, g, 0
        else:
            misses += 1
        M -= 1
    return Filter(key, h=h)


def design(spec, key):
    """Design one filter type at the smallest size that meets the spec."""
    spec.check()
    kind = TYPES[key][1]
    est = _estimate_taps(spec)
    if est > MAX_TAPS and kind == "fir" and key != "halfband":
        return Filter(key, note=f"needs more than {MAX_TAPS} taps")
    if key == "equiripple":
        return _min_taps(lambda N: _fir_remez(N, spec), spec, est - 2, key)
    if key == "ls":
        return _min_taps(lambda N: _fir_ls(N, spec), spec, est, key)
    if key == "kaiser":
        A = max(spec.atten_db, -20 * np.log10(spec.dp))
        N0, beta = sps.kaiserord(A, (spec.fstop - spec.fpass) / (spec.fs / 2))
        return _min_taps(lambda N: _fir_kaiser(N, spec, beta), spec, N0 - 3, key)
    if key == "halfband":
        if not (spec.fpass < spec.fs / 4 < spec.fstop):
            return Filter(key, note="needs passband edge < fs/4 < stopband edge")
        fp = max(spec.fpass, spec.fs / 2 - spec.fstop) / spec.fs       # symmetric transition about fs/4
        d = min(spec.dp, spec.ds)
        est_hb = (-20 * np.log10(d) - 13) / (14.6 * (0.5 - 2 * fp)) + 1
        q = max(0, int(est_hb // 4) - 2)
        while 4 * q + 3 <= MAX_TAPS:
            h = np.zeros(4 * q + 3)
            h[2 * q + 1] = 0.5
            try:
                h[0::2] = sps.remez(2 * q + 2, [0, 2 * fp], [1], fs=1, maxiter=100) / 2
                if _meets(Filter(key, h=h), spec):
                    return Filter(key, h=h)
            except Exception:
                pass
            q += 1
        return Filter(key, note=f"needs more than {MAX_TAPS} taps")
    rp, rs, a = spec.ripple_db, spec.atten_db, dict(fs=spec.fs)
    try:
        if key == "butter":
            N, wn = sps.buttord(spec.fpass, spec.fstop, rp, rs, **a)
            sos = sps.butter(N, wn, output="sos", **a)
        elif key == "cheby1":
            N, wn = sps.cheb1ord(spec.fpass, spec.fstop, rp, rs, **a)
            sos = sps.cheby1(N, rp, wn, output="sos", **a)
        elif key == "cheby2":
            N, wn = sps.cheb2ord(spec.fpass, spec.fstop, rp, rs, **a)
            sos = sps.cheby2(N, rs, wn, output="sos", **a)
        else:
            N, wn = sps.ellipord(spec.fpass, spec.fstop, rp, rs, **a)
            sos = sps.ellip(N, rp, rs, wn, output="sos", **a)
    except Exception as e:
        return Filter(key, note=f"design failed: {e}")
    if N > 60 or not np.all(np.isfinite(sos)):
        return Filter(key, note=f"order {N} is impractical")
    g = sos[:, 0].copy()
    sosn = sos.copy()
    sosn[:, :3] /= g[:, None]
    return Filter(key, sos=sosn, gain=float(np.prod(g)), order=int(N))


_cache = {}


def design_all(spec, keys):
    """Design several types. Results are cached per (spec, type)."""
    out = []
    for k in keys:
        if (spec, k) not in _cache:
            _cache[(spec, k)] = design(spec, k)
        out.append(_cache[(spec, k)])
    return out


# --------------------------------------------------------------------------- multipliers
def _is_pow2(v):
    v = abs(v)
    if v == 0:
        return False
    if isinstance(v, (int, np.integer)):
        return (int(v) & (int(v) - 1)) == 0
    return abs(np.frexp(v)[0] - 0.5) < 1e-12


def _odd_part(v):
    v = abs(v)
    if isinstance(v, (int, np.integer)):
        v = int(v)
        while v and v % 2 == 0:
            v //= 2
        return v
    return float(np.round(np.frexp(v)[0], 12))            # mantissa: equal up to a power of two


def _csd_nonzeros(v):
    v, n = abs(int(v)), 0
    while v:
        if v & 1:
            n += 1
            v += 1 if (v & 3) == 3 else -1                 # canonical signed digit recoding
        v >>= 1
    return n


def multipliers(filt):
    """Multiplier counts for a filter (uses the quantised integers when present).

    direct      one multiplier per non-zero coefficient
    symmetric   after sharing: FIR pairs h[i] = h[N-1-i] use one multiplier (pre-adder);
                IIR sections skip the unit b0 and share b0 = b2
    nontrivial  `symmetric` minus coefficients that are +/- a power of two (a shift, no multiplier)
    distinct    distinct constants up to sign and power-of-two scaling: the number of constant
                multipliers if every product of the same constant is shared (transposed form)
    csd_adders  adders for a shift-and-add build of the `nontrivial` set (CSD digits - 1 each);
                quantised filters only
    """
    if not filt.ok:
        return dict(direct=None, symmetric=None, nontrivial=None, distinct=None, csd_adders=None)
    quant = filt.q is not None
    tiny = lambda v: (v == 0) if quant else (abs(v) < 1e-12)
    if filt.kind == "fir":
        c = list(filt.q) if quant else list(filt.h)
        direct = sum(not tiny(v) for v in c)
        sym = all((a == b) if quant else abs(a - b) < 1e-12 for a, b in zip(c, c[::-1]))
        uniq = c[: (len(c) + 1) // 2] if sym else c
    else:
        rows = filt.q if quant else filt.sos[:, [1, 2, 4, 5]]
        one = (1 << filt.frac) if quant else 1.0
        g = filt.qgain[0] if quant else filt.gain
        direct, uniq = 1, [g]                              # the gain
        for b1, b2, a1, a2 in rows:
            direct += 1 + sum(not tiny(v) for v in (b1, b2, a1, a2))        # b0 is 1 before the gain is folded in
            uniq += [b1, a1, a2] + ([] if abs(b2 - one) < 1e-9 else [b2])
    uniq = [v for v in uniq if not tiny(v)]
    nontriv = [v for v in uniq if not _is_pow2(v)]
    return dict(direct=direct, symmetric=len(uniq), nontrivial=len(nontriv),
                distinct=len({_odd_part(v) for v in nontriv}),
                csd_adders=sum(_csd_nonzeros(v) - 1 for v in nontriv) if quant else None)


def min_bits(filt, spec, lo=4, hi=32):
    """Smallest coefficient width that stays within 1 dB of the stopband spec and 10% of the ripple spec."""
    if not filt.ok:
        return None
    for b in range(lo, hi + 1):
        q = filt.quantized(b)
        if not q.stable():
            continue
        r, a = measure(q, spec)
        if a >= spec.atten_db - 1.0 and r <= spec.ripple_db * 1.1:
            return b
    return None


# --------------------------------------------------------------------------- inputs
def input_spectrum(kind, f, bw, alpha=0.35, adj_db=10.0):
    """Amplitude spectrum of a test input at frequencies f >= 0. bw is the two-sided bandwidth
    (the symbol rate for RRC / raised cosine, the -3 dB width for Gaussian)."""
    f = np.abs(np.asarray(f, float))

    def rc(x):
        lo, hi = (1 - alpha) * bw / 2, (1 + alpha) * bw / 2
        mid = 0.5 * (1 + np.cos(np.pi / max(alpha * bw, 1e-12) * (np.clip(x, lo, hi) - lo)))
        return np.where(x <= lo, 1.0, np.where(x >= hi, 0.0, mid))

    if kind == "rrc":
        return np.sqrt(rc(f))
    if kind == "rc":
        return rc(f)
    if kind == "rect":
        return (f <= bw / 2).astype(float)
    if kind == "gauss":
        return 2.0 ** (-0.5 * (2 * f / bw) ** 2)
    if kind == "white":
        return np.ones_like(f)
    if kind == "rrc_adj":
        return np.sqrt(rc(f)) + 10 ** (adj_db / 20) * np.sqrt(rc(np.abs(f - adjacent_offset(bw, alpha))))
    raise ValueError(kind)


def adjacent_offset(bw, alpha):
    """Centre of the adjacent channel: channels spaced 10% wider than their occupied bandwidth."""
    return 1.1 * (1 + alpha) * bw


def signal_metrics(filt, spec, kind, bw, alpha=0.35, adj_db=10.0):
    """In-band loss of the wanted signal, adjacent-channel rejection, and EVM from the ISI the filter adds."""
    out = dict(loss_db=None, acr_db=None, evm_pct=None)
    if not filt.ok or kind in ("none",):
        return out
    f = np.linspace(0, spec.fs / 2, 8193)
    H2 = np.abs(filt.response(f, spec.fs)) ** 2
    want = input_spectrum("rrc" if kind == "rrc_adj" else kind, f, bw, alpha) ** 2
    out["loss_db"] = float(-10 * np.log10(np.sum(want * H2) / np.sum(want)))
    if kind == "rrc_adj":
        adj = (input_spectrum(kind, f, bw, alpha, adj_db) - np.sqrt(want)) ** 2
        if adj.sum() > 0:
            out["acr_db"] = float(10 * np.log10(adj.sum() / max(np.sum(adj * H2), 1e-300)))
    edge = (1 + alpha) * bw / 2
    if kind in ("rrc", "rrc_adj") and edge < spec.fs / 2:
        # Wanted RRC signal -> this filter -> matched RRC receiver. Overall pulse P(f) = RC(f) * H(f).
        n = 1024
        fg = (np.arange(n) + 0.5) * edge / n
        Pf = input_spectrum("rc", fg, bw, alpha) * filt.response(fg, spec.fs)
        T = 1.0 / bw
        ph = np.unwrap(np.angle(filt.response(np.array([1e-6 * spec.fs, 0.02 * bw]), spec.fs)))
        delay = -(ph[1] - ph[0]) / (2 * np.pi * (0.02 * bw - 1e-6 * spec.fs))          # group delay near DC
        tau = delay + np.linspace(-0.5, 0.5, 81) * T
        k = np.arange(-60, 61)
        t = tau[:, None] + k[None, :] * T
        p = 2 * np.real(np.tensordot(np.exp(2j * np.pi * t[..., None] * fg), Pf, axes=([2], [0]))) * edge / n
        i = int(np.argmax(np.abs(p[:, 60])))
        isi = np.sqrt(np.sum(p[i] ** 2) - p[i, 60] ** 2)
        out["evm_pct"] = float(100 * isi / abs(p[i, 60]))
    return out


# --------------------------------------------------------------------------- summary
def summary(filters, spec, bits=None, input_kind="none", bw=None, alpha=0.35, adj_db=10.0, with_min_bits=True):
    """One dict per filter: size, achieved ripple / attenuation, delay, multiplier counts and signal metrics.
    With bits set, the response figures and multiplier counts are those of the quantised filter."""
    rows = []
    for flt in filters:
        row = dict(key=flt.key, type=flt.label, size=flt.size_text, note=flt.note)
        if flt.ok:
            use = flt.quantized(bits) if bits else flt
            r, a = measure(use, spec)
            row.update(ripple_db=r, atten_db=a, note=use.note or flt.note)
            if flt.kind == "fir":
                row["delay"] = (len(flt.h) - 1) / 2
            else:
                f = np.linspace(0, spec.fpass, 400)
                ph = np.unwrap(np.angle(flt.response(f, spec.fs)))
                gd = -np.gradient(ph, 2 * np.pi * f / spec.fs)
                row["delay"], row["delay_var"] = float(gd[1]), float(np.ptp(gd[1:-1]))
            row.update(multipliers(use))
            if with_min_bits:
                row["min_bits"] = min_bits(flt, spec)
            if input_kind != "none":
                row.update(signal_metrics(use, spec, input_kind, bw, alpha, adj_db))
        rows.append(row)
    return rows


def _html_table(rows, bits, input_kind):
    cols = [("type", "Filter"), ("size", "Taps / order"), ("direct", "Multipliers: every non-zero coefficient"),
            ("symmetric", "after symmetry sharing"), ("nontrivial", "<b>non-trivial</b> (no zeros, no powers of two)"),
            ("distinct", "distinct constants"), ("csd_adders", "CSD adders"), ("ripple_db", "Ripple dB p-p"),
            ("atten_db", "Stopband dB"), ("delay", "Delay, samples"), ("min_bits", "Min coefficient bits")]
    if not bits:
        cols = [c for c in cols if c[0] != "csd_adders"]
    if input_kind != "none":
        cols.append(("loss_db", "Signal loss dB"))
    if input_kind == "rrc_adj":
        cols.append(("acr_db", "Adjacent rejection dB"))
    if input_kind in ("rrc", "rrc_adj"):
        cols.append(("evm_pct", "EVM from ISI %"))

    def cell(r, k):
        v = r.get(k)
        if k == "type":
            return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{TYPES[r["key"]][2]};margin-right:6px"></span>{v}'
        if k == "size" and r.get("note"):
            return f'{v} <span style="color:#a15c00">({r["note"]})</span>'
        if v is None:
            return "–"
        if k == "nontrivial":
            return f"<b>{v}</b>"
        if k == "delay":
            return f"{v:.1f}" + (f" (varies by {r['delay_var']:.1f} in band)" if "delay_var" in r else "")
        if k == "ripple_db":
            return f"{v:.4f}"
        if k == "evm_pct":
            return f"{v:.3f}"
        if k in ("atten_db", "acr_db"):
            return f"{v:.1f}"
        if k == "loss_db":
            return f"{v:.3f}"
        return str(v)

    th = "".join(f'<th style="text-align:left;padding:4px 10px;border-bottom:1px solid #999;vertical-align:bottom;max-width:130px">{t}</th>' for _, t in cols)
    tr = "".join("<tr>" + "".join(f'<td style="text-align:left;padding:4px 10px;white-space:nowrap">{cell(r, k)}</td>' for k, _ in cols) + "</tr>" for r in rows)
    head = f"Response and multiplier figures are for coefficients quantised to {bits} bits." if bits else "Figures are for unquantised coefficients."
    return f'<div style="font-size:12px;color:#666;margin:6px 0">{head}</div><table style="border-collapse:collapse;font-size:13px"><tr>{th}</tr>{tr}</table>'


# --------------------------------------------------------------------------- plots
def _db(a, floor):
    return np.maximum(20 * np.log10(np.abs(a) + 1e-300), floor - 20)


def plot_responses(filters, spec, bits=None, floor=None):
    """Magnitude responses with the spec mask, plus passband detail, group delay and impulse responses.
    With bits set, the quantised response is drawn dashed in the same colour."""
    good = [f for f in filters if f.ok]
    floor = floor if floor is not None else -(np.ceil(spec.atten_db / 20) * 20 + 40)
    f = _grid(spec, max([x.size for x in good if x.kind == "fir"], default=0))
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :])
    a1, a2, a3 = (fig.add_subplot(gs[1, i]) for i in range(3))
    grey = "#8a939c"
    ax.fill_between([spec.fstop, spec.fs / 2], -spec.atten_db, 10, color=grey, alpha=0.18, lw=0)
    ax.fill_between([0, spec.fpass], floor, -max(spec.ripple_db, 0.3), color=grey, alpha=0.18, lw=0)
    gd_max = 1.0
    for flt in good:
        H = flt.response(f, spec.fs)
        ax.plot(f, _db(H, floor), color=flt.color, lw=1.6, label=f"{flt.label}, {flt.size_text}")
        pb = f <= spec.fpass * 1.02
        a1.plot(f[pb], 20 * np.log10(np.abs(H[pb])), color=flt.color, lw=1.4)
        if bits:
            Hq = flt.quantized(bits).response(f, spec.fs)
            ax.plot(f, _db(Hq, floor), color=flt.color, lw=1.1, ls="--")
            a1.plot(f[pb], 20 * np.log10(np.abs(Hq[pb]) + 1e-300), color=flt.color, lw=1.0, ls="--")
        fg = f[f <= spec.fstop]
        if flt.kind == "fir":
            a2.plot(fg, np.full(len(fg), (len(flt.h) - 1) / 2), color=flt.color, lw=1.4)
            gd_max = max(gd_max, (len(flt.h) - 1) / 2)
        else:
            ph = np.unwrap(np.angle(H[: len(fg)]))
            gd = -np.gradient(ph, 2 * np.pi * fg / spec.fs)
            a2.plot(fg[1:], gd[1:], color=flt.color, lw=1.4)
            gd_max = max(gd_max, np.nanmax(gd[1:][fg[1:] <= spec.fpass]))
        imp = flt.impulse(None if flt.kind == "fir" else int(min(400, max(60, 12 * flt.order))))
        a3.plot(np.arange(len(imp)), imp, color=flt.color, lw=1.1, marker="o" if len(imp) <= 80 else None, ms=2.5)
    if bits:
        ax.plot([], [], color="#444", lw=1.1, ls="--", label=f"dashed: {bits}-bit coefficients")
    ax.set(xlim=(0, spec.fs / 2), ylim=(floor, 5), xlabel="frequency", ylabel="dB")
    ax.set_title("Magnitude response. Shaded: regions the spec forbids.", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    a1.set(xlim=(0, spec.fpass * 1.02), ylim=(-1.3 * spec.ripple_db, 0.8 * spec.ripple_db), xlabel="frequency", ylabel="dB")
    a1.set_title("Passband detail (FIR ripples about 0 dB, IIR below it)", loc="left", fontsize=10)
    a2.set(xlim=(0, spec.fstop), ylim=(0, 1.2 * gd_max), xlabel="frequency", ylabel="samples")
    a2.axvline(spec.fpass, color=grey, lw=0.8, ls=":")
    a2.set_title("Group delay (to the stopband edge)", loc="left", fontsize=10)
    a3.set(xlabel="sample", ylabel="amplitude")
    a3.set_title("Impulse response", loc="left", fontsize=10)
    for a in (ax, a1, a2, a3):
        a.grid(True, color="#d6dce2", lw=0.6)
    fig.tight_layout()
    return fig


def plot_signal(filters, spec, kind, bw, alpha=0.35, adj_db=10.0, bits=None, floor=None):
    """Input spectrum and what each filter leaves of it. Uses the quantised filters when bits is set."""
    good = [f for f in filters if f.ok]
    floor = floor if floor is not None else -(np.ceil(spec.atten_db / 20) * 20 + 40)
    f = _grid(spec, max([x.size for x in good if x.kind == "fir"], default=0))
    X = input_spectrum(kind, f, bw, alpha, adj_db)
    ref = input_spectrum("rrc" if kind == "rrc_adj" else kind, np.array([0.0]), bw, alpha)[0]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(f, _db(X / ref, floor), color="#10151b", lw=1.8, label=f"input: {INPUTS[kind]}")
    for flt in good:
        use = flt.quantized(bits) if bits else flt
        ax.plot(f, _db(X / ref * use.response(f, spec.fs), floor), color=flt.color, lw=1.4, label=f"after {flt.label}")
    ax.axvline(spec.fpass, color="#8a939c", lw=0.8, ls=":")
    ax.axvline(spec.fstop, color="#8a939c", lw=0.8, ls=":")
    ax.set(xlim=(0, spec.fs / 2), ylim=(floor, max(5, (adj_db + 5) if kind == "rrc_adj" else 5)), xlabel="frequency", ylabel="dB")
    ax.set_title("Input spectrum before and after filtering" + (f" ({bits}-bit coefficients)" if bits else "") + ". Dotted: passband and stopband edges.", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, color="#d6dce2", lw=0.6)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- export
def export(filters, spec, folder="filter_export", bits=None):
    """Write coefficients for every designed filter. Returns the list of files written.

    FIR:  <type>_float.txt (one tap per line); with bits also <type>_q<bits>.mem (two's complement
          hex for $readmemh) and <type>_q<bits>.coe (Xilinx COE).
    IIR:  <type>_sos.txt (scipy second-order sections, gain folded into the first section); with bits
          also <type>_sos_q<bits>.txt (integers b1 b2 a1 a2 per section, and the gain).
    All:  filters.json with the spec, every coefficient set and the multiplier counts.
    """
    os.makedirs(folder, exist_ok=True)
    files, doc = [], {"spec": spec.__dict__, "filters": {}}

    def save(name, text):
        path = os.path.join(folder, name)
        with open(path, "w") as fh:
            fh.write(text)
        files.append(path)

    for flt in filters:
        if not flt.ok:
            continue
        q = flt.quantized(bits) if bits else None
        entry = {"label": flt.label, "kind": flt.kind, "size": flt.size, "multipliers": multipliers(q or flt)}
        if flt.kind == "fir":
            save(f"{flt.key}_float.txt", "\n".join(f"{v:.17g}" for v in flt.h) + "\n")
            entry["taps"] = [float(v) for v in flt.h]
            if q:
                digits = (bits + 3) // 4
                save(f"{flt.key}_q{bits}.mem", "\n".join(f"{int(v) & ((1 << bits) - 1):0{digits}x}" for v in q.q) + "\n")
                save(f"{flt.key}_q{bits}.coe", "radix=10;\ncoefdata=\n" + ",\n".join(str(int(v)) for v in q.q) + ";\n")
                entry.update(quantized=[int(v) for v in q.q], frac_bits=q.frac, bits=bits)
        else:
            full = np.column_stack([np.ones(len(flt.sos)), flt.sos[:, 1:]])
            full[0, :3] *= flt.gain
            save(f"{flt.key}_sos.txt", "# b0 b1 b2 a0 a1 a2 per section\n" + "\n".join(" ".join(f"{v:.17g}" for v in r) for r in full) + "\n")
            entry.update(sos=full.tolist())
            if q:
                txt = (f"# {bits}-bit coefficients, {q.frac} fractional bits. Section: y = x + b1*x1 + b2*x2 - a1*y1 - a2*y2 (b0 = a0 = 1)\n"
                       f"# overall gain = {q.qgain[0]} / 2^{q.qgain[1]}\n# b1 b2 a1 a2\n")
                save(f"{flt.key}_sos_q{bits}.txt", txt + "\n".join(" ".join(str(int(v)) for v in r) for r in q.q) + "\n")
                entry.update(quantized=q.q.tolist(), frac_bits=q.frac, gain=list(q.qgain), bits=bits, stable=q.stable())
        doc["filters"][flt.key] = entry
    save("filters.json", json.dumps(doc, indent=1))
    return files


# --------------------------------------------------------------------------- GUI
def designer():
    """Interactive designer for Jupyter."""
    import ipywidgets as w
    from IPython.display import display, HTML
    sty = {"description_width": "150px"}
    box = lambda v, d, **k: w.BoundedFloatText(value=v, description=d, style=sty, layout=w.Layout(width="270px"), **k)
    c = dict(
        fs=box(100, "Sample rate", min=1e-9, max=1e15), fpass=box(10, "Passband edge", min=1e-9, max=1e15),
        fstop=box(15, "Stopband edge", min=1e-9, max=1e15),
        ripple=box(0.1, "Passband ripple, dB p-p", min=1e-4, max=6, step=0.01),
        atten=box(60, "Stopband atten., dB", min=10, max=140, step=1),
        quant=w.Checkbox(value=False, description="Quantise coefficients", indent=False, layout=w.Layout(width="200px")),
        bits=w.BoundedIntText(value=16, min=4, max=32, description="Coefficient bits", style=sty, layout=w.Layout(width="230px")),
        inp=w.Dropdown(options=[(v, k) for k, v in INPUTS.items()], value="none", description="Input spectrum", style=sty, layout=w.Layout(width="400px")),
        bw=box(14, "Bandwidth (two-sided)", min=1e-9, max=1e15),
        alpha=w.FloatSlider(value=0.35, min=0.0, max=1.0, step=0.01, description="Roll-off alpha", continuous_update=False, style=sty, layout=w.Layout(width="400px")),
        adj=box(10, "Adjacent level, dB", min=-60, max=80, step=1),
    )
    checks = {k: w.Checkbox(value=k in ("equiripple", "ls", "kaiser"), description=TYPES[k][0], indent=False, layout=w.Layout(width="240px")) for k in TYPES}
    folder = w.Text(value="filter_export", description="Export folder", style=sty, layout=w.Layout(width="330px"))
    btn = w.Button(description="Export coefficients", button_style="primary")
    msg = w.HTML()

    def current():
        spec = Spec(c["fs"].value, c["fpass"].value, c["fstop"].value, c["ripple"].value, c["atten"].value)
        return spec, [k for k in TYPES if checks[k].value], (c["bits"].value if c["quant"].value else None)

    def show(**_):
        spec, keys, bits = current()
        try:
            spec.check()
        except ValueError as e:
            print(f"Specification problem: {e}.")
            return
        if not keys:
            print("Tick at least one filter type.")
            return
        filters = design_all(spec, keys)
        kind = c["inp"].value
        display(HTML(_html_table(summary(filters, spec, bits, kind, c["bw"].value, c["alpha"].value, c["adj"].value), bits, kind)))
        if not any(f.ok for f in filters):
            return
        plot_responses(filters, spec, bits)
        if kind != "none":
            plot_signal(filters, spec, kind, c["bw"].value, c["alpha"].value, c["adj"].value, bits)
        plt.show()

    def on_export(_):
        spec, keys, bits = current()
        try:
            files = export(design_all(spec, keys), spec, folder.value, bits)
            msg.value = f"Wrote {len(files)} files to <code>{os.path.abspath(folder.value)}</code>"
        except Exception as e:
            msg.value = f'<span style="color:#b00020">Export failed: {e}</span>'

    btn.on_click(on_export)
    ctl = {**{k: v for k, v in c.items()}, **{"t_" + k: v for k, v in checks.items()}}
    out = w.interactive_output(show, ctl)
    head = lambda t: w.HTML(f'<b style="font-size:13px">{t}</b>')
    ui = w.VBox([
        w.HBox([
            w.VBox([head("Specification (lowpass)"), c["fs"], c["fpass"], c["fstop"], c["ripple"], c["atten"]]),
            w.VBox([head("Filter types to compare"), w.GridBox(list(checks.values()), layout=w.Layout(grid_template_columns="240px 240px"))]),
            w.VBox([head("Input to the filter"), c["inp"], c["bw"], c["alpha"], c["adj"]]),
        ]),
        w.HBox([c["quant"], c["bits"], folder, btn, msg]),
        out,
    ])
    display(ui)
