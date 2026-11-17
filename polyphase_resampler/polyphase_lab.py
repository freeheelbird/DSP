"""Polyphase Resampler Lab (Python version).

Prototype filter design (equiripple, least-squares, half-band), the polyphase
structure with the contents of every storage unit, and exact spectra at each
stage of   x[n] -> up L -> H(z) -> down M -> y[m].

Quick start in a notebook:

    %matplotlib inline
    import polyphase_lab as pl
    pl.lab()                       # interactive controls (needs ipywidgets)

or call the pieces directly:

    h = pl.design_prototype(L=3, M=2, N=48, method="equiripple")
    pl.plot_filter(h, L=3, M=2)
    pl.plot_structure(h, L=3, M=2, m=23, form="L")
    pl.plot_spectra(h, L=3, M=2, signal="band", bw=0.75)

Requires numpy, scipy, matplotlib. ipywidgets only for lab().
"""
import numpy as np
from scipy import signal as sps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a939c"
METHODS = ("equiripple", "ls", "halfband")
SIGNALS = ("band", "tones", "noise", "interferer")


# --------------------------------------------------------------------------- design
def band_edges(L, M, pass_edge=0.8, stop_edge=1.2, halfband=False):
    """Passband edge, stopband edge and cutoff in cycles/sample at the prototype rate (L * f_in)."""
    if halfband:
        fp = min(pass_edge, 0.98) * 0.25
        return fp, 0.5 - fp, 0.25
    fc = 0.5 / max(L, M)
    fs = min(stop_edge * fc, 0.497)
    return min(pass_edge * fc, fs - 0.003), fs, fc


def _ls_fit(nu, bands):
    """Weighted least-squares fit of sum_k a_k cos(nu_k w) to D on bands [(w1, w2, D, W), ...]."""
    nu = np.asarray(nu, float)
    d, s = nu[:, None] - nu[None, :], nu[:, None] + nu[None, :]

    def I(al, w1, w2):
        al = np.asarray(al, float)
        safe = np.where(np.abs(al) < 1e-12, 1.0, al)
        return np.where(np.abs(al) < 1e-12, w2 - w1, (np.sin(safe * w2) - np.sin(safe * w1)) / safe)

    Q, b = np.zeros((len(nu), len(nu))), np.zeros(len(nu))
    for w1, w2, D, W in bands:
        Q += 0.5 * W * (I(d, w1, w2) + I(s, w1, w2))
        b += W * D * I(nu, w1, w2)
    Q += np.eye(len(nu)) * 1e-14 * np.trace(Q) / len(nu)
    return np.linalg.solve(Q, b)


def design_prototype(L, M, N, method="equiripple", pass_edge=0.8, stop_edge=1.2, weight=1.0, hb_opt="equiripple"):
    """Linear-phase lowpass prototype with (near) unity DC gain. Multiply by L when interpolating.

    method     "equiripple" (Parks-McClellan), "ls" (least-squares) or "halfband"
    pass_edge  passband edge as a multiple of the cutoff fc = 0.5 / max(L, M)   (prototype rate)
    stop_edge  stopband edge as a multiple of fc
    weight     stopband error weight relative to the passband
    hb_opt     how a half-band filter is optimised: "equiripple" or "ls"

    A half-band filter always cuts off at a quarter of the prototype rate, so it only suits
    max(L, M) == 2. Its length is snapped to N = 4q + 3 and every other tap is exactly zero.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    fp, fs, _ = band_edges(L, M, pass_edge, stop_edge, method == "halfband")
    if method == "halfband":
        q = max(0, round((N - 3) / 4))
        N, c = 4 * q + 3, 2 * q + 1
        h = np.zeros(N)
        h[c] = 0.5
        if hb_opt == "ls":
            a = _ls_fit(2 * np.arange(q + 1) + 1, [(0, 2 * np.pi * fp, 0.5, 1.0)])
            k = 2 * np.arange(q + 1) + 1
            h[c - k] = h[c + k] = a / 2
        else:  # design a one-band filter at half the rate, then interleave zeros
            g = sps.remez(2 * q + 2, [0, 2 * fp], [1], fs=1, maxiter=200)
            h[0::2] = g / 2
        return h
    if method == "equiripple":
        return sps.remez(N, [0, fp, fs, 0.5], [1, 0], weight=[1, weight], fs=1, maxiter=200)
    bands = [(0, 2 * np.pi * fp, 1.0, 1.0), (2 * np.pi * fs, np.pi, 0.0, weight)]
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


def filter_stats(h, L, M, pass_edge=0.8, stop_edge=1.2, halfband=False):
    """Passband ripple (dB peak to peak) and stopband rejection (dB) of a prototype."""
    fp, fs, _ = band_edges(L, M, pass_edge, stop_edge, halfband)
    w, H = sps.freqz(h, worN=max(4096, 16 * len(h)))
    f, d = w / (2 * np.pi), 20 * np.log10(np.abs(H) / abs(H[0]) + 1e-300)
    return {"taps": len(h), "taps_per_leg": -(-len(h) // L), "ripple_db": float(np.ptp(d[f <= fp])),
            "rejection_db": float(-d[f >= fs].max()), "pass_edge_fin": fp * L, "stop_edge_fin": fs * L}


# --------------------------------------------------------------------------- structure
def legs(h, R):
    """Polyphase decomposition into R legs. Row r holds h[r], h[r + R], h[r + 2R], ... (zero padded)."""
    K = -(-len(h) // R)
    hp = np.zeros(K * R)
    hp[:len(h)] = h
    return hp.reshape(K, R).T


def leg_contents(h, L, M, m, form="L"):
    """Which input sample sits in every storage unit when output y[m] is computed.

    form="L": L legs, output commutator. All legs share one delay line; leg p = (m*M) % L is used.
    form="M": M legs, input commutator. Each leg has its own delay line and all legs are summed.
              With L > 1 the legs hold the zero-stuffed input, so some units hold 0 (returned as None).

    Returns (x_index, tap_index, active_leg). x_index[r][j] is n such that the unit holds x[n];
    tap_index[r][j] is i such that the unit is weighted by h[i] (None where there is no tap).
    """
    N, R = len(h), (L if form == "L" else M)
    K = -(-N // R)
    p, n = (m * M) % L, (m * M) // L
    xi, ti = [], []
    for r in range(R):
        row = []
        for j in range(K):
            if form == "L":
                row.append(n - j)
            else:
                vi = (m - j) * M - r
                row.append(vi // L if vi % L == 0 else None)
        xi.append(row)
        ti.append([r + j * R if r + j * R < N else None for j in range(K)])
    return xi, ti, (p if form == "L" else None)


def polyphase_resample(x, h, L, M):
    """Rational resampler exactly as the L-leg diagram computes it (one output per loop pass).

    y[m] = L * sum_j h[p + j*L] * x[n - j],   p = (m*M) % L,   n = (m*M) // L
    Matches scipy.signal.upfirdn(L * h, x, L, M).
    """
    x = np.asarray(x)
    E = legs(h, L)                     # E[p, j] = h[p + j*L]
    K = E.shape[1]
    xp = np.concatenate([np.zeros(K - 1, x.dtype), x])
    n_out = -(-(len(x) * L) // M)
    y = np.zeros(n_out, dtype=np.result_type(x, float))
    for m in range(n_out):
        p, n = (m * M) % L, (m * M) // L
        y[m] = L * np.dot(E[p], xp[n:n + K][::-1])     # delay line holds x[n], x[n-1], ...
    return y


# --------------------------------------------------------------------------- spectra
def resample_spectra(h, L, M, signal="band", bw=0.75, seed=12345):
    """Exact steady-state amplitude spectra of a periodic test signal at each stage.

    Frequencies are in units of f_in at every stage. The filter is applied at unity DC gain
    (the interpolation gain of L is implied). Returns a dict of frequency / amplitude arrays:
    f_in, x | f_up, up, filt, H | f_out, out, err   (err = image and alias content only).
    """
    Q = max(-(-8192 // (L * M)), -(-512 // M))
    Q += (Q * M) % 2
    P, LQ = M * Q, L * Q
    PL, half = P * L, P // 2
    rng = np.random.default_rng(seed)
    X = np.zeros(P, complex)
    k = np.arange(1, half)
    ph = np.exp(2j * np.pi * rng.random(half - 1))
    B = bw * half

    def tone(frac, db):
        X[int(np.clip(round(frac * half), 1, half - 1))] += 10 ** (db / 20) * np.exp(2j * np.pi * rng.random())

    if signal in ("band", "interferer"):
        X[1:half] = np.where(k <= B, 10 ** (-30 * (k / B) / 20), 0) * ph
        if signal == "interferer":
            tone(0.93, 0)
    elif signal == "noise":
        X[1:half] = ph
    elif signal == "tones":
        for f, d in ((0.12, 0), (0.37, -6), (0.63, -12), (0.94, -18)):
            tone(f * bw, d)
    else:
        raise ValueError(f"signal must be one of {SIGNALS}")
    X[P - k] = np.conj(X[k])
    V = np.tile(X, L)                                   # zero-stuffing repeats the spectrum L times
    H = np.fft.fft(h, PL)
    W = V * H
    Y = W.reshape(M, LQ).sum(axis=0)                    # decimation folds M slices on top of each other
    KO = LQ // 2
    want = np.where(np.arange(KO + 1) < half, W[:KO + 1], 0)
    return {"f_in": np.arange(half + 1) / P, "x": np.abs(X[:half + 1]),
            "f_up": np.arange(PL // 2 + 1) / P, "up": np.abs(V[:PL // 2 + 1]),
            "filt": np.abs(W[:PL // 2 + 1]), "H": np.abs(H[:PL // 2 + 1]),
            "f_out": np.arange(KO + 1) / P, "out": np.abs(Y[:KO + 1]), "err": np.abs(Y[:KO + 1] - want),
            "half": half}


# --------------------------------------------------------------------------- plots
def _db(a, ref, floor):
    return np.maximum(20 * np.log10(np.maximum(a, 1e-300) / ref), floor - 40)


def _style(ax, xlabel, ylabel, floor=None):
    ax.grid(True, color="#d6dce2", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(x=0)
    if floor is not None:
        ax.set_ylim(floor, 10)


def plot_filter(h, L, M, m=0, pass_edge=0.8, stop_edge=1.2, halfband=False, floor=-120):
    """Magnitude response and impulse response. Taps used for output y[m] are highlighted."""
    fp, fs, fc = band_edges(L, M, pass_edge, stop_edge, halfband)
    w, H = sps.freqz(h, worN=max(4096, 16 * len(h)))
    f = w / (2 * np.pi) * L
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.6))
    a.axvspan(0, fp * L, color=BLUE, alpha=0.1)
    a.axvspan(fs * L, L / 2, color=GREY, alpha=0.15)
    a.axvline(fc * L, color=GREY, ls="--", lw=1)
    a.plot(f, _db(np.abs(H), abs(H[0]), floor), color=BLUE, lw=1.6)
    st = filter_stats(h, L, M, pass_edge, stop_edge, halfband)
    a.set_title(f"|H(f)|   ripple {st['ripple_db']:.4f} dB p-p,  rejection {st['rejection_db']:.1f} dB", loc="left", fontsize=10)
    _style(a, "frequency ( x f_in )", "dB", floor)
    i, res = np.arange(len(h)), (m * M) % L
    on = i % L == res
    for mask, col, lw in ((~on, GREY, 1.0), (on, BLUE, 1.8)):
        if mask.any():
            b.vlines(i[mask], 0, h[mask], color=col, lw=lw)
            b.plot(i[mask], h[mask], "o", color=col, ms=3)
    b.axhline(0, color=GREY, lw=0.8)
    b.set_title(f"h[i], {len(h)} taps.  Blue: leg {res} of {L} (i mod {L} = {res})", loc="left", fontsize=10)
    _style(b, "tap index i", "h[i]")
    fig.tight_layout()
    return fig


def plot_structure(h, L, M, m, form="L", max_units=16):
    """Draw the legs with the sample held in every storage unit and the tap that weights it."""
    xi, ti, p = leg_contents(h, L, M, m, form)
    R, K = len(xi), len(xi[0])
    Kd = min(K, max_units)
    n_prev = ((m - 1) * M) // L
    fig, ax = plt.subplots(figsize=(min(18, 2.2 + 1.05 * Kd), 0.5 + 1.0 * R))
    for r in range(R):
        y = -r
        active = form == "M" or r == p
        ax.text(-0.25, y, f"leg {r}", ha="right", va="center", fontsize=9, color=BLUE if active and form == "L" else "black", fontweight="bold")
        for j in range(Kd):
            n, i = xi[r][j], ti[r][j]
            zero = n is None or i is None
            new = (n is not None) and (n > n_prev if form == "L" else (m - j) * M - r > (m - 1) * M)
            alpha = (1.0 if active else 0.4) * (0.35 if zero else 1.0)
            ax.add_patch(Rectangle((j + 0.08, y - 0.2), 0.84, 0.4, fc="#dbe8f8" if active and not zero else "#f1f4f7",
                                   ec=ORANGE if new else (BLUE if active else GREY), lw=2 if new else 1, alpha=alpha))
            ax.text(j + 0.5, y, "0" if n is None else f"x[{n}]", ha="center", va="center", fontsize=8, alpha=alpha, family="monospace")
            if i is not None:
                lab = f"h[{i}]" + ("=0" if abs(h[i]) < 1e-12 else "")
                ax.text(j + 0.5, y - 0.36, lab, ha="center", va="center", fontsize=7, color="#56606b", alpha=alpha, family="monospace")
        if Kd < K:
            ax.text(Kd + 0.15, y, f"... {K - Kd} more", va="center", fontsize=8, color="#56606b")
    if form == "L":
        n = (m * M) // L
        title = (f"y[{m}]: leg p = ({m}*{M}) mod {L} = {p}, newest input x[{n}], {n - n_prev} new since y[{m - 1}].  "
                 f"All {L} legs share one {K}-sample delay line.")
    else:
        title = f"y[{m}] sums all {M} legs; the input commutator loads leg {M - 1} up to leg 0." + (
            f"  0 = zero-stuffed sample; real samples meet taps of L-form leg {(m * M) % L}." if L > 1 else "")
    ax.set_title(title + "   Orange outline: shifted in since the previous output.", loc="left", fontsize=9)
    ax.set_xlim(-1.2, Kd + 1.6)
    ax.set_ylim(-R + 0.35, 0.55)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_legs(h, L, M, m=0, form="L"):
    """Magnitude and group delay of every leg. Each leg is close to an all-pass with its own fractional delay."""
    R = L if form == "L" else M
    E, hi = legs(h, R) * R, ((m * M) % L if form == "L" else 0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.2))
    for r in [x for x in range(R) if x != hi] + [hi]:
        w, H = sps.freqz(E[r], worN=512)
        _, gd = sps.group_delay((E[r], [1.0]), w=w)
        kw = dict(color=BLUE, lw=2, label=f"leg {hi}") if r == hi else dict(color=GREY, lw=0.8, alpha=0.6)
        a.plot(w / (2 * np.pi), np.maximum(20 * np.log10(np.abs(H) + 1e-300), -80), **kw)
        b.plot(w / (2 * np.pi), gd, **kw)
    c = (len(h) - 1) / 2 / R
    b.set_ylim(c - (R - 1) / R - 0.2, c + 0.2)
    a.set_ylim(-60, 10)
    a.legend(loc="lower left", fontsize=8)
    rate = "f_in" if form == "L" else "f_out"
    a.set_title("Leg magnitude", loc="left", fontsize=10)
    b.set_title("Leg group delay", loc="left", fontsize=10)
    _style(a, f"frequency ( x {rate}, the leg rate )", "dB")
    _style(b, f"frequency ( x {rate} )", "samples")
    fig.tight_layout()
    return fig


def plot_spectra(h, L, M, signal="band", bw=0.75, floor=-120):
    """Four stacked spectra: input, after up-L (with |H|), after H(z), and the output after down-M."""
    r = resample_spectra(h, L, M, signal, bw)
    ref, hf = r["x"].max(), r["half"]
    d = lambda a: _db(a, ref, floor)
    fig, ax = plt.subplots(4, 1, figsize=(13, 10))
    ax[0].plot(r["f_in"], d(r["x"]), color=BLUE, lw=1.2)
    ax[0].set_title("1. Input x[n], rate f_in", loc="left", fontsize=10)
    for a_, y, name in ((ax[1], r["up"], "images"), (ax[2], r["filt"], "residual images")):
        a_.plot(r["f_up"][:hf + 1], d(y[:hf + 1]), color=BLUE, lw=1.2, label="baseband")
        if L > 1:
            a_.plot(r["f_up"][hf:], d(y[hf:]), color=ORANGE, lw=1.2, label=name)
        if M > 1:
            a_.axvline(L / (2 * M), color=GREY, ls="--", lw=1, label="f_out/2 (content above folds down)")
    ax[1].plot(r["f_up"], d(r["H"] * ref), color=AQUA, lw=1.4, label="filter |H|")
    ax[1].set_title(f"2. After up {L}, rate {L} f_in", loc="left", fontsize=10)
    ax[2].set_title(f"3. After H(z), rate {L} f_in", loc="left", fontsize=10)
    ax[3].plot(r["f_out"], d(r["err"]), color=ORANGE, lw=1.2, label="image and alias content only")
    ax[3].plot(r["f_out"], d(r["out"]), color=BLUE, lw=1.2, label="output")
    ax[3].set_title(f"4. Output y[m] after down {M}, rate {L / M:g} f_in", loc="left", fontsize=10)
    for a_ in ax:
        _style(a_, "frequency ( x f_in )", "dB", floor)
    for a_ in ax[1:]:
        a_.legend(loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- interactive
def lab():
    """Interactive version for Jupyter: sliders for L, M, the filter, the test signal and the output step."""
    import ipywidgets as w
    from IPython.display import display
    sl = dict(continuous_update=False, style={"description_width": "110px"}, layout=w.Layout(width="340px"))
    c = {
        "L": w.IntSlider(3, 1, 64, description="Interpolate L", **sl),
        "M": w.IntSlider(2, 1, 64, description="Decimate M", **sl),
        "N": w.IntSlider(48, 3, 1025, description="Taps N", **sl),
        "method": w.Dropdown(options=[("Equiripple", "equiripple"), ("Least-squares", "ls"), ("Half-band", "halfband")], description="Design", **sl),
        "hb_opt": w.Dropdown(options=[("Equiripple", "equiripple"), ("Least-squares", "ls")], description="Half-band by", **sl),
        "pass_edge": w.FloatSlider(0.8, min=0.2, max=0.98, step=0.01, description="Pass edge x fc", **sl),
        "stop_edge": w.FloatSlider(1.2, min=0.5, max=2.0, step=0.01, description="Stop edge x fc", **sl),
        "weight": w.FloatLogSlider(1.0, base=10, min=-2, max=3, step=0.05, description="Stop weight", **sl),
        "signal": w.Dropdown(options=[("Sloped band", "band"), ("Four tones", "tones"), ("Flat noise", "noise"), ("Band + tone near Nyquist", "interferer")], description="Test signal", **sl),
        "bw": w.FloatSlider(0.75, min=0.1, max=1.0, step=0.01, description="Bandwidth", **sl),
        "form": w.ToggleButtons(options=[("L legs, output commutator", "L"), ("M legs, input commutator", "M")], description="Structure", style={"description_width": "110px"}),
        "m": w.IntSlider(23, 0, 400, description="Output index m", **sl),
        "floor": w.Dropdown(options=[-80, -120, -160], value=-120, description="Plot floor dB", **sl),
    }

    def show(L, M, N, method, hb_opt, pass_edge, stop_edge, weight, signal, bw, form, m, floor):
        if method != "halfband" and stop_edge < pass_edge + 0.02:
            stop_edge = pass_edge + 0.02
        hb = method == "halfband"
        try:
            h = design_prototype(L, M, N, method, pass_edge, stop_edge, weight, hb_opt)
        except Exception as e:                               # remez can fail on extreme specs
            print(f"Design failed ({e}). Widen the transition band or change N.")
            return
        if hb and max(L, M) != 2:
            print(f"Note: a half-band filter cuts off at 1/4 of the prototype rate, which suits L or M = 2. "
                  f"Here the cutoff belongs at 1/{2 * max(L, M)}, so images or aliases get through.")
        print(f"f_out = {L}/{M} f_in = {L / M:g} f_in.   {len(h)} taps, {-(-len(h) // L)} per leg of {L}.")
        plot_filter(h, L, M, m, pass_edge, stop_edge, hb, floor)
        plot_structure(h, L, M, m, form)
        plot_legs(h, L, M, m, form)
        plot_spectra(h, L, M, signal, bw, floor)
        plt.show()

    out = w.interactive_output(show, c)
    col = lambda *k: w.VBox([c[x] for x in k])
    display(w.VBox([w.HBox([col("L", "M", "N", "m"), col("method", "hb_opt", "pass_edge", "stop_edge", "weight"), col("signal", "bw", "floor")]), c["form"], out]))
