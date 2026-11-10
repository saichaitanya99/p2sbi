import numpy as np

OVERSAMPLE = 8
MARGIN     = 4.0
MIN_SPAN   = 64.0
MAX_WORK   = 2 ** 18


def scintle_coordinate(nu_mhz, bw_ref_mhz, bw_index, ref_mhz):
    x = np.asarray(nu_mhz, dtype=float) / ref_mhz
    if abs(bw_index - 1.0) < 1e-9:
        return ref_mhz / bw_ref_mhz * np.log(x)
    return ref_mhz / bw_ref_mhz * (x ** (1.0 - bw_index) - 1.0) / (1.0 - bw_index)


def time_variance(t_obs_s, dt_d_s):
    from scipy.special import erf
    s = np.asarray(t_obs_s / np.asarray(dt_d_s, dtype=float), dtype=float)
    small = s < 1e-4
    s_ = np.where(small, 1.0, s)
    v = (2.0 / s_**2) * (s_ * 0.5 * np.sqrt(np.pi) * erf(s_) - 0.5 * (1.0 - np.exp(-s_**2)))
    return np.where(small, 1.0, v)


def frequency_variance(width_u):
    W = np.asarray(width_u, dtype=float)
    small = W < 1e-4
    W_ = np.where(small, 1.0, W)
    v = 2.0 * (W_ * np.arctan(W_) - 0.5 * np.log1p(W_**2)) / W_**2
    return np.where(small, 1.0, v)


def snapshot_weights(n_t):
    n_t = np.maximum(np.asarray(n_t, dtype=float), 1.0)
    K = int(np.ceil(n_t.max() - 1e-9))
    W = np.zeros((n_t.size, K))
    for i, N in enumerate(n_t):
        n = int(np.floor(N + 1e-9))
        W[i, :n] = 1.0
        if N - n > 1e-9 and n < K:
            a, b, c = N - 1.0, -2.0 * n, N * n - n * n
            W[i, n] = (-b - np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    return W


def draw_scint_gain(values, freqs_mhz, edges_mhz, ref_mhz, method="auto"):
    bw, a = float(values["bandwidth_mhz"]), float(values["bandwidth_index"])
    dt, b = float(values["timescale_s"]), float(values["timescale_index"])
    t_obs = float(values["t_obs_s"])
    freqs = np.asarray(freqs_mhz, dtype=float)
    edges = np.asarray(edges_mhz, dtype=float)

    u_edges = scintle_coordinate(edges, bw, a, ref_mhz)
    u_lo, u_hi = u_edges[:-1], u_edges[1:]
    n_t = 1.0 / time_variance(t_obs, dt * (freqs / ref_mhz) ** b)

    span = max(u_edges[-1] - u_edges[0] + 2 * MARGIN, MIN_SPAN)
    M = int(2 ** np.ceil(np.log2(OVERSAMPLE * span)))
    K = int(np.ceil(n_t.max() - 1e-9))
    if method == "asymptotic" or (method == "auto" and M * K > MAX_WORK):
        var = frequency_variance(u_hi - u_lo) / n_t
        k = 1.0 / var
        return np.random.gamma(k, 1.0 / k)

    L = M / OVERSAMPLE
    u0 = u_edges[0] - 0.5 * (L - (u_edges[-1] - u_edges[0]))
    u = u0 + np.arange(M) * (L / M)
    w = 2.0 * np.pi * np.arange(M // 2) / L
    P = np.exp(-w); P /= P.sum()
    z = np.random.normal(size=(K, M // 2, 2))
    c = np.zeros((K, M), dtype=complex)
    c[:, :M // 2] = np.sqrt(P / 2.0) * (z[..., 0] + 1j * z[..., 1])
    E = np.fft.ifft(c, axis=1) * M
    I = (E * E.conj()).real

    du = L / M
    C = np.concatenate([np.zeros((K, 1)), np.cumsum(0.5 * (I[:, 1:] + I[:, :-1]) * du, axis=1)], axis=1)
    G = (_interp_rows(u_hi, u, C) - _interp_rows(u_lo, u, C)) / (u_hi - u_lo)
    Wt = snapshot_weights(n_t)
    return (Wt * G.T).sum(axis=1) / Wt.sum(axis=1)


def _interp_rows(x, u, C):
    j = np.clip(np.searchsorted(u, x) - 1, 0, u.size - 2)
    t = (x - u[j]) / (u[j + 1] - u[j])
    return C[:, j] * (1.0 - t) + C[:, j + 1] * t


def scattering_tau_ms(bandwidth_mhz):
    return 1.0 / (2.0 * np.pi * float(bandwidth_mhz) * 1e6) * 1e3
