import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvalsh
from collections import deque
import time
import warnings
from scipy.stats import ttest_rel, ttest_ind, chisquare

warnings.filterwarnings("ignore")

# ============================================================
# 1. Geometry Generators (φ-blind)
# ============================================================
def random_geometric_graph(N, radius=0.25):
    points = np.random.rand(N, 2)
    tree = KDTree(points)
    adj = tree.sparse_distance_matrix(tree, radius).toarray()
    adj = (adj > 0).astype(float)
    np.fill_diagonal(adj, 0)
    return points, adj

def lattice_graph(N):
    side = int(np.sqrt(N))
    xs, ys = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    points = np.column_stack([xs.ravel(), ys.ravel()])[:N]
    adj = np.zeros((N, N))
    for i in range(N):
        dist = np.linalg.norm(points - points[i], axis=1)
        adj[i, (dist < 1.5/side) & (dist > 1e-8)] = 1
    adj = ((adj + adj.T) > 0).astype(float)
    np.fill_diagonal(adj, 0)
    return points, adj

def penrose_graph(N_points=250):
    angles = 2 * np.pi * np.arange(5) / 5
    phys_vecs = np.column_stack((np.cos(angles), np.sin(angles)))
    perp_vecs = np.column_stack((np.cos(2*angles), np.sin(2*angles)))
    max_int = int(np.sqrt(N_points) * 1.8)
    coords = np.arange(-max_int, max_int + 1)
    grid = np.meshgrid(*([coords]*5), indexing='ij')
    points_5d = np.stack(grid, axis=-1).reshape(-1, 5)
    points_5d = points_5d[np.linalg.norm(points_5d, axis=1) < max_int*1.2]
    phys = points_5d @ phys_vecs
    perp = points_5d @ perp_vecs
    mask = np.linalg.norm(perp, axis=1) < 1.4
    phys = phys[mask]
    if len(phys) > N_points:
        idx = np.random.choice(len(phys), N_points, replace=False)
        phys = phys[idx]
    tree = KDTree(phys)
    adj = tree.sparse_distance_matrix(tree, 0.35).toarray()
    adj = (adj > 0).astype(float)
    np.fill_diagonal(adj, 0)
    return phys, adj

# ============================================================
# 2. Core Components
# ============================================================
class MemoryBuffer:
    def __init__(self, tau, dt, maxlen=400):
        self.tau = tau
        self.dt = dt
        self.buffer = deque(maxlen=maxlen)
        self.weights = np.exp(-np.arange(maxlen) * dt / tau)
        self.weights /= self.weights.sum()

    def update(self, C):
        self.buffer.append(C.copy())

    def predict(self):
        if not self.buffer:
            return None
        n = len(self.buffer)
        w = self.weights[:n]
        w /= w.sum()
        return np.sum(np.array(self.buffer) * w[:, None], axis=0)

class PinkNoise:
    def __init__(self, n_channels, n_octaves=10):
        self.n_channels = n_channels
        self.n_octaves = n_octaves
        self.reset()

    def reset(self):
        self.octave_values = np.random.randn(self.n_octaves, self.n_channels)
        self.counter = 0
        self.max_counter = 1 << self.n_octaves

    def next(self):
        if self.counter == 0:
            self.octave_values = np.random.randn(self.n_octaves, self.n_channels)
        else:
            lsb = (self.counter & -self.counter).bit_length() - 1
            self.octave_values[lsb] = np.random.randn(self.n_channels)
        return np.sum(self.octave_values, axis=0) / np.sqrt(self.n_octaves)

def negentropy_gradient(A):
    A2 = A**2
    safe = np.clip(A2, 1e-8, 1 - 1e-8)
    return 2 * A * (np.log(safe) - np.log(1 - safe))

def prediction_gradient(C, C_pred, g_pred):
    dtheta = -g_pred * np.imag(np.conj(C) * C_pred)
    dA = -g_pred * np.real(np.conj(C) * (C - C_pred) / (np.abs(C) + 1e-8))
    return dtheta, dA

def self_reference_gradient(C, adj, g_self):
    A = np.abs(C)
    A2 = A**2
    contrib = g_self * (adj @ A2)
    return A * contrib

# ============================================================
# 3. Metrics (continuous base inference + robustness)
# ============================================================
def spectral_ratio(W):
    deg = np.asarray(W.sum(axis=1)).flatten()
    L = diags(deg, 0) - W
    try:
        vals = eigs(L, k=2, which='SM', return_eigenvectors=False)
        vals = np.sort(np.real(vals))
        return vals[1] / vals[0] if vals[0] > 1e-6 else np.nan
    except:
        W_dense = W.toarray()
        L_dense = -W_dense
        np.fill_diagonal(L_dense, np.sum(W_dense, axis=1))
        eigvals = eigvalsh(L_dense)
        eigvals = eigvals[eigvals > 1e-6]
        return eigvals[1] / eigvals[0] if len(eigvals) >= 2 else np.nan

def refine_base(ratio, b_init, window=0.08, steps=80):
    bmin = max(1.01, b_init - window)
    bmax = b_init + window
    bases = np.linspace(bmin, bmax, steps)
    best_dev = float('inf')
    best_b = b_init
    for b in bases:
        if b <= 1: continue
        n_val = np.log(ratio) / np.log(b)
        dev = abs(n_val - round(n_val))
        if dev < best_dev:
            best_dev = dev
            best_b = b
    return best_dev, best_b

def infer_base_continuous(ratio, bmin=1.2, bmax=3.0, n=150):
    if ratio <= 0 or np.isnan(ratio):
        return np.nan, np.nan
    bases = np.linspace(bmin, bmax, n)
    best_dev = float('inf')
    best_b = np.nan
    for b in bases:
        if b <= 1: continue
        n_val = np.log(ratio) / np.log(b)
        dev = abs(n_val - round(n_val))
        if dev < best_dev:
            best_dev = dev
            best_b = b
    # Local refinement around best candidate
    if not np.isnan(best_b):
        best_dev, best_b = refine_base(ratio, best_b)
    return best_dev, best_b

def steady_mean(arr, frac=0.35):
    if len(arr) == 0:
        return np.nan
    n = len(arr)
    start = int((1 - frac) * n)
    return np.nanmean(arr[start:])

def box_counting_field(points, A, grid_size=192):
    x_min, x_max = points[:,0].min(), points[:,0].max()
    y_min, y_max = points[:,1].min(), points[:,1].max()
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    tree = KDTree(points)
    _, idx = tree.query(grid_points)
    A_grid = A[idx].reshape(grid_size, grid_size)
    dfs = []
    for thresh_factor in [0.3, 0.5, 0.7]:
        threshold = thresh_factor * A_grid.max()
        mask = A_grid > threshold
        box_sizes = np.logspace(np.log10(4), np.log10(grid_size//4), num=8, dtype=int)
        counts = [sum(1 for i in range(0, grid_size, size) 
                     for j in range(0, grid_size, size) 
                     if np.any(mask[i:i+size, j:j+size])) for size in box_sizes]
        log_size = np.log(1.0 / box_sizes)
        log_count = np.log(counts)
        valid = np.array(counts) > 0
        if np.sum(valid) >= 2:
            df = np.polyfit(log_size[valid], log_count[valid], 1)[0]
            dfs.append(df)
    return np.mean(dfs) if dfs else np.nan

def renormalize_points(points, factor=2):
    if len(points) < 10:
        return points
    min_x, max_x = points[:,0].min(), points[:,0].max()
    min_y, max_y = points[:,1].min(), points[:,1].max()
    spacing = np.std(points, axis=0).mean()
    nx = max(2, int((max_x - min_x) / (factor * spacing)))
    ny = max(2, int((max_y - min_y) / (factor * spacing)))
    xbins = np.linspace(min_x, max_x, nx+1)
    ybins = np.linspace(min_y, max_y, ny+1)
    new_points = []
    for i in range(nx):
        for j in range(ny):
            mask = ((points[:,0] >= xbins[i]) & (points[:,0] < xbins[i+1]) &
                    (points[:,1] >= ybins[j]) & (points[:,1] < ybins[j+1]))
            if np.any(mask):
                new_points.append(np.mean(points[mask], axis=0))
    return np.array(new_points) if new_points else points

# ============================================================
# 4. Main Simulation
# ============================================================
def simulate(points, adj, params, ablation=False, steps=6000, record_interval=30):
    N = len(points)
    phase = 2 * np.pi * np.random.rand(N)
    A = 0.5 + 0.1 * np.random.randn(N)
    A = np.clip(A, 0.01, 0.99)
    C = A * np.exp(1j * phase)

    mem = MemoryBuffer(params["tau"], params["dt"])
    pink = PinkNoise(N) if params.get("noise_type") == "pink" else None

    original_points = points.copy()
    history = {"time": [], "R": [], "ratio": [], "base_dist": [], "best_base": [],
               "drift": [], "fractal_dim": [], "dubito_frac": []}

    for step in range(steps):
        mem.update(C)
        C_pred = mem.predict()

        dphase = phase[:, None] - phase[None, :]
        sin_d = np.sin(dphase)
        cos_d = np.cos(dphase)

        if step % params["update_every"] == 0:
            w = np.exp(-params["beta"] * (1 - cos_d))
            W = csr_matrix(w * adj)
            W = (W + W.T) / 2
            W.setdiag(0)

        coupling = params["D"] * np.array((W.multiply(sin_d)).sum(axis=1)).flatten()

        dtheta_pred = np.zeros(N)
        dA_pred = np.zeros(N)
        if C_pred is not None and not ablation:
            dtheta_pred, dA_pred = prediction_gradient(C, C_pred, params["g_pred"])

        dA_self = self_reference_gradient(C, adj, params["g_self"]) if params["g_self"] != 0 else np.zeros(N)
        dA_negent = -negentropy_gradient(A)

        dtheta = coupling + dtheta_pred
        dA = dA_pred + dA_self + dA_negent

        noise = params["noise"] * (pink.next() if pink else np.random.randn(N))
        dtheta += noise
        if len(dA) == N:
            dA += noise

        A += params["dt"] * dA
        A = np.clip(A, 0.01, 0.99)
        phase += params["dt"] * dtheta
        phase %= 2 * np.pi
        C = A * np.exp(1j * phase)

        if step % 20 == 0:
            rows, cols = W.nonzero()
            force = np.zeros((N, 2))
            dx = points[cols] - points[rows]
            dist = np.linalg.norm(dx, axis=1) + 1e-8
            unit = dx / dist[:, None]
            w_edge = np.exp(-params["beta"] * (1 - np.cos(phase[cols] - phase[rows])))
            sin_edge = np.sin(phase[cols] - phase[rows])
            contrib = w_edge[:, None] * sin_edge[:, None] * unit
            np.add.at(force, rows, contrib)
            np.add.at(force, cols, -contrib)
            points += params["dt_back"] * force

        if step % record_interval == 0:
            R = np.abs(np.mean(np.exp(1j * phase)))
            ratio = spectral_ratio(W)
            base_dist, best_base = infer_base_continuous(ratio)
            drift = np.mean(np.linalg.norm(points - original_points, axis=1))
            Df = box_counting_field(points, A)

            total_f = np.abs(dtheta).mean() + np.abs(dA).mean()
            pred_f = np.abs(dtheta_pred).mean() + np.abs(dA_pred).mean()
            dub_frac = pred_f / (total_f + 1e-8)

            history["time"].append(step * params["dt"])
            history["R"].append(R)
            history["ratio"].append(ratio)
            history["base_dist"].append(base_dist)
            history["best_base"].append(best_base)
            history["drift"].append(drift)
            history["fractal_dim"].append(Df)
            history["dubito_frac"].append(dub_frac)

    return history

# ============================================================
# 5. Runner
# ============================================================
def run_blind_experiment(graph_types=["random", "lattice", "penrose"], n_runs=8):
    params = {
        "dt": 0.01, "D": 1.15, "g_pred": 0.26, "g_self": 0.04,
        "beta": 4.2, "tau": 30, "noise": 0.01, "dt_back": 0.002,
        "update_every": 50, "noise_type": "pink", "compute_fractal": True
    }

    results = {}
    for gtype in graph_types:
        results[gtype] = {"full": [], "ablation": []}
        for run in range(n_runs):
            np.random.seed(42 + run)
            if gtype == "random":
                points, adj = random_geometric_graph(200)
            elif gtype == "lattice":
                points, adj = lattice_graph(200)
            elif gtype == "penrose":
                points, adj = penrose_graph(200)
            else:
                continue

            hist_full = simulate(points.copy(), adj.copy(), params, ablation=False, steps=6000)
            results[gtype]["full"].append(hist_full)

            hist_abl = simulate(points.copy(), adj.copy(), params, ablation=True, steps=6000)
            results[gtype]["ablation"].append(hist_abl)

            print(f"{gtype} run {run+1}/{n_runs} completed")

    return results, params

# ============================================================
# 6. Statistical Decision Layer (final version)
# ============================================================
def statistical_decision_layer(results):
    print("\n=== STATISTICAL DECISION LAYER ===\n")
    phi = (1 + np.sqrt(5)) / 2

    for gtype in results:
        print(f"--- {gtype.upper()} ---")
        for mode in ["full", "ablation"]:
            runs = results[gtype][mode]

            steady_base_dist = [steady_mean(h["base_dist"]) for h in runs]
            steady_df = [steady_mean(h["fractal_dim"]) for h in runs]
            steady_dub = [steady_mean(h["dubito_frac"]) for h in runs]
            steady_best_b = [steady_mean(h["best_base"]) for h in runs if not np.isnan(steady_mean(h["best_base"]))]

            print(f"{mode.upper():<10} | Base-dist: {np.nanmean(steady_base_dist):.4f}±{np.nanstd(steady_base_dist):.4f}")
            print(f"           | D_f: {np.nanmean(steady_df):.3f} | Dubito frac: {np.nanmean(steady_dub):.3f}")
            print(f"           | Mean inferred base: {np.nanmean(steady_best_b):.4f} ± {np.nanstd(steady_best_b):.4f}")

            # φ advantage (paired)
            phi_dists = []
            alt_dists = []
            for h in runs:
                ratio = steady_mean(h["ratio"])
                if np.isnan(ratio) or ratio <= 0:
                    continue
                n_phi = np.log(ratio) / np.log(phi)
                d_phi = abs(n_phi - round(n_phi))
                best_alt = float("inf")
                for b in np.linspace(1.2, 3.0, 100):
                    if abs(b - phi) < 1e-5: continue
                    n = np.log(ratio) / np.log(b)
                    d = abs(n - round(n))
                    if d < best_alt:
                        best_alt = d
                phi_dists.append(d_phi)
                alt_dists.append(best_alt)

            if len(phi_dists) > 3:
                t, p = ttest_rel(phi_dists, alt_dists)
                mean_diff = np.mean(np.array(alt_dists) - np.array(phi_dists))
                print(f"           | φ advantage p-value: {p:.4f} (Δ = {mean_diff:.4f})")

            # Correlation dubito vs structure
            if len(steady_dub) > 3 and len(steady_base_dist) > 3:
                corr = np.corrcoef(steady_dub, steady_base_dist)[0,1]
                print(f"           | corr(dubito, base-dist): {corr:.3f}")

            # Delta drift / structure
            delta_base = np.nanmean(steady_base_dist) - np.nanmean([steady_mean(h["base_dist"]) for h in results[gtype]["ablation"]])
            print(f"           | Δ base-dist (full - ablation): {delta_base:.4f}")

        print()

def steady_mean(arr, frac=0.35):
    if len(arr) == 0:
        return np.nan
    n = len(arr)
    start = int((1 - frac) * n)
    return np.nanmean(arr[start:])

# ============================================================
# Ready to Run
# ============================================================
# results, params = run_blind_experiment(n_runs=6)
# statistical_decision_layer(results)