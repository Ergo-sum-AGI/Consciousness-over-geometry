import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.linalg import eigs
from collections import deque
import time
import warnings
from scipy.stats import ttest_ind
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. Penrose points (vectorized)
# ============================================================
def generate_penrose_points(N_points=300):
    """
    Generate points approximating a Penrose tiling using the cut-and-project method.
    Returns an array of shape (N_points, 2).
    """
    angles = 2 * np.pi * np.arange(5) / 5
    phys_vecs = np.column_stack((np.cos(angles), np.sin(angles)))
    perp_vecs = np.column_stack((np.cos(2*angles), np.sin(2*angles)))

    max_int = int(np.sqrt(N_points) * 1.8)
    coords = np.arange(-max_int, max_int + 1)
    grid = np.meshgrid(*([coords] * 5), indexing='ij')
    points_5d = np.stack(grid, axis=-1).reshape(-1, 5)
    points_5d = points_5d[np.linalg.norm(points_5d, axis=1) < max_int * 1.2]

    phys = points_5d @ phys_vecs
    perp = points_5d @ perp_vecs
    mask = np.linalg.norm(perp, axis=1) < 1.4
    phys = phys[mask]

    if len(phys) > N_points:
        idx = np.random.choice(len(phys), N_points, replace=False)
        phys = phys[idx]
    return phys

# ============================================================
# 2. Build adjacency (sparse, symmetric)
# ============================================================
def build_adjacency(points, k=6):
    """
    Build a sparse symmetric adjacency matrix using k‑nearest neighbors.
    Returns a csr_matrix.
    """
    tree = KDTree(points)
    n = len(points)
    rows, cols = [], []
    for i in range(n):
        _, idx = tree.query(points[i], k=k+1)
        for j in idx[1:]:
            rows.append(i)
            cols.append(j)

    rows_arr = np.array(rows)
    cols_arr = np.array(cols)
    # Symmetrize by adding reverse edges
    rows = np.concatenate([rows_arr, cols_arr])
    cols = np.concatenate([cols_arr, rows_arr])

    data = np.ones(len(rows))
    adj = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj

# ============================================================
# 3. Memory buffer (exponential kernel)
# ============================================================
class MemoryBuffer:
    def __init__(self, tau, dt, maxlen=800):
        self.tau = tau
        self.dt = dt
        self.buffer = deque(maxlen=maxlen)
        self.weights = np.exp(-np.arange(maxlen) * dt / tau)
        # Avoid underflow: if all weights are zero, set uniform
        if self.weights.sum() == 0:
            self.weights = np.ones(maxlen) / maxlen
        else:
            self.weights /= self.weights.sum()

    def update(self, C):
        self.buffer.append(C.copy())

    def get_prediction(self):
        if not self.buffer:
            return None
        n = len(self.buffer)
        w = self.weights[:n]
        w /= w.sum()
        past = np.array(self.buffer)  # shape (n, N)
        C_pred = np.sum(past * w[:, None], axis=0)
        return C_pred

# ============================================================
# 4. Pink noise (Voss‑McCartney)
# ============================================================
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
        noise = np.sum(self.octave_values, axis=0) / np.sqrt(self.n_octaves)
        self.counter = (self.counter + 1) % self.max_counter
        return noise

# ============================================================
# 5. Free energy gradients
# ============================================================
def negentropy_gradient(A):
    """Derivative of the negentropy term with respect to amplitude."""
    A2 = A**2
    safe = np.clip(A2, 1e-8, 1 - 1e-8)
    return 2 * A * (np.log(safe) - np.log(1 - safe))

def prediction_gradient(C, C_pred, g_pred):
    """Gradients of the prediction error term w.r.t. phase and amplitude."""
    dtheta = -g_pred * np.imag(np.conj(C) * C_pred)
    dA = -g_pred * np.real(np.conj(C) * (C - C_pred) / (np.abs(C) + 1e-8))
    return dtheta, dA

def self_reference_gradient(C, adj, g_self):
    """Amplitude gradient from the self‑reference term (non‑local coupling)."""
    A = np.abs(C)
    A2 = A**2
    contrib = g_self * (adj @ A2)
    return A * contrib

# ============================================================
# 6. Diagnostics
# ============================================================
def spectral_entropy(W):
    """Compute entropy of Laplacian eigenvalue distribution."""
    deg = np.asarray(W.sum(axis=1)).flatten()
    L = diags(deg, 0) - W
    try:
        # Use sparse eigs for large matrices, fallback to dense for small
        if L.shape[0] > 200:
            vals = eigs(L, k=min(30, L.shape[0]-2), which='SM', return_eigenvectors=False)
        else:
            vals = np.linalg.eigvalsh(L.toarray())
        eigvals = np.real(vals)
        eigvals = eigvals[eigvals > 1e-8]
        if len(eigvals) == 0:
            return np.nan
        p = eigvals / eigvals.sum()
        return -np.sum(p * np.log(p + 1e-12))
    except Exception:
        return np.nan

def phi_distance(ratio):
    """Return signed distance from ratio to the nearest integer power of φ."""
    if ratio <= 0 or np.isnan(ratio):
        return np.nan
    phi = (1 + np.sqrt(5)) / 2
    n = np.log(ratio) / np.log(phi)
    return n - round(n)

def box_counting_field(points, A, grid_size=192, threshold_factor=0.5):
    """
    Box‑counting dimension of the amplitude field.
    Uses an adaptive threshold = threshold_factor * max(A).
    """
    x_min, x_max = points[:,0].min(), points[:,0].max()
    y_min, y_max = points[:,1].min(), points[:,1].max()
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    tree = KDTree(points)
    _, idx = tree.query(grid_points)
    A_grid = A[idx].reshape(grid_size, grid_size)
    threshold = threshold_factor * A_grid.max()
    mask = A_grid > threshold

    box_sizes = np.logspace(np.log10(4), np.log10(grid_size//4), num=10, dtype=int)
    counts = []
    for size in box_sizes:
        n_boxes = 0
        for i in range(0, grid_size, size):
            for j in range(0, grid_size, size):
                if np.any(mask[i:i+size, j:j+size]):
                    n_boxes += 1
        counts.append(n_boxes)
    log_size = np.log(1.0 / box_sizes)
    log_count = np.log(counts)
    valid = np.array(counts) > 0
    if np.sum(valid) < 2:
        return np.nan
    return np.polyfit(log_size[valid], log_count[valid], 1)[0]

def renormalize_points(points, factor=2):
    """Coarse‑grain points by averaging in cells of size factor * typical spacing."""
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
# 7. Main simulation (sparse movement + all fixes)
# ============================================================
def simulate_unified(
    N_points=250,
    steps=8000,
    dt=0.01,
    D=1.15,
    g_pred=0.26,
    g_self=0.04,
    tau_memory=30,
    noise_amp=0.01,
    noise_type='pink',
    beta_plastic=4.2,
    update_weights_every=50,
    move_points_every=10,
    dt_back=0.002,
    k_neighbors=6,
    use_sparse_eigs=True,
    record_interval=20,
    use_amplitude=True,
    compute_fractal=True,
    ensemble_id=0,
    save_results=False
):
    """
    Run the consciousness field simulation.
    Returns:
        points, original_points, W, history, snapshots, C, struct_factor, rg_ratios
    """
    points = generate_penrose_points(N_points)
    original_points = points.copy()
    n = len(points)
    adj_binary = build_adjacency(points, k=k_neighbors)

    if use_amplitude:
        A = 0.5 + 0.1 * np.random.randn(n)
        A = np.clip(A, 0.01, 0.99)
        phase = 2 * np.pi * np.random.rand(n)
        C = A * np.exp(1j * phase)
    else:
        phase = 2 * np.pi * np.random.rand(n)
        C = np.exp(1j * phase)

    W = adj_binary.copy().astype(float)
    mem = MemoryBuffer(tau=tau_memory, dt=dt)

    if noise_type == 'pink':
        pink_amp = PinkNoise(n_channels=n)
        pink_phase = PinkNoise(n_channels=n)
    else:
        pink_amp = pink_phase = None

    history = {
        'time': [], 'R': [], 'local_order': [], 'ratio': [], 'drift': [],
        'mean_A': [], 'spectral_entropy': [], 'phi_distance': [],
        'fractal_dim': [], 'dubito_fraction': []
    }
    snapshots = []

    def compute_ratio(W):
        """Compute eigenvalue ratio of the Laplacian of W (sparse or dense)."""
        deg = np.asarray(W.sum(axis=1)).flatten()
        L = diags(deg, 0, format='csr') - W
        try:
            if W.shape[0] > 200 and use_sparse_eigs:
                vals = eigs(L, k=2, which='SM', return_eigenvectors=False)
                vals = np.sort(np.real(vals))
                if vals[0] > 1e-6:
                    return vals[1] / vals[0]
                else:
                    return np.nan
            else:
                # Dense fallback
                L_dense = L.toarray()
                eigvals = np.linalg.eigvalsh(L_dense)
                eigvals = eigvals[eigvals > 1e-6]
                if len(eigvals) >= 2:
                    return eigvals[1] / eigvals[0]
                else:
                    return np.nan
        except Exception:
            return np.nan

    # Pre-allocate large arrays (can be memory heavy, but okay for N<=500)
    delta = np.zeros((n, n))
    sin_diff = np.zeros((n, n))
    cos_diff = np.zeros((n, n))

    t_start = time.time()
    for step in range(steps):
        mem.update(C)

        # Phase differences
        delta[:,:] = phase[:, None] - phase[None, :]
        sin_diff[:,:] = np.sin(delta)
        cos_diff[:,:] = np.cos(delta)

        # Update plastic weights
        if step % update_weights_every == 0:
            w_ij = np.exp(-beta_plastic * (1 - cos_diff))
            w_ij = w_ij * adj_binary.toarray()
            W = csr_matrix(w_ij)
            W = (W + W.T) / 2
            W.setdiag(0)
            W.eliminate_zeros()

        # Coupling term (diffusion)
        coupling_theta = D * (W @ sin_diff)

        # Prediction term
        C_pred = mem.get_prediction()
        dtheta_pred = np.zeros(n)
        dA_pred = np.zeros(n)
        if C_pred is not None:
            dtheta_pred, dA_pred = prediction_gradient(C, C_pred, g_pred)

        # Self‑reference and negentropy
        dA_self = self_reference_gradient(C, adj_binary, g_self) if g_self != 0 else np.zeros(n)
        dA_negent = -negentropy_gradient(A) if use_amplitude else np.zeros(n)

        # Total derivatives
        dtheta = coupling_theta + dtheta_pred
        dA = dA_pred + dA_self + dA_negent

        # Noise
        if noise_type == 'pink':
            noise_theta = noise_amp * pink_phase.next()
            noise_A = noise_amp * pink_amp.next() if use_amplitude else 0
        else:
            noise_theta = noise_amp * np.random.randn(n)
            noise_A = noise_amp * np.random.randn(n) if use_amplitude else 0
        dtheta += noise_theta
        dA += noise_A

        # Euler step
        if use_amplitude:
            A += dt * dA
            A = np.clip(A, 0.01, 0.99)
            phase += dt * dtheta
            phase %= 2 * np.pi
            C = A * np.exp(1j * phase)
        else:
            phase += dt * dtheta
            phase %= 2 * np.pi
            C = np.exp(1j * phase)

        # Sparse‑friendly point movement
        if move_points_every > 0 and step % move_points_every == 0 and step > 0:
            rows, cols = W.nonzero()
            force = np.zeros((n, 2))
            dx = points[cols] - points[rows]
            dist = np.linalg.norm(dx, axis=1) + 1e-8
            unit_dx = dx / dist[:, None]
            # Use W.data directly (already the non-zero weights in the same order as rows, cols)
            weights = W.data
            sin_vals = sin_diff[rows, cols]
            force_contrib = weights[:, None] * sin_vals[:, None] * unit_dx
            np.add.at(force, rows, force_contrib)
            points += dt_back * force
            # Rebuild adjacency for the new positions
            adj_binary = build_adjacency(points, k=k_neighbors)

        # Record diagnostics
        if step % record_interval == 0:
            R = np.abs(np.sum(C)) / n
            local_order = (W.multiply(cos_diff).sum() / W.sum()) if W.sum() > 0 else 0
            history['time'].append(step * dt)
            history['R'].append(R)
            history['local_order'].append(local_order)
            if use_amplitude:
                history['mean_A'].append(np.mean(A))

            if step % (record_interval * 10) == 0:
                ratio = compute_ratio(W)
                dist_phi = phi_distance(ratio)
                entropy = spectral_entropy(W)

                history['ratio'].append(ratio)
                history['phi_distance'].append(dist_phi)
                history['spectral_entropy'].append(entropy)

                # Fractal dimension (field-based)
                Df = np.nan
                if compute_fractal and use_amplitude:
                    Df = box_counting_field(points, A)
                history['fractal_dim'].append(Df)

                # Dubito fraction
                total_force = np.abs(dtheta).mean() + np.abs(dA).mean()
                pred_force = np.abs(dtheta_pred).mean() + np.abs(dA_pred).mean()
                dub_frac = pred_force / (total_force + 1e-8)
                history['dubito_fraction'].append(dub_frac)

                print(f"Step {step}: R={R:.3f}, ratio={ratio if not np.isnan(ratio) else '--'}, "
                      f"drift={np.mean(np.linalg.norm(points - original_points, axis=1)):.5f}, "
                      f"dub_frac={dub_frac:.3f}, D_f={Df if not np.isnan(Df) else '--':.3f}")
            else:
                # Append NaNs for diagnostics not recorded at this step
                for key in ['ratio', 'phi_distance', 'spectral_entropy', 'fractal_dim', 'dubito_fraction']:
                    history[key].append(np.nan)

            drift = np.mean(np.linalg.norm(points - original_points, axis=1))
            history['drift'].append(drift)

            if step % (record_interval * 10) == 0:
                if use_amplitude:
                    snapshots.append((step, (A.copy(), phase.copy())))
                else:
                    snapshots.append((step, phase.copy()))

    print(f"Simulation finished in {time.time() - t_start:.2f} seconds")

    # Renormalization flow (coarse‑graining)
    rg_ratios = []
    rg_points = points.copy()
    for _ in range(2):
        rg_points = renormalize_points(rg_points, factor=2)
        if len(rg_points) < 10:
            break
        k_rg = min(4, len(rg_points) - 1)
        adj_rg = build_adjacency(rg_points, k=k_rg)
        W_rg = adj_rg.astype(float)
        ratio_rg = compute_ratio(W_rg)
        rg_ratios.append(ratio_rg)

    if save_results:
        np.savez(f"simulation_ens{ensemble_id}.npz", history=history)

    return points, original_points, W, history, snapshots, C, None, rg_ratios

# ============================================================
# 8. Ensemble runner
# ============================================================
def run_ensemble(num_runs=8, base_params=None, include_ablation=True):
    """
    Run multiple simulations with different seeds.
    If include_ablation=True, also runs with g_pred=0.
    Returns (full_results, ablation_results) or (full_results,) if no ablation.
    """
    if base_params is None:
        base_params = {
            'N_points': 250,
            'steps': 8000,
            'D': 1.15,
            'g_pred': 0.26,
            'g_self': 0.04,
            'tau_memory': 30,
            'noise_amp': 0.01,
            'noise_type': 'pink',
            'beta_plastic': 4.2,
            'update_weights_every': 50,
            'move_points_every': 10,
            'dt_back': 0.002,
            'k_neighbors': 6,
            'use_sparse_eigs': True,
            'record_interval': 20,
            'use_amplitude': True,
            'compute_fractal': True,
            'save_results': True
        }

    full_results = []
    for i in range(num_runs):
        print(f"\n=== Full model run {i+1}/{num_runs} ===")
        np.random.seed(42 + i)
        res = simulate_unified(**base_params, ensemble_id=i)
        full_results.append(res)

    if include_ablation:
        ablation_params = base_params.copy()
        ablation_params['g_pred'] = 0.0
        ablation_results = []
        for i in range(num_runs):
            print(f"\n=== Ablation (no dubito) run {i+1}/{num_runs} ===")
            np.random.seed(42 + i)
            res = simulate_unified(**ablation_params, ensemble_id=i+100)
            ablation_results.append(res)
        return full_results, ablation_results
    else:
        return full_results

# ============================================================
# 9. Analysis and plotting
# ============================================================
def analyze_ensemble(full_results, ablation_results=None):
    """
    Extract metrics, print statistics, and plot histograms.
    """
    def extract_metrics(results):
        final_ratios = []
        final_drifts = []
        final_fractal = []
        dubito_frac = []
        phi_dists = []
        for _, _, _, hist, _, _, _, _ in results:
            # Last non‑NaN ratio
            ratios = np.array(hist['ratio'])
            ratios = ratios[~np.isnan(ratios)]
            if len(ratios) > 0:
                final_ratios.append(ratios[-1])
            # Last drift
            final_drifts.append(hist['drift'][-1])
            # Last fractal dimension (non‑NaN)
            fractal = np.array(hist['fractal_dim'])
            fractal = fractal[~np.isnan(fractal)]
            if len(fractal) > 0:
                final_fractal.append(fractal[-1])
            # Dubito fraction (mean of last 5 recorded)
            dub = np.array(hist['dubito_fraction'])
            dub = dub[~np.isnan(dub)]
            if len(dub) > 0:
                dubito_frac.append(np.mean(dub[-5:]))
            # All phi distances
            dists = np.array(hist['phi_distance'])
            phi_dists.extend(dists[~np.isnan(dists)])
        return {
            'ratio': final_ratios,
            'drift': final_drifts,
            'fractal': final_fractal,
            'dubito_frac': dubito_frac,
            'phi_distances': phi_dists
        }

    full = extract_metrics(full_results)
    print("\n=== FULL MODEL STATISTICS ===")
    print(f"Final eigenvalue ratio: mean = {np.nanmean(full['ratio']):.4f} ± {np.nanstd(full['ratio']):.4f}")
    print(f"Final drift: mean = {np.nanmean(full['drift']):.5f} ± {np.nanstd(full['drift']):.5f}")
    print(f"Final fractal dimension: mean = {np.nanmean(full['fractal']):.4f} ± {np.nanstd(full['fractal']):.4f}")
    print(f"Mean dubito fraction (last 5): {np.nanmean(full['dubito_frac']):.3f}")
    print(f"Phi distance mean: {np.nanmean(full['phi_distances']):.4f} ± {np.nanstd(full['phi_distances']):.4f}")

    if ablation_results:
        abl = extract_metrics(ablation_results)
        print("\n=== ABLATION (no dubito) STATISTICS ===")
        print(f"Final eigenvalue ratio: mean = {np.nanmean(abl['ratio']):.4f} ± {np.nanstd(abl['ratio']):.4f}")
        print(f"Final drift: mean = {np.nanmean(abl['drift']):.5f} ± {np.nanstd(abl['drift']):.5f}")
        print(f"Final fractal dimension: mean = {np.nanmean(abl['fractal']):.4f} ± {np.nanstd(abl['fractal']):.4f}")
        print(f"Mean dubito fraction (last 5): {np.nanmean(abl['dubito_frac']):.3f}")
        print(f"Phi distance mean: {np.nanmean(abl['phi_distances']):.4f} ± {np.nanstd(abl['phi_distances']):.4f}")

        # Statistical tests
        t_drift, p_drift = ttest_ind(full['drift'], abl['drift'], nan_policy='omit')
        t_fractal, p_fractal = ttest_ind(full['fractal'], abl['fractal'], nan_policy='omit')
        t_phi, p_phi = ttest_ind(full['phi_distances'], abl['phi_distances'], nan_policy='omit')
        print(f"\nStatistical differences (t‑test):")
        print(f"Drift: p = {p_drift:.4f}")
        print(f"Fractal dimension: p = {p_fractal:.4f}")
        print(f"Phi distance clustering: p = {p_phi:.4f}")

        # Plot histograms
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0,0].hist(full['ratio'], bins=15, alpha=0.7, label='Full')
        axs[0,0].hist(abl['ratio'], bins=15, alpha=0.7, label='Ablation')
        axs[0,0].axvline((1+np.sqrt(5))/2, color='k', linestyle='--', label='φ')
        axs[0,0].axvline(((1+np.sqrt(5))/2)**2, color='gray', linestyle='--', label='φ²')
        axs[0,0].set_xlabel('Eigenvalue ratio')
        axs[0,0].legend()
        axs[0,0].set_title('Laplacian ratio distribution')

        axs[0,1].hist(full['drift'], bins=15, alpha=0.7, label='Full')
        axs[0,1].hist(abl['drift'], bins=15, alpha=0.7, label='Ablation')
        axs[0,1].set_xlabel('Geometry drift')
        axs[0,1].legend()
        axs[0,1].set_title('Point displacement')

        axs[0,2].hist(full['fractal'], bins=15, alpha=0.7, label='Full')
        axs[0,2].hist(abl['fractal'], bins=15, alpha=0.7, label='Ablation')
        axs[0,2].axvline(2.382, color='r', linestyle='--', label='Predicted 2.382')
        axs[0,2].set_xlabel('Fractal dimension')
        axs[0,2].legend()
        axs[0,2].set_title('Field fractal dimension')

        axs[1,0].hist(full['phi_distances'], bins=50, alpha=0.7, label='Full', density=True)
        axs[1,0].hist(abl['phi_distances'], bins=50, alpha=0.7, label='Ablation', density=True)
        axs[1,0].set_xlabel('Distance to nearest φⁿ')
        axs[1,0].set_ylabel('Density')
        axs[1,0].legend()
        axs[1,0].set_title('φ‑family distance distribution')

        # Additional: dubito fraction vs drift scatter (optional)
        axs[1,1].scatter(full['dubito_frac'], full['drift'], alpha=0.6, label='Full')
        axs[1,1].scatter(abl['dubito_frac'], abl['drift'], alpha=0.6, label='Ablation')
        axs[1,1].set_xlabel('Dubito fraction')
        axs[1,1].set_ylabel('Geometry drift')
        axs[1,1].legend()
        axs[1,1].set_title('Dubito effect on drift')

        # Renormalization flow placeholder (if needed)
        axs[1,2].text(0.5, 0.5, 'Renormalization flow\n(see rg_ratios in output)', ha='center', va='center')
        axs[1,2].set_title('Coarse‑graining (optional)')

        plt.tight_layout()
        plt.show()
    else:
        # Just plot full model distributions
        plt.figure(figsize=(12, 4))
        plt.subplot(1,3,1)
        plt.hist(full['phi_distances'], bins=50)
        plt.xlabel('Distance to nearest φⁿ')
        plt.title('φ‑family distance')
        plt.subplot(1,3,2)
        plt.hist(full['fractal'], bins=15)
        plt.axvline(2.382, color='r', linestyle='--', label='Predicted')
        plt.xlabel('Fractal dimension')
        plt.legend()
        plt.subplot(1,3,3)
        plt.hist(full['drift'], bins=15)
        plt.xlabel('Geometry drift')
        plt.title('Point displacement')
        plt.tight_layout()
        plt.show()

# ============================================================
# 10. Example usage (single test + ensemble)
# ============================================================
if __name__ == "__main__":
    # Single test run
    print("Running single simulation for verification...")
    points, orig, W, hist, snaps, C, _, rg = simulate_unified(
        N_points=150, steps=1000, save_results=False
    )
    print("Test done. Ready for ensemble.")

    # Uncomment to run ensemble (may take hours)
    # full, ablation = run_ensemble(num_runs=4, include_ablation=True)
    # analyze_ensemble(full, ablation)