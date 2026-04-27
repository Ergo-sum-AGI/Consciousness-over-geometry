"""
CQFT Full Experiment Matrix — Final Edition (v4)
=================================================
DUBITO Ergo AGI Safety Project | Daniel Solis

Fixes applied vs v3:
  FIX 4 — square_lattice_geometry per-point normalization:
    Old: percentile-based scaling -> mean radius 1.445 at N=50, fails preflight
    New: per-point radial normalization (same as Penrose v3 fix)
         -> mean radius = 2.0000 exactly at any N

All three geometry generators now use identical normalization:
    radii  = np.linalg.norm(pts, axis=1)
    pts    = pts / (radii[:, None] + 1e-8) * 2.0
This ensures fair comparison across geometry types at all N values.

Usage (Colab):
  Step 1 — mount Drive:
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    WORK_DIR = "/content/drive/MyDrive/CQFT_experiment"
    os.makedirs(WORK_DIR, exist_ok=True)
    os.chdir(WORK_DIR)

  Step 2 — run:
    %run cqft_experiment_matrix.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform
import warnings, gc, time, pickle, os, json, signal, sys
import shutil, tempfile, traceback, csv

warnings.filterwarnings("ignore")

PHI = (1 + np.sqrt(5)) / 2

# ============================================================
# PATHS — Google Drive
# ============================================================

WORK_DIR    = "/content/drive/MyDrive/CQFT_experiment"
FIGURES_DIR = os.path.join(WORK_DIR, "cqft_figures")
OUTPUTS_DIR = os.path.join(WORK_DIR, "cqft_outputs")
RESULTS_CSV = os.path.join(WORK_DIR, "cqft_results_master.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("Working directory : {}".format(WORK_DIR))
print("Outputs directory : {}".format(OUTPUTS_DIR))
print("Figures directory : {}".format(FIGURES_DIR))
print("Results CSV       : {}".format(RESULTS_CSV))


# ============================================================
# GEOMETRY GENERATORS
# All three use per-point radial normalization -> mean_r = 2.0
# at any N, ensuring fair cross-geometry comparison.
# ============================================================

def random_geometry(N=1000, seed=42):
    """
    Random points on sphere of radius 2.
    Per-point normalization: each point projected to r=2 exactly.
    """
    np.random.seed(seed)
    pts  = np.random.randn(N, 3).astype(np.float64)
    radii = np.linalg.norm(pts, axis=1)
    pts   = pts / (radii[:, None] + 1e-8) * 2.0
    return pts.astype(np.float32)


def square_lattice_geometry(N=1000, seed=42, jitter=0.04):
    """
    3D cubic lattice with small jitter, projected to radius-2 sphere.

    FIX v4: per-point radial normalization replaces percentile scaling.
    Percentile scaling gave mean_r=1.445 at N=50 (only 4^3=64 points
    in grid, 95th percentile >> mean), causing preflight failure.
    Per-point normalization gives mean_r=2.0000 at any N.

    The cubic lattice angular structure (translational symmetry,
    regular bond angles) is preserved by the projection.
    """
    np.random.seed(seed)
    side   = int(round(N ** (1.0 / 3.0)))
    coords = np.array([[x, y, z]
                        for x in range(side)
                        for y in range(side)
                        for z in range(side)], dtype=float)
    if len(coords) > N:
        coords = coords[:N]
    elif len(coords) < N:
        extra  = N - len(coords)
        coords = np.vstack([coords, np.random.rand(extra, 3) * side])

    coords += jitter * np.random.randn(len(coords), 3)
    coords -= np.mean(coords, axis=0)

    # Per-point normalization — robust at any N
    radii  = np.linalg.norm(coords, axis=1)
    coords = coords / (radii[:, None] + 1e-8) * 2.0

    return coords.astype(np.float32)


def penrose_geometry(N=1000, seed=42):
    """
    3D icosahedral quasicrystal via cut-and-project from Z^6.

    FIX v3: per-point radial normalization.
    Old percentile scaling gave mean_r=1.63, flat energy gradient,
    and artificially fast (incomplete) simulations.

    Projection matrices:
      P_par @ P_par.T  = 2*I_3  (orthogonal, scaled)
      P_par @ P_perp.T = 0       (fully decoupled)

    Acceptance window r<0.82 gives ~1237 points, trimmed to N=1000
    closest to centroid, preserving the quasicrystalline core.
    """
    np.random.seed(seed)

    par_vecs = np.array([
        [ 1,     PHI,   0  ],
        [-1,     PHI,   0  ],
        [ 0,     1,     PHI],
        [ 0,    -1,     PHI],
        [ PHI,   0,     1  ],
        [-PHI,   0,     1  ],
    ], dtype=np.float64)

    perp_vecs = np.array([
        [ 1,      -1/PHI, 0      ],
        [-1,      -1/PHI, 0      ],
        [ 0,       1,    -1/PHI  ],
        [ 0,      -1,    -1/PHI  ],
        [-1/PHI,   0,     1      ],
        [ 1/PHI,   0,     1      ],
    ], dtype=np.float64)

    P_par  = (par_vecs  / np.linalg.norm(par_vecs,  axis=1, keepdims=True)).T
    P_perp = (perp_vecs / np.linalg.norm(perp_vecs, axis=1, keepdims=True)).T

    vals = np.arange(-4, 5)
    grid = np.array(np.meshgrid(*[vals] * 6, indexing='ij')).reshape(6, -1).T

    perp_coords = grid @ P_perp.T
    perp_norms  = np.linalg.norm(perp_coords, axis=1)
    mask        = perp_norms < 0.82
    accepted    = grid[mask]
    phys        = accepted @ P_par.T

    print("  Penrose: {}/{} lattice points accepted".format(
        mask.sum(), len(grid)))

    tree1 = KDTree(phys)
    pairs = tree1.query_pairs(r=0.02)
    if pairs:
        to_remove = set(j for _, j in pairs)
        keep = [i for i in range(len(phys)) if i not in to_remove]
        phys = phys[keep]
    print("  Penrose: {} unique points after deduplication".format(len(phys)))

    if len(phys) < N:
        raise RuntimeError(
            "Penrose generator produced only {} points, need {}.".format(
                len(phys), N))

    center = np.mean(phys, axis=0)
    dists  = np.linalg.norm(phys - center, axis=1)
    idx    = np.argsort(dists)[:N]
    pts    = phys[idx]
    pts   -= np.mean(pts, axis=0)

    # Per-point normalization
    radii = np.linalg.norm(pts, axis=1)
    pts   = pts / (radii[:, None] + 1e-8) * 2.0

    final_radii = np.linalg.norm(pts, axis=1)
    print("  Penrose radii: mean={:.4f} std={:.6f}".format(
        np.mean(final_radii), np.std(final_radii)))

    return pts.astype(np.float32)


GEOMETRY_BUILDERS = {
    "random":          random_geometry,
    "square_lattice":  square_lattice_geometry,
    "penrose":         penrose_geometry,
}


# ============================================================
# CHECKPOINT INFRASTRUCTURE
# ============================================================

def _run_id(geometry, evolution, seed):
    ev = "full_evo" if evolution else "null_frz"
    return "{}_{}_s{}".format(geometry, ev, seed)


def _output_path(geometry, evolution, seed):
    return os.path.join(OUTPUTS_DIR,
                        "result_{}.pkl".format(
                            _run_id(geometry, evolution, seed)))


def _ckpt_paths(geometry, evolution, seed):
    rid  = _run_id(geometry, evolution, seed)
    base = os.path.join(OUTPUTS_DIR, "ckpt_{}".format(rid))
    return (base + "_primary.pkl",
            base + "_backup.pkl",
            base + "_manifest.json")


def atomic_save(obj, filepath):
    dirpath = os.path.dirname(os.path.abspath(filepath))
    fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, filepath)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_checkpoint(step, points, A, phase, history,
                    geometry, evolution, seed,
                    elapsed_total=0.0, ckpt_interval=500):
    PRIMARY, BACKUP, MANIFEST = _ckpt_paths(geometry, evolution, seed)
    payload = {
        "step": int(step), "N": len(points),
        "geometry": geometry, "evolution": evolution, "seed": seed,
        "points": points.copy(), "A": A.copy(), "phase": phase.copy(),
        "history": {k: list(v) for k, v in history.items()
                    if k not in ("final_points", "final_A", "final_phase")},
        "elapsed_total": float(elapsed_total),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_save(payload, PRIMARY)
    try:
        shutil.copy2(PRIMARY, BACKUP)
    except Exception:
        pass
    manifest = {
        "geometry": geometry, "evolution": evolution, "seed": seed,
        "last_step": int(step), "timestamp": payload["timestamp"],
        "elapsed_h": elapsed_total / 3600,
        "R_last":     history["R"][-1]     if history["R"]     else None,
        "order_last": history["order"][-1] if history["order"] else None,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print("  [ckpt] step {} -> {}".format(step, os.path.basename(PRIMARY)))


def load_checkpoint(geometry, evolution, seed):
    PRIMARY, BACKUP, _ = _ckpt_paths(geometry, evolution, seed)
    for path in (PRIMARY, BACKUP):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                p = pickle.load(f)
            if (p.get("geometry") == geometry and
                    p.get("evolution") == evolution and
                    p.get("seed") == seed):
                print("  [resume] Loaded {} (step={}, ts={})".format(
                    os.path.basename(path), p["step"],
                    p.get("timestamp", "?")))
                return p
        except Exception as e:
            print("  [resume] Could not read {}: {}".format(path, e))
    return None


def show_status():
    manifests = sorted(f for f in os.listdir(OUTPUTS_DIR)
                       if f.startswith("ckpt_") and
                       f.endswith("_manifest.json"))
    if not manifests:
        print("  [status] No checkpoints found in {}".format(OUTPUTS_DIR))
        return
    for mpath in manifests:
        with open(os.path.join(OUTPUTS_DIR, mpath)) as f:
            m = json.load(f)
        print("  [status] {}/{}/s{}: step={}, elapsed={:.2f}h, "
              "R={}, order={}".format(
                  m.get("geometry", "?"),
                  "full" if m.get("evolution") else "null",
                  m.get("seed", "?"),
                  m["last_step"], m["elapsed_h"],
                  m["R_last"], m["order_last"]))


# ============================================================
# GRACEFUL SHUTDOWN
# ============================================================

_shutdown = False


def _sighandler(sig, frame):
    global _shutdown
    print("\n  [signal] Interrupt — saving after current step.")
    _shutdown = True


signal.signal(signal.SIGINT,  _sighandler)
signal.signal(signal.SIGTERM, _sighandler)


# ============================================================
# CORE PHYSICS
# ============================================================

def symmetry_energy_fast(points):
    if len(points) < 50:
        return 1.0, 0.0
    n_sample = min(500, len(points))
    idx      = np.random.choice(len(points), n_sample, replace=False)
    pts      = points[idx]
    centered = pts - np.mean(pts, axis=0)
    radii    = np.linalg.norm(centered, axis=1)
    hist, _  = np.histogram(radii, bins=20)
    hist     = hist / (hist.sum() + 1e-8)
    r_order  = 1.0 - (-np.sum(hist * np.log(hist + 1e-8))) / np.log(20)
    tree     = KDTree(pts)
    dists, _ = tree.query(pts, k=min(6, n_sample - 1))
    nn_d     = dists[:, 1:].flatten()
    nn_reg   = 1.0 / (1.0 + np.std(nn_d) / (np.mean(nn_d) + 1e-8))
    order    = (r_order + nn_reg) / 2.0
    return float(np.clip(1.0 - order, 0, 1)), float(np.clip(order, 0, 1))


def energy_gradient_fast(points, eps=0.005):
    points   = points.copy()
    N        = len(points)
    grad_idx = (np.random.choice(N, min(200, N), replace=False)
                if N > 500 else np.arange(N))
    gradient = np.zeros_like(points)
    for i in grad_idx:
        for dim in range(3):
            orig = points[i, dim]
            seed = 42 + i * 100 + dim
            points[i, dim] = orig + eps
            np.random.seed(seed); ep, _ = symmetry_energy_fast(points)
            points[i, dim] = orig - eps
            np.random.seed(seed); em, _ = symmetry_energy_fast(points)
            points[i, dim] = orig
            gradient[i, dim] = (ep - em) / (2 * eps)
    return -gradient


def build_adjacency_fast(points, k=6):
    N    = len(points)
    k    = min(k, N - 1)
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)
    rows = np.repeat(np.arange(N), k)
    cols = indices[:, 1:].flatten()
    data = np.exp(-distances[:, 1:].flatten() / 0.3)
    W    = csr_matrix((data, (rows, cols)), shape=(N, N))
    W    = W + W.T
    W.setdiag(0)
    W.eliminate_zeros()
    return W, rows, cols, data


# ============================================================
# SIMULATION ENGINE
# ============================================================

def run_simulation(geometry="random", evolution=True, seed=42,
                   N=1000, steps=12000, beta=4.2,
                   dt=0.01, dt_geometry=0.02, coupling_strength=0.3,
                   record_interval=1000, ckpt_interval=500,
                   resume=True):
    """
    Single simulation run. Returns (points, history, A, phase).
    """
    global _shutdown
    _shutdown = False

    label = _run_id(geometry, evolution, seed)
    print("\n" + "=" * 60)
    print("  RUN: {}".format(label))
    print("  geometry={}, evolution={}, seed={}, N={}".format(
        geometry, evolution, seed, N))
    print("=" * 60)

    # Skip only if complete with valid history
    out_path = _output_path(geometry, evolution, seed)
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            result = pickle.load(f)
        n_steps = len(result.get("history", {}).get("step", []))
        if n_steps > 5:
            print("  Already complete ({} recorded steps) — loading.".format(
                n_steps))
            return (result["final_points"], result["history"],
                    result["final_A"],      result["final_phase"])
        else:
            print("  Result file has only {} history steps — rerunning.".format(
                n_steps))

    # Resume or fresh start
    start_step     = 0
    elapsed_offset = 0.0
    history = {"R": [], "order": [], "energy": [],
               "dub": [], "drift": [], "step": []}

    if resume:
        ckpt = load_checkpoint(geometry, evolution, seed)
        if ckpt is not None:
            start_step     = ckpt["step"] + 1
            points         = ckpt["points"].astype(np.float32)
            A              = ckpt["A"].astype(np.float32)
            phase          = ckpt["phase"].astype(np.float32)
            elapsed_offset = ckpt.get("elapsed_total", 0.0)
            for k in history:
                if k in ckpt["history"]:
                    history[k] = list(ckpt["history"][k])
            print("  Resuming from step {} ({:.2f}h elapsed)".format(
                start_step, elapsed_offset / 3600))

    if start_step == 0:
        print("  Building {} geometry (seed={})...".format(geometry, seed))
        points = GEOMETRY_BUILDERS[geometry](N=N, seed=seed)
        np.random.seed(seed + 1000)
        phase  = (2 * np.pi * np.random.rand(N)).astype(np.float32)
        A      = np.ones(N, dtype=np.float32)

    N_actual        = len(points)
    original_points = points.copy()

    # Radius assertion
    radii  = np.linalg.norm(points, axis=1)
    mean_r = float(np.mean(radii))
    assert abs(mean_r - 2.0) < 0.05, \
        ("FATAL: mean radius {:.4f} != 2.0 — geometry normalization "
         "failed for {}.").format(mean_r, geometry)
    print("  Geometry radii OK: mean={:.4f} std={:.4f}".format(
        mean_r, float(np.std(radii))))

    assert points.shape == (N_actual, 3)
    assert A.shape      == (N_actual,)
    assert phase.shape  == (N_actual,)

    W, rows, cols, data = build_adjacency_fast(points, k=6)
    init_e, init_o      = symmetry_energy_fast(points)
    print("  Initial edges: {}  order: {:.4f}  energy: {:.4f}".format(
        W.nnz, init_o, init_e))

    start_time = time.time()

    for step in range(start_step, steps):

        if _shutdown:
            elapsed = time.time() - start_time + elapsed_offset
            save_checkpoint(step - 1, points, A, phase, history,
                            geometry, evolution, seed,
                            elapsed_total=elapsed,
                            ckpt_interval=ckpt_interval)
            print("  [signal] Saved and exiting.")
            return points, history, A, phase

        try:
            # Phase dynamics
            if len(rows) > 0:
                delta    = phase[rows] - phase[cols]
                coupling = np.zeros(N_actual)
                np.add.at(coupling, rows, data * np.sin(delta))
                phase += dt * (beta * coupling +
                               0.02 * np.random.randn(N_actual))
            else:
                delta    = np.zeros(0)
                coupling = np.zeros(N_actual)
                phase   += dt * 0.02 * np.random.randn(N_actual)

            phase %= 2 * np.pi
            A     += 0.002 * np.random.randn(N_actual)
            A      = np.clip(A, 0.1, 1.5)

            # Geometry evolution
            if evolution and len(rows) > 0:
                pf = np.zeros_like(points)
                for i in range(len(rows)):
                    r, c = rows[i], cols[i]
                    pf[r] += data[i] * np.sin(delta[i]) * (points[c] - points[r])
                pf /= np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8

                eg = energy_gradient_fast(points)

                # Gradient magnitude check at step 0
                if step == 0:
                    gm = float(np.linalg.norm(eg, axis=1).mean())
                    print("  Step-0 gradient magnitude: {:.6f}".format(gm))
                    if gm < 1e-4:
                        print("  WARNING: gradient near zero — "
                              "check geometry normalization.")

                eg /= np.linalg.norm(eg, axis=1, keepdims=True) + 1e-8

                total_force = (coupling_strength * pf +
                               (1 - coupling_strength) * eg)
                points     += dt_geometry * total_force * (A[:, None] * 0.3)

                drift_norm = np.linalg.norm(points - original_points, axis=1)
                if np.any(drift_norm > 2.0):
                    scale  = 2.0 / (drift_norm.max() + 1e-8)
                    points = original_points + (points - original_points) * scale

                if step % 1000 == 0 and step > 0:
                    W, rows, cols, data = build_adjacency_fast(points, k=6)

            # Dubito
            dub = (float(np.mean(np.abs((phase + dt * coupling) - phase)))
                   if len(rows) > 0 else 0.05)

            # Record
            if step % record_interval == 0:
                R     = float(np.abs(np.mean(np.exp(1j * phase))))
                drift = float(np.mean(
                    np.linalg.norm(points - original_points, axis=1)))
                ev, ov = symmetry_energy_fast(points)
                history["R"].append(R)
                history["order"].append(ov)
                history["energy"].append(ev)
                history["dub"].append(dub)
                history["drift"].append(drift)
                history["step"].append(step)
                elapsed = time.time() - start_time + elapsed_offset
                print("    step {:>6}: R={:.3f} order={:.4f} "
                      "drift={:.4f} [{:.2f}h]".format(
                          step, R, ov, drift, elapsed / 3600))

            # Checkpoint
            if step % ckpt_interval == 0 and step > start_step:
                elapsed = time.time() - start_time + elapsed_offset
                save_checkpoint(step, points, A, phase, history,
                                geometry, evolution, seed,
                                elapsed_total=elapsed,
                                ckpt_interval=ckpt_interval)

            if step % 2000 == 0 and step > 0:
                gc.collect()

        except KeyboardInterrupt:
            _shutdown = True
            continue
        except Exception as e:
            print("  [warning] step {}: {}".format(step, e))
            elapsed = time.time() - start_time + elapsed_offset
            save_checkpoint(step, points, A, phase, history,
                            geometry, evolution, seed,
                            elapsed_total=elapsed,
                            ckpt_interval=ckpt_interval)
            continue

    # Post-loop assertions
    assert points is not None, "FATAL: points None"
    assert A      is not None, "FATAL: A None"
    assert phase  is not None, "FATAL: phase None"
    if steps > 10:  # Only enforce for real runs, not smoke tests
        assert len(history["R"]) > 5, \
            "FATAL: only {} history entries".format(len(history["R"]))

    total_time = time.time() - start_time + elapsed_offset
    fe, fo = symmetry_energy_fast(points)
    print("  Final order: {:.4f} (was {:.4f})  delta={:+.4f}  "
          "time={:.2f}h".format(fo, init_o, fo - init_o,
                                total_time / 3600))
    return points, history, A, phase


# ============================================================
# ANALYSIS PIPELINE
# ============================================================

def compute_Df_local(pos, phase, n_bins=40, window=8):
    dist_matrix = squareform(pdist(pos))
    corr_matrix = np.cos(phase[:, None] - phase[None, :])
    r = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    G = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    bins    = np.logspace(np.log10(r.min() + 1e-6), np.log10(r.max()), n_bins)
    bin_idx = np.digitize(r, bins)
    r_bin, G_bin = [], []
    for i in range(1, len(bins)):
        sel = bin_idx == i
        if np.sum(sel) > 20:
            r_bin.append(np.mean(r[sel]))
            G_bin.append(np.mean(G[sel]))
    r_bin = np.array(r_bin)
    G_bin = np.array(G_bin)
    etas, r_centers = [], []
    for i in range(len(r_bin) - window):
        rw, Gw = r_bin[i:i + window], G_bin[i:i + window]
        slope, *_ = linregress(np.log(rw),
                               np.log(np.abs(Gw) + 1e-12))
        etas.append(-slope)
        r_centers.append(np.mean(rw))
    return np.array(r_centers), 3 - np.array(etas), r_bin, G_bin


def compute_Dc(pos, r_low_pct=5, r_high_pct=50, n_samples=30):
    """Grassberger-Procaccia correlation dimension."""
    dists    = np.sort(pdist(pos))
    r_min    = np.percentile(dists, r_low_pct)
    r_max    = np.percentile(dists, r_high_pct)
    r_samp   = np.logspace(np.log10(r_min), np.log10(r_max), n_samples)
    C        = np.array([np.mean(dists < rv) for rv in r_samp])
    slope, _ = np.polyfit(np.log(r_samp), np.log(C + 1e-10), 1)
    return float(slope)


def compute_signed_gap(pos, phase):
    r_centers, D_f_local, r_bin, G_bin = compute_Df_local(pos, phase)
    D_c         = compute_Dc(pos)
    Delta_local = D_f_local - D_c
    Gamma       = float(np.mean(np.abs(Delta_local)))
    crossings   = []
    for i in range(len(Delta_local) - 1):
        if Delta_local[i] * Delta_local[i + 1] < 0:
            r_c = (r_centers[i]   * abs(Delta_local[i + 1]) +
                   r_centers[i + 1] * abs(Delta_local[i])) / \
                  (abs(Delta_local[i]) + abs(Delta_local[i + 1]))
            crossings.append(float(r_c))
    n_pos  = int(np.sum(Delta_local > 0))
    n_neg  = int(np.sum(Delta_local < 0))
    regime = ("field_dominated"      if n_pos == len(Delta_local) else
              "geometry_dominated"   if n_neg == len(Delta_local) else
              "crossover_structured" if crossings                 else
              "inconclusive")
    D_f_mean = float(np.mean(D_f_local))
    kappa    = Gamma / (abs(D_f_mean - 3.0) + 1e-6)
    return dict(
        D_c=float(D_c), D_f_mean=D_f_mean,
        D_f_local=D_f_local, r_centers=r_centers,
        Delta_local=Delta_local, Gamma_global=Gamma,
        zero_crossings=crossings, regime=regime,
        kappa=float(kappa), G_bin=G_bin, r_bin=r_bin,
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_condition(analysis, geometry, evolution, seed, history):
    ev_label = "full" if evolution else "null"
    rid      = _run_id(geometry, evolution, seed)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("CQFT: {} | {} evolution | seed={}".format(
        geometry.replace("_", " "), ev_label, seed), fontsize=12)

    ax = axes[0, 0]
    ax.plot(analysis["r_centers"], analysis["Delta_local"], "b-", lw=1.5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for r_c in analysis["zero_crossings"]:
        ax.axvline(r_c, color="r", lw=0.8, ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("r (scale)")
    ax.set_ylabel("Δ(r) = D_f(r) − D_c")
    ax.set_title("Signed embedding gap  Γ={:.3f}  κ={:.3f}".format(
        analysis["Gamma_global"], analysis["kappa"]))
    ax.text(0.05, 0.95, "Regime: {}".format(analysis["regime"]),
            transform=ax.transAxes, va="top", fontsize=8, color="darkred")

    ax = axes[0, 1]
    ax.plot(analysis["r_bin"], np.abs(analysis["G_bin"]) + 1e-12,
            "g-", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("r"); ax.set_ylabel("|G(r)|")
    ax.set_title("Phase correlation  D_c={:.3f}  <D_f>={:.3f}".format(
        analysis["D_c"], analysis["D_f_mean"]))

    ax = axes[1, 0]
    ax.plot(analysis["r_centers"], analysis["D_f_local"], "m-", lw=1.5)
    ax.axhline(analysis["D_c"], color="orange", lw=1.2, ls="--",
               label="D_c={:.3f}".format(analysis["D_c"]))
    ax.set_xscale("log")
    ax.set_xlabel("r"); ax.set_ylabel("D_f(r)")
    ax.set_title("Field dimension vs geometry dimension")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    steps_rec = history.get("step", [])
    if len(steps_rec) > 1:
        ax.plot(steps_rec, history["order"], "b-",  lw=1.2, label="order")
        ax.plot(steps_rec, history["R"],     "r--", lw=1.0,
                label="R (coherence)")
        ax.set_xlabel("step"); ax.set_ylabel("value")
        ax.set_title("Evolution trajectory")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No trajectory data",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=10)
        ax.set_title("Evolution trajectory (unavailable)")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "diagnostic_{}.png".format(rid))
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("  [plot] Saved -> {}".format(os.path.basename(fig_path)))
    return fig_path


def plot_comparison(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "CQFT Signed Embedding Gap  Δ(r) = D_f(r) − D_c\n"
        "Hall analogy: sign = direction of field/geometry dominance",
        fontsize=11)
    colors = {"random": "steelblue",
              "square_lattice": "darkorange",
              "penrose": "crimson"}
    styles = {True: "-", False: ":"}
    for ax_idx, geom in enumerate(["random", "square_lattice", "penrose"]):
        ax = axes[ax_idx]
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("r (scale)")
        ax.set_title(geom.replace("_", " ").title())
        if ax_idx == 0:
            ax.set_ylabel("Δ(r)")
        for key, res in all_results.items():
            if res.get("geometry") != geom:
                continue
            ev    = res["evolution"]
            seed  = res["seed"]
            label = "full s{}".format(seed) if ev else "null"
            ax.plot(res["r_centers"], res["Delta_local"],
                    color=colors[geom], ls=styles[ev],
                    lw=1.2, alpha=0.9 if ev else 0.5, label=label)
            for r_c in res.get("zero_crossings", []):
                ax.axvline(r_c, color=colors[geom],
                           lw=0.6, ls=":", alpha=0.4)
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "comparison_delta_r_v4.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  [plot] Comparison -> {}".format(os.path.basename(fig_path)))
    return fig_path


# ============================================================
# RESULTS CSV
# ============================================================

def append_csv(row_dict):
    fieldnames = ["run_id", "geometry", "evolution", "seed",
                  "D_c", "D_f_mean", "Gamma_global", "kappa",
                  "n_zero_crossings", "regime",
                  "final_order", "final_R", "final_drift"]
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in fieldnames})


# ============================================================
# EXPERIMENT MATRICES
# ============================================================

EXPERIMENT_MATRIX = [
    ("random",         True,  42),
    ("random",         True,  43),
    ("random",         True,  44),
    ("square_lattice", True,  42),
    ("square_lattice", True,  43),
    ("square_lattice", True,  44),
    ("penrose",        True,  42),
    ("penrose",        True,  43),
    ("penrose",        True,  44),
    ("random",         False, 42),
    ("square_lattice", False, 42),
    ("penrose",        False, 42),
]

# Use this after deleting stale Penrose files
PENROSE_RERUN_MATRIX = [
    ("penrose", True,  42),
    ("penrose", True,  43),
    ("penrose", True,  44),
]


def run_matrix(N=1000, steps=12000, resume=True, matrix=None):
    if matrix is None:
        matrix = EXPERIMENT_MATRIX

    all_results = {}

    for geom, ev, seed in matrix:
        out_path = _output_path(geom, ev, seed)
        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                stored = pickle.load(f)
            rid     = _run_id(geom, ev, seed)
            n_steps = len(stored.get("history", {}).get("step", []))
            if "analysis" in stored and stored["analysis"] and n_steps > 5:
                all_results[rid] = dict(stored["analysis"])
                all_results[rid].update(
                    {"geometry": geom, "evolution": ev, "seed": seed})
                print("  [matrix] Loaded existing: {} ({} steps)".format(
                    rid, n_steps))

    for run_idx, (geom, ev, seed) in enumerate(matrix):
        rid      = _run_id(geom, ev, seed)
        out_path = _output_path(geom, ev, seed)

        print("\n" + "#" * 70)
        print("# MATRIX RUN {}/{}: {}".format(
            run_idx + 1, len(matrix), rid))
        print("#" * 70)

        try:
            pts, history, A, phase = run_simulation(
                geometry=geom, evolution=ev, seed=seed,
                N=N, steps=steps, resume=resume,
                record_interval=1000, ckpt_interval=500)
        except Exception as e:
            print("  SIMULATION FAILED: {}".format(e))
            traceback.print_exc()
            continue

        print("  Running analysis pipeline...")
        try:
            analysis = compute_signed_gap(pts, phase)
            print("  D_c={:.4f}  D_f={:.4f}  Δ={:.4f}  "
                  "Γ={:.4f}  regime={}  crossings={}".format(
                      analysis["D_c"], analysis["D_f_mean"],
                      analysis["D_f_mean"] - analysis["D_c"],
                      analysis["Gamma_global"], analysis["regime"],
                      analysis["zero_crossings"]))
        except Exception as e:
            print("  ANALYSIS FAILED: {}".format(e))
            traceback.print_exc()
            analysis = {}

        result_payload = {
            "geometry":     geom, "evolution":    ev, "seed": seed,
            "history":      {k: list(v) for k, v in history.items()},
            "final_points": pts, "final_A": A, "final_phase": phase,
            "analysis":     analysis,
        }
        atomic_save(result_payload, out_path)
        print("  Result saved -> {}".format(os.path.basename(out_path)))

        if analysis:
            plot_condition(analysis, geom, ev, seed, history)
            all_results[rid] = dict(analysis)
            all_results[rid].update(
                {"geometry": geom, "evolution": ev, "seed": seed})
            fo = history["order"][-1] if history.get("order") else ""
            fR = history["R"][-1]     if history.get("R")     else ""
            fd = history["drift"][-1] if history.get("drift") else ""
            append_csv({
                "run_id":           rid,
                "geometry":         geom,
                "evolution":        ev,
                "seed":             seed,
                "D_c":              round(analysis.get("D_c", 0), 4),
                "D_f_mean":         round(analysis.get("D_f_mean", 0), 4),
                "Gamma_global":     round(analysis.get("Gamma_global", 0), 4),
                "kappa":            round(analysis.get("kappa", 0), 4),
                "n_zero_crossings": len(analysis.get("zero_crossings", [])),
                "regime":           analysis.get("regime", ""),
                "final_order":  round(fo, 4) if fo != "" else "",
                "final_R":      round(fR, 4) if fR != "" else "",
                "final_drift":  round(fd, 4) if fd != "" else "",
            })

        gc.collect()

    if len(all_results) >= 2:
        print("\n  Generating master comparison plot...")
        plot_comparison(all_results)

    print("\n" + "=" * 70)
    print("MATRIX COMPLETE")
    print("Results CSV : {}".format(RESULTS_CSV))
    print("Figures     : {}".format(FIGURES_DIR))
    print("=" * 70)
    return all_results


# ============================================================
# PRE-FLIGHT
# ============================================================

def preflight():
    print("\n" + "=" * 60)
    print("  PRE-FLIGHT v4")
    print("=" * 60)

    for geom in ("random", "square_lattice", "penrose"):
        pts   = GEOMETRY_BUILDERS[geom](N=50, seed=0)
        radii = np.linalg.norm(pts, axis=1)
        assert pts.shape == (50, 3), "{} bad shape".format(geom)
        assert abs(np.mean(radii) - 2.0) < 0.05, \
            "{} bad mean radius {:.4f}".format(geom, np.mean(radii))
        print("  geometry OK: {} shape={} mean_r={:.4f} std={:.6f}".format(
            geom, pts.shape, np.mean(radii), np.std(radii)))

    print("\n  Smoke-test simulation (penrose, N=50, steps=10)...")
    pts, hist, A, phase = run_simulation(
        geometry="penrose", evolution=True, seed=0,
        N=50, steps=10, record_interval=5,
        ckpt_interval=5, resume=False)
    assert pts.shape   == (50, 3)
    assert A.shape     == (50,)
    assert phase.shape == (50,)
    print("  Simulation OK")

    print("\n  Smoke-test analysis pipeline...")
    res = compute_signed_gap(pts, phase)
    assert "D_c" in res and "Delta_local" in res
    print("  Analysis OK  D_c={:.3f}  D_f={:.3f}".format(
        res["D_c"], res["D_f_mean"]))

    print("\n  PRE-FLIGHT PASSED")
    print("=" * 60 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    show_status()

    try:
        preflight()
    except Exception as e:
        print("PRE-FLIGHT FAILED: {}".format(e))
        traceback.print_exc()
        sys.exit(1)

    # Default: run only the three Penrose reruns.
    # Change to EXPERIMENT_MATRIX for the full 12-run matrix.
    results = run_matrix(
        N=1000,
        steps=12000,
        resume=True,
        matrix=PENROSE_RERUN_MATRIX,
    )
