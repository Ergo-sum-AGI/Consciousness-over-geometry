# ============================================================
# CELL 1: SCALED-UP EXPERIMENT - N=1000, steps=12000
# HIGH STATISTICAL POWER
# WARNING: This will take 6 hours on colab
# ============================================================

print("="*70)
print("CELL 4: SCALED-UP EXPERIMENT")
print("N=1000 points, steps=12000")
print("="*70)

import numpy as np
import matplotlib
matplotlib.use("Agg")
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import warnings
import gc
import time
import pickle
warnings.filterwarnings("ignore")

PHI = (1 + np.sqrt(5)) / 2

# ============================================================
# SYMMETRY ENERGY FUNCTION (Optimized for large N)
# ============================================================
def symmetry_energy_fast(points):
    """Optimized symmetry energy for large point clouds"""
    if len(points) < 50:
        return 1.0, 0.0
    
    # Subsample for speed with large N
    n_sample = min(500, len(points))
    idx = np.random.choice(len(points), n_sample, replace=False)
    points_sub = points[idx]
    
    center = np.mean(points_sub, axis=0)
    centered = points_sub - center
    
    # Radial order
    radii = np.linalg.norm(centered, axis=1)
    hist, _ = np.histogram(radii, bins=20)
    hist = hist / (hist.sum() + 1e-8)
    radial_entropy = -np.sum(hist * np.log(hist + 1e-8))
    max_entropy = np.log(20)
    radial_order = 1.0 - (radial_entropy / max_entropy)
    
    # Nearest neighbor regularity (subsampled)
    tree = KDTree(points_sub)
    dists, _ = tree.query(points_sub, k=min(6, n_sample-1))
    nn_distances = dists[:, 1:].flatten()
    nn_std = np.std(nn_distances)
    nn_mean = np.mean(nn_distances)
    nn_regularity = 1.0 / (1.0 + (nn_std / (nn_mean + 1e-8)))
    
    order = (radial_order + nn_regularity) / 2.0
    energy = 1.0 - order
    return float(np.clip(energy, 0.0, 1.0)), float(np.clip(order, 0.0, 1.0))


def energy_gradient_fast(points, eps=0.005):
    """Optimized energy gradient for large N"""
    N = len(points)
    if N > 500:
        # Use random subset for gradient computation
        grad_idx = np.random.choice(N, min(200, N), replace=False)
    else:
        grad_idx = np.arange(N)
    
    gradient = np.zeros_like(points)
    
    for i in grad_idx:
        for dim in range(3):
            original = points[i, dim]
            
            points[i, dim] = original + eps
            energy_plus, _ = symmetry_energy_fast(points)
            
            points[i, dim] = original - eps
            energy_minus, _ = symmetry_energy_fast(points)
            
            points[i, dim] = original
            gradient[i, dim] = (energy_plus - energy_minus) / (2 * eps)
    
    return -gradient


def random_points_3d_large(N=1000):
    """Random points on sphere for large N"""
    pts = np.random.randn(N, 3).astype(np.float32)
    pts /= (np.linalg.norm(pts, axis=1)[:, None] + 1e-8)
    pts *= 2.0
    return pts


def build_adjacency_fast(points, k=8):
    """Fast adjacency building for large N"""
    N = len(points)
    k = min(k, N - 1)
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k+1)
    
    rows = np.repeat(np.arange(N), k)
    cols = indices[:, 1:].flatten()
    data = np.exp(-distances[:, 1:].flatten() / 0.3)
    
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W + W.T
    W.setdiag(0)
    W.eliminate_zeros()
    
    return W, rows, cols, data


def simulate_scaled(N=1000, steps=12000, beta=4.2, dt=0.01, 
                    dt_geometry=0.02, coupling_strength=0.3,
                    enable_evolution=True, start_from="random",
                    record_interval=500):
    """
    SCALED simulation for large N
    """
    print(f"\n  Starting SCALED simulation: N={N}, steps={steps}")
    print(f"  dt_geometry={dt_geometry}, coupling_strength={coupling_strength}")
    print(f"  evolution={enable_evolution}")
    
    # Initialize points
    if start_from == "random":
        points = random_points_3d_large(N)
    else:
        points = random_points_3d_large(N)
    
    N_actual = len(points)
    print(f"  Actual points: {N_actual}")
    
    # Initialize state
    np.random.seed(42)
    phase = 2 * np.pi * np.random.rand(N_actual).astype(np.float32)
    A = np.ones(N_actual, dtype=np.float32)
    original_points = points.copy()
    
    history = {'R': [], 'order': [], 'energy': [], 'dub': [], 'drift': [], 'step': []}
    
    # Build adjacency
    W, rows, cols, data = build_adjacency_fast(points, k=6)
    print(f"  Initial edges: {W.nnz}")
    
    # Initial order
    initial_energy, initial_order = symmetry_energy_fast(points)
    print(f"  Initial order: {initial_order:.4f}, energy: {initial_energy:.4f}")
    
    start_time = time.time()
    
    # Simulation loop
    for step in range(steps):
        try:
            # Phase dynamics
            if len(rows) > 0:
                delta = phase[rows] - phase[cols]
                coupling = np.zeros(N_actual)
                np.add.at(coupling, rows, data * np.sin(delta))
                phase += dt * (beta * coupling + 0.02 * np.random.randn(N_actual))
            else:
                phase += dt * 0.02 * np.random.randn(N_actual)
            
            phase %= 2 * np.pi
            A += 0.002 * np.random.randn(N_actual)
            A = np.clip(A, 0.1, 1.5)
            
            # Geometry evolution
            if enable_evolution and len(rows) > 0:
                # Phase force
                phase_force = np.zeros_like(points)
                for i in range(len(rows)):
                    r, c = rows[i], cols[i]
                    phase_force[r] += data[i] * np.sin(delta[i]) * (points[c] - points[r])
                
                # Normalize phase force
                phase_norm = np.linalg.norm(phase_force, axis=1, keepdims=True)
                phase_force = phase_force / (phase_norm + 1e-8)
                
                # Energy gradient force
                energy_grad = energy_gradient_fast(points)
                grad_norm = np.linalg.norm(energy_grad, axis=1, keepdims=True)
                energy_grad = energy_grad / (grad_norm + 1e-8)
                
                # Combined force
                total_force = (coupling_strength * phase_force + 
                              (1 - coupling_strength) * energy_grad)
                
                # Move points
                points += dt_geometry * total_force * (A[:, None] * 0.3)
                
                # Bound drift
                max_drift = 2.0
                drift_norm = np.linalg.norm(points - original_points, axis=1)
                if np.any(drift_norm > max_drift):
                    scale = max_drift / (drift_norm.max() + 1e-8)
                    points = original_points + (points - original_points) * scale
                
                # Rebuild adjacency periodically
                if step % 1000 == 0 and step > 0:
                    W, rows, cols, data = build_adjacency_fast(points, k=6)
            
            # Dubito
            if len(rows) > 0:
                phase_pred = phase + dt * coupling
                dub = float(np.mean(np.abs(phase_pred - phase)))
            else:
                dub = 0.05
            
            # Recording
            if step % record_interval == 0:
                R = float(np.abs(np.mean(np.exp(1j * phase))))
                drift = float(np.mean(np.linalg.norm(points - original_points, axis=1)))
                energy_val, order_val = symmetry_energy_fast(points)
                
                history['R'].append(R)
                history['order'].append(order_val)
                history['energy'].append(energy_val)
                history['dub'].append(dub)
                history['drift'].append(drift)
                history['step'].append(step)
                
                elapsed = time.time() - start_time
                print(f"    step {step}: R={R:.3f}, order={order_val:.4f}, "
                      f"energy={energy_val:.4f}, drift={drift:.4f} [{elapsed:.1f}s]")
            
            # Periodic cleanup
            if step % 2000 == 0 and step > 0:
                gc.collect()
                
        except Exception as e:
            print(f"  Warning at step {step}: {e}")
            continue
    
    total_time = time.time() - start_time
    final_energy, final_order = symmetry_energy_fast(points)
    
    print(f"\n  Final order: {final_order:.4f} (was {initial_order:.4f})")
    print(f"  Change: {final_order - initial_order:+.4f}")
    print(f"  Total time: {total_time:.1f}s")
    
    return points, history


# ============================================================
# RUN SCALED SIMULATION
# ============================================================
print("\n🚀 Running SCALED simulation...")
print("   N=1000 points, steps=12000")
print("   ⚠️ This will take 30-60 minutes")
print("")

points_evolved, history_evolved = simulate_scaled(
    N=1000,
    steps=12000,
    beta=4.2,
    dt=0.01,
    dt_geometry=0.02,
    coupling_strength=0.3,
    enable_evolution=True,
    start_from="random",
    record_interval=1000
)

# Save results
with open('history_evolved_scaled.pkl', 'wb') as f:
    pickle.dump(history_evolved, f)

print(f"\n{'='*50}")
print(f"✅ SCALED simulation complete")
if history_evolved['order']:
    print(f"   Initial order: {history_evolved['order'][0]:.4f}")
    print(f"   Final order:   {history_evolved['order'][-1]:.4f}")
    print(f"   Change:        {history_evolved['order'][-1] - history_evolved['order'][0]:+.4f}")
    print(f"   Final R:       {history_evolved['R'][-1]:.4f}")
    print(f"   Final drift:   {history_evolved['drift'][-1]:.4f}")
print(f"{'='*50}")

# Clean up
gc.collect()
print("\n✅ Cell 4 complete - Scaled experiment done!")

# ===============================================================
# CELL 2: SCALED ABLATION - NO geometry evolution
# matching scaled parameters
# ===============================================================

print("="*70)
print("CELL 5: SCALED ABLATION - NO COUPLING")
print("N=1000 points, steps=6000")
print("="*70)

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import warnings
import gc
import time
import pickle
warnings.filterwarnings("ignore")

# Reuse functions from Cell 4
def symmetry_energy_fast(points):
    if len(points) < 50:
        return 1.0, 0.0
    
    n_sample = min(500, len(points))
    idx = np.random.choice(len(points), n_sample, replace=False)
    points_sub = points[idx]
    
    center = np.mean(points_sub, axis=0)
    centered = points_sub - center
    
    radii = np.linalg.norm(centered, axis=1)
    hist, _ = np.histogram(radii, bins=20)
    hist = hist / (hist.sum() + 1e-8)
    radial_entropy = -np.sum(hist * np.log(hist + 1e-8))
    max_entropy = np.log(20)
    radial_order = 1.0 - (radial_entropy / max_entropy)
    
    tree = KDTree(points_sub)
    dists, _ = tree.query(points_sub, k=min(6, n_sample-1))
    nn_distances = dists[:, 1:].flatten()
    nn_std = np.std(nn_distances)
    nn_mean = np.mean(nn_distances)
    nn_regularity = 1.0 / (1.0 + (nn_std / (nn_mean + 1e-8)))
    
    order = (radial_order + nn_regularity) / 2.0
    energy = 1.0 - order
    return float(np.clip(energy, 0.0, 1.0)), float(np.clip(order, 0.0, 1.0))

def random_points_3d_large(N=1000):
    pts = np.random.randn(N, 3).astype(np.float32)
    pts /= (np.linalg.norm(pts, axis=1)[:, None] + 1e-8)
    pts *= 2.0
    return pts

def build_adjacency_fast(points, k=8):
    N = len(points)
    k = min(k, N - 1)
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k+1)
    rows = np.repeat(np.arange(N), k)
    cols = indices[:, 1:].flatten()
    data = np.exp(-distances[:, 1:].flatten() / 0.3)
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W + W.T
    W.setdiag(0)
    W.eliminate_zeros()
    return W, rows, cols, data

def simulate_ablation_scaled(N=1000, steps=6000, beta=4.2, dt=0.01, record_interval=500):
    """Ablation: NO geometry evolution"""
    print(f"\n  Starting SCALED ABLATION: N={N}, steps={steps}")
    
    np.random.seed(42)
    points = random_points_3d_large(N)
    original_points = points.copy()
    N_actual = len(points)
    
    phase = 2 * np.pi * np.random.rand(N_actual).astype(np.float32)
    A = np.ones(N_actual, dtype=np.float32)
    
    history = {'R': [], 'order': [], 'energy': [], 'dub': [], 'drift': [], 'step': []}
    
    W, rows, cols, data = build_adjacency_fast(points, k=6)
    print(f"  Initial edges: {W.nnz}")
    
    initial_energy, initial_order = symmetry_energy_fast(points)
    print(f"  Initial order: {initial_order:.4f}")
    
    start_time = time.time()
    
    for step in range(steps):
        if len(rows) > 0:
            delta = phase[rows] - phase[cols]
            coupling = np.zeros(N_actual)
            np.add.at(coupling, rows, data * np.sin(delta))
            phase += dt * (beta * coupling + 0.02 * np.random.randn(N_actual))
        else:
            phase += dt * 0.02 * np.random.randn(N_actual)
        
        phase %= 2 * np.pi
        A += 0.002 * np.random.randn(N_actual)
        A = np.clip(A, 0.1, 1.5)
        
        # CRITICAL: NO geometry evolution!
        
        if step % record_interval == 0:
            R = float(np.abs(np.mean(np.exp(1j * phase))))
            drift = float(np.mean(np.linalg.norm(points - original_points, axis=1)))
            energy_val, order_val = symmetry_energy_fast(points)
            
            if len(rows) > 0:
                phase_pred = phase + dt * coupling
                dub = float(np.mean(np.abs(phase_pred - phase)))
            else:
                dub = 0.05
            
            history['R'].append(R)
            history['order'].append(order_val)
            history['energy'].append(energy_val)
            history['dub'].append(dub)
            history['drift'].append(drift)
            history['step'].append(step)
            
            elapsed = time.time() - start_time
            print(f"    step {step}: R={R:.3f}, order={order_val:.4f}, drift={drift:.4f} [{elapsed:.1f}s]")
        
        if step % 2000 == 0 and step > 0:
            gc.collect()
    
    total_time = time.time() - start_time
    final_energy, final_order = symmetry_energy_fast(points)
    
    print(f"\n  Final order: {final_order:.4f} (was {initial_order:.4f})")
    print(f"  Change: {final_order - initial_order:+.4f}")
    print(f"  Total time: {total_time:.1f}s")
    
    return points, history

print("\n🚀 Running SCALED ABLATION simulation...")
points_frozen, history_frozen = simulate_ablation_scaled(
    N=1000,
    steps=6000,
    beta=4.2,
    dt=0.01,
    record_interval=500
)

with open('history_frozen_scaled.pkl', 'wb') as f:
    pickle.dump(history_frozen, f)

print(f"\n✅ SCALED ABLATION complete")
if history_frozen['order']:
    print(f"   Initial order: {history_frozen['order'][0]:.4f}")
    print(f"   Final order:   {history_frozen['order'][-1]:.4f}")
    print(f"   Change:        {history_frozen['order'][-1] - history_frozen['order'][0]:+.4f}")

gc.collect()
print("\n✅ Cell 5 complete")

# ===============================================================
# CELL 3: SCALED ANALYSIS
# Run this last
# ===============================================================

print("="*70)
print("CELL 6: SCALED ANALYSIS")
print("="*70)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import os

# Load data
history_evolved = None
history_frozen = None

if os.path.exists('history_evolved_scaled.pkl'):
    with open('history_evolved_scaled.pkl', 'rb') as f:
        history_evolved = pickle.load(f)
    print("✅ Loaded 'history_evolved_scaled.pkl'")
else:
    print("❌ 'history_evolved_scaled.pkl' not found")

if os.path.exists('history_frozen_scaled.pkl'):
    with open('history_frozen_scaled.pkl', 'rb') as f:
        history_frozen = pickle.load(f)
    print("✅ Loaded 'history_frozen_scaled.pkl'")
else:
    print("❌ 'history_frozen_scaled.pkl' not found")

print("\n" + "="*70)
print("RESULTS")
print("="*70)

if history_evolved and history_evolved['order']:
    print(f"\n📊 SCALED WITH COUPLING (Evolved):")
    print(f"   Initial order: {history_evolved['order'][0]:.6f}")
    print(f"   Final order:   {history_evolved['order'][-1]:.6f}")
    print(f"   Change:        {history_evolved['order'][-1] - history_evolved['order'][0]:+.6f}")
    print(f"   Final R:       {history_evolved['R'][-1]:.6f}")
    print(f"   Final drift:   {history_evolved['drift'][-1]:.6f}")

if history_frozen and history_frozen['order']:
    print(f"\n📊 SCALED WITHOUT COUPLING (Frozen):")
    print(f"   Initial order: {history_frozen['order'][0]:.6f}")
    print(f"   Final order:   {history_frozen['order'][-1]:.6f}")
    print(f"   Change:        {history_frozen['order'][-1] - history_frozen['order'][0]:+.6f}")
    print(f"   Final R:       {history_frozen['R'][-1]:.6f}")
    print(f"   Final drift:   {history_frozen['drift'][-1]:.6f}")

print("\n" + "="*70)
print("FALSIFIABILITY CRITERION - SCALED")
print("="*70)

if history_evolved and history_frozen:
    evolved_change = history_evolved['order'][-1] - history_evolved['order'][0]
    frozen_change = history_frozen['order'][-1] - history_frozen['order'][0]
    
    if evolved_change > 0.05 and abs(frozen_change) < 0.01:
        print("\n✅ SCALED PASS: Field→geometry coupling significantly improves order")
        print(f"   Evolved improvement: {evolved_change:+.6f}")
        print(f"   Frozen change:       {frozen_change:+.6f}")
        print("   → Architecture suitable for Volatco (CONFIRMED at scale)")
    elif evolved_change > 0.02:
        print("\n📈 SCALED PARTIAL: Improvement detected")
        print(f"   Evolved improvement: {evolved_change:+.6f}")
        print("   → Increase steps or coupling strength")
    else:
        print("\n⚠️ SCALED: Improvement smaller than expected")
        print(f"   Evolved improvement: {evolved_change:+.6f}")

# Visualization
print("\n📈 Generating scaled visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if history_evolved and history_evolved['step']:
    axes[0,0].plot(history_evolved['step'], history_evolved['order'], 
                   'g-', linewidth=2, label='With coupling (evolved)')
if history_frozen and history_frozen['step']:
    axes[0,0].plot(history_frozen['step'], history_frozen['order'], 
                   'r--', linewidth=2, label='Without coupling (frozen)')
axes[0,0].set_xlabel('Step')
axes[0,0].set_ylabel('Structural Order')
axes[0,0].set_title('SCALED: Structural Order Evolution')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

if history_evolved and history_evolved['step']:
    axes[0,1].plot(history_evolved['step'], history_evolved['R'], 
                   'g-', linewidth=2, label='With coupling')
if history_frozen and history_frozen['step']:
    axes[0,1].plot(history_frozen['step'], history_frozen['R'], 
                   'r--', linewidth=2, label='Without coupling')
axes[0,1].set_xlabel('Step')
axes[0,1].set_ylabel('Phase Coherence R')
axes[0,1].set_title('SCALED: Phase Coherence')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

if history_evolved and history_evolved['step']:
    axes[1,0].plot(history_evolved['step'], history_evolved['drift'], 
                   'b-', linewidth=2, label='With coupling')
if history_frozen and history_frozen['step']:
    axes[1,0].plot(history_frozen['step'], history_frozen['drift'], 
                   'r--', linewidth=2, label='Without coupling')
axes[1,0].set_xlabel('Step')
axes[1,0].set_ylabel('Geometric Drift')
axes[1,0].set_title('SCALED: Point Movement')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

if history_evolved and history_evolved['step']:
    axes[1,1].plot(history_evolved['step'], history_evolved['dub'], 
                   'b-', linewidth=2, label='Dubito')
axes[1,1].set_xlabel('Step')
axes[1,1].set_ylabel('Dubito (Prediction Error)')
axes[1,1].set_title('SCALED: Prediction Error')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("volatco_scaled_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Saved plot → volatco_scaled_analysis.png")
print("\n" + "="*70)
print("SCALED EXPERIMENT COMPLETE")
print("="*70)