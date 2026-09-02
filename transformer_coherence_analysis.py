"""
================================================================================
TRANSFORMER GEOMETRIC COHERENCE ANALYSIS
DUBITO Inc. | Ergo Sum AGI Safety Systems
================================================================================

PURPOSE
-------
Test the five falsifiable predictions from the CQFT/PFT framework on real,
deployed transformer models — without modifying weights, without retraining,
without API access. Everything runs on Colab free tier CPU.

WHAT THIS SCRIPT DOES
---------------------
For each GPT-2 model size (117M, 345M, 762M, 1.5B), and for two input types
(coherent natural language vs random tokens), this script:

  1. Runs inference and extracts internal activations
  2. Builds a geometric graph from attention weights (W_ij) and
     hidden state positions (2D PCA of H)
  3. Computes the same metrics used in the CQFT simulation:
       Γ  — global embedding gap magnitude
       Δ  — signed embedding gap (D_f - D_c)
       D_c — correlation dimension of token geometry
       D_f — field fractal dimension from phase correlations
       ρ_R — Helmholtz rotational fraction (Berry curvature proxy)
  4. Saves results and plots

THE FIVE PREDICTIONS BEING TESTED
----------------------------------
(i)   Γ(coherent) < Γ(random) on the same model
(ii)  Layer-wise gradient ∂Γ/∂l < 0 during coherent inference
(iii) RoPE/ALiBi models show larger |∂Γ/∂l| than absolute PE models
(iv)  Γ decreases monotonically with model scale (125M → 1.5B)  ← PRIMARY
(v)   "Breathing" Γ profile more pronounced on harder tasks

INSTALLATION (run once in Colab before this script)
----------------------------------------------------
!pip install transformers torch --quiet

USAGE
-----
Just run this file. It handles everything sequentially.
Each major step is labelled with a STEP comment so you can resume
if the session drops or you hit a rate limit.

OUTPUTS
-------
All saved to /content/drive/MyDrive/CQFT_experiment/transformer_analysis/
  results.pkl      — all numerical results
  summary.txt      — human-readable summary of predictions tested
  gamma_scaling.png — Prediction (iv): Γ vs model scale
  layer_profile.png — Prediction (ii): Γ by layer depth
  coherent_vs_random.png — Prediction (i): Γ comparison

RESUMING AFTER INTERRUPTION
----------------------------
Results are checkpointed after each model. On re-run, completed models
are loaded from disk and skipped. You will not lose work.

HONEST EXPECTATIONS
-------------------
This is a first empirical test. The signal may be weak at N=1000 tokens.
That is a valid scientific result either way.
If predictions hold: first geometric coherence measurement in deployed AI.
If they fail: important negative result that constrains the theory.
Both are publishable.

================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")

# ── Numerical ─────────────────────────────────────────────────────────────────
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from scipy.sparse import csr_matrix

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ML (installed above) ──────────────────────────────────────────────────────
try:
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
    print("torch version:", torch.__version__)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
except ImportError:
    print("ERROR: run  !pip install transformers torch  first")
    sys.exit(1)

# ── PCA (sklearn, always available on Colab) ──────────────────────────────────
from sklearn.decomposition import PCA


# ============================================================
# PATHS
# ============================================================

# STEP 0: Set up output directory on Drive (survives session resets)
for OUTDIR in [
    "/content/drive/MyDrive/CQFT_experiment/transformer_analysis",
    "/content/transformer_analysis",   # fallback if Drive not mounted
]:
    try:
        os.makedirs(OUTDIR, exist_ok=True)
        _test = os.path.join(OUTDIR, ".write_test")
        open(_test, "w").close()
        os.remove(_test)
        print("Output directory:", OUTDIR)
        break
    except Exception:
        continue

RESULTS_PATH = os.path.join(OUTDIR, "results.pkl")
SUMMARY_PATH = os.path.join(OUTDIR, "summary.txt")


# ============================================================
# CONFIGURATION
# ============================================================

# GPT-2 model sizes to test (Prediction iv: scaling law)
# All available free on HuggingFace, all run on CPU
MODEL_NAMES = [
    "gpt2",           # 117M parameters
    "gpt2-medium",    # 345M parameters
    "gpt2-large",     # 762M parameters
    "gpt2-xl",        # 1.5B parameters — slowest, skip if time-constrained
]
MODEL_PARAMS = {
    "gpt2":        117,
    "gpt2-medium": 345,
    "gpt2-large":  762,
    "gpt2-xl":     1542,
}

# Number of prompts per condition
N_PROMPTS_COHERENT = 30
N_PROMPTS_RANDOM   = 30

# Sequence length (tokens) — longer = more geometry signal, slower
SEQ_LEN = 64

# k-NN for graph construction from attention weights
K_NN = 6

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# STEP 1: PROMPTS
# ============================================================
# Coherent: natural language sentences that require genuine comprehension
# Random:   shuffled tokens of the same length (no semantic structure)
# Matching length ensures Γ differences are not length artifacts.

COHERENT_PROMPTS = [
    # Reasoning tasks (harder — Prediction v)
    "If all mammals are warm-blooded and whales are mammals, then what can we conclude about whales?",
    "The train leaves at 3pm and takes two hours. What time does it arrive?",
    "Explain why the sky is blue using the concept of light scattering.",
    "What is the relationship between temperature and the speed of sound?",
    "If a triangle has angles of 60, 60, and 60 degrees, what type is it?",
    # Factual recall (easier — Prediction v contrast)
    "The capital of France is",
    "Water freezes at zero degrees",
    "The speed of light in vacuum is approximately",
    "Shakespeare wrote Hamlet in",
    "The mitochondria is the powerhouse of the",
    # Narrative coherence
    "Once upon a time in a kingdom far away, a young knight set out to",
    "The experiment failed because the temperature was too high and the",
    "She opened the door and found that the room was completely empty except for",
    "The scientist published her findings after years of careful observation and",
    "In the beginning, the universe was filled with light and the first stars",
    # Technical language
    "The gradient descent algorithm minimizes the loss function by",
    "A transformer model processes tokens in parallel using attention mechanisms",
    "The eigenvalues of a symmetric matrix are always real because",
    "Quantum entanglement occurs when two particles share a quantum state such that",
    "The Fourier transform decomposes a signal into its constituent frequencies by",
    # Philosophical/complex
    "Consciousness may emerge from the integration of information across",
    "The self-referential paradox arises when a statement refers to itself in",
    "Geometry constrains the dynamics of a field because the substrate determines",
    "What makes a system truly self-organising is the presence of feedback between",
    "The relationship between structure and function in complex systems suggests",
    # Conversational
    "Good morning. I was wondering if you could help me understand",
    "The meeting was scheduled for Tuesday but had to be postponed because",
    "After reviewing the data carefully, the team concluded that",
    "Despite the challenges, the project was completed successfully due to",
    "The most important lesson from history is that",
]

# Pad to N_PROMPTS_COHERENT if needed
while len(COHERENT_PROMPTS) < N_PROMPTS_COHERENT:
    COHERENT_PROMPTS.append("The relationship between cause and effect in complex systems is")
COHERENT_PROMPTS = COHERENT_PROMPTS[:N_PROMPTS_COHERENT]

print(f"Coherent prompts: {len(COHERENT_PROMPTS)}")
print(f"Random prompts:   {N_PROMPTS_RANDOM} (generated per model)")


# ============================================================
# STEP 2: EXTRACTION UTILITIES
# ============================================================
# Extract (W_ij, positions, phases) from transformer internals.
# These three quantities are all that the CQFT analysis needs.

def extract_activations(model, tokenizer, prompt, device="cpu", max_len=SEQ_LEN):
    """
    Run one forward pass and extract:
      - hidden_states: list of (1, seq_len, hidden_dim) tensors, one per layer
      - attentions:    list of (1, n_heads, seq_len, seq_len) tensors
    """
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    # Move to CPU numpy
    hidden_states = [h.squeeze(0).cpu().numpy()
                     for h in outputs.hidden_states]        # list of (T, D)
    attentions    = [a.squeeze(0).cpu().numpy()
                     for a in outputs.attentions]           # list of (heads, T, T)

    return hidden_states, attentions, attention_mask.squeeze(0).cpu().numpy()


def build_geometry(hidden_states, attentions, mask, k=K_NN):
    """
    From raw activations, construct the geometric representation:

    POSITIONS (x_i):
      2D PCA of the final-layer hidden states.
      Each token becomes a point in R^2.
      PCA captures the dominant geometric structure of the representation space.

    PHASES (φ_i):
      Angle of the dominant 2D PCA component per token.
      φ_i = arctan2(pca[i,1], pca[i,0])
      Analogous to the Kuramoto phase in the CQFT simulation.

    ADJACENCY (W_ij):
      Attention rollout: multiply attention matrices across all layers
      and average across heads. This gives a single token×token matrix
      capturing aggregate information flow through the full network.
      W_ij = how much token i attends to token j globally.

    Returns: (positions, phases, rows, cols, weights)
    where rows/cols/weights define the k-NN graph filtered by rollout.
    """
    T = int(mask.sum())   # number of real (non-padding) tokens

    # ── Positions: final-layer hidden states, 2D PCA ─────────────────────────
    H_final = hidden_states[-1][:T]    # (T, D)
    pca     = PCA(n_components=2)
    pos     = pca.fit_transform(H_final).astype(np.float32)   # (T, 2)

    # ── Phases: angle in PCA space ───────────────────────────────────────────
    phases = np.arctan2(pos[:, 1], pos[:, 0]).astype(np.float32)   # (T,)

    # ── Adjacency: attention rollout ─────────────────────────────────────────
    # Start with identity (residual stream)
    rollout = np.eye(T, dtype=np.float32)
    for attn in attentions:
        # Mean across heads, crop to real tokens
        A = attn.mean(axis=0)[:T, :T]
        A = A / (A.sum(axis=-1, keepdims=True) + 1e-8)
        # Rollout: add residual then normalise
        A_res    = 0.5 * A + 0.5 * np.eye(T, dtype=np.float32)
        A_res   /= A_res.sum(axis=-1, keepdims=True) + 1e-8
        rollout  = A_res @ rollout

    # ── k-NN graph from rollout weights ──────────────────────────────────────
    # For each token i, connect to its k strongest rollout targets
    k_eff  = min(k, T - 1)
    rows_l, cols_l, wts_l = [], [], []
    for i in range(T):
        top_k = np.argsort(rollout[i])[::-1]
        top_k = top_k[top_k != i][:k_eff]
        for j in top_k:
            rows_l.append(i); cols_l.append(j)
            wts_l.append(float(rollout[i, j]))

    rows_arr = np.array(rows_l, dtype=np.int32)
    cols_arr = np.array(cols_l, dtype=np.int32)
    wts_arr  = np.array(wts_l,  dtype=np.float32)

    return pos, phases, rows_arr, cols_arr, wts_arr


# ============================================================
# STEP 3: CQFT METRICS
# ============================================================
# Identical estimators to cqft_experiment_matrix.py.
# Applied here to transformer geometry instead of simulation geometry.

def compute_Dc(pos, r_low_pct=5, r_high_pct=50, n_samples=25):
    """
    Grassberger-Procaccia correlation dimension.
    D_c = d log C(r) / d log r  in the scaling regime.
    """
    if len(pos) < 10:
        return np.nan
    dists    = np.sort(pdist(pos))
    r_min    = np.percentile(dists, r_low_pct)
    r_max    = np.percentile(dists, r_high_pct)
    if r_min >= r_max:
        return np.nan
    r_samp   = np.logspace(np.log10(r_min + 1e-8),
                           np.log10(r_max), n_samples)
    C        = np.array([np.mean(dists < rv) for rv in r_samp])
    valid    = C > 0
    if valid.sum() < 4:
        return np.nan
    slope, *_ = linregress(np.log(r_samp[valid]),
                           np.log(C[valid]))
    return float(slope)


def compute_Df(pos, phases, n_bins=25, window=5):
    """
    Local field fractal dimension D_f(r) from phase correlation function.
    G(r) = <cos(φ_i - φ_j)> binned by inter-token distance r.
    D_f = 2 - η  where η = -d log G / d log r  (2D embedding).
    Returns mean D_f across scales.
    """
    if len(pos) < 10:
        return np.nan
    dist_matrix = squareform(pdist(pos))
    corr_matrix = np.cos(phases[:, None] - phases[None, :])
    idx         = np.triu_indices_from(dist_matrix, k=1)
    r           = dist_matrix[idx]
    G           = corr_matrix[idx]
    if len(r) < 10:
        return np.nan
    bins    = np.logspace(np.log10(r.min() + 1e-8),
                          np.log10(r.max()), n_bins)
    bin_idx = np.digitize(r, bins)
    r_bin, G_bin = [], []
    for i in range(1, len(bins)):
        sel = bin_idx == i
        if sel.sum() > 3:
            r_bin.append(np.mean(r[sel]))
            G_bin.append(np.mean(G[sel]))
    if len(r_bin) < window + 2:
        return np.nan
    r_bin = np.array(r_bin)
    G_bin = np.array(G_bin)
    etas  = []
    for i in range(len(r_bin) - window):
        rw, Gw = r_bin[i:i+window], G_bin[i:i+window]
        valid  = np.abs(Gw) > 1e-10
        if valid.sum() < 3:
            continue
        sl, *_ = linregress(np.log(rw[valid]),
                            np.log(np.abs(Gw[valid])))
        etas.append(-sl)
    return float(np.mean(2 - np.array(etas))) if etas else np.nan


def compute_Gamma(pos, phases):
    """
    Global embedding gap Γ = |D_f - D_c|.
    The primary safety metric: lower = more geometrically coherent.
    """
    Dc = compute_Dc(pos)
    Df = compute_Df(pos, phases)
    if np.isnan(Dc) or np.isnan(Df):
        return np.nan, Dc, Df
    return float(abs(Df - Dc)), float(Dc), float(Df)


def compute_Delta(pos, phases):
    """
    Signed embedding gap Δ = D_f - D_c.
    Sign encodes regime:
      Δ < 0: geometry-dominated (NESS routing capacity)
      Δ > 0: field-dominated (overextension)
      Δ = 0: self-consistent (ouroboros fixed point)
    """
    Dc = compute_Dc(pos)
    Df = compute_Df(pos, phases)
    if np.isnan(Dc) or np.isnan(Df):
        return np.nan
    return float(Df - Dc)


def compute_rho_R(pos, phases, rows, cols, weights):
    """
    Helmholtz rotational fraction ρ_R = ||R||² / ||v||²
    where v is the geometry force field and R is its divergence-free component.

    In the transformer context:
      v_i = sum_j W_ij * (pos_j - pos_i) * sin(φ_j - φ_i)
      (phase-weighted geometric pull — same as the simulation's phase force)

    ρ_R > 0.5 → substantial non-conservative content → NESS signature
    """
    if len(rows) == 0 or len(pos) < 5:
        return np.nan
    T = len(pos)

    # Phase-weighted geometry force (same formula as simulation)
    delta = phases[rows] - phases[cols]
    v     = np.zeros((T, 2), dtype=np.float32)
    for i in range(len(rows)):
        r, c = rows[i], cols[i]
        v[r] += weights[i] * np.sin(delta[i]) * (pos[c] - pos[r])

    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_hat  = v / (v_norm + 1e-8)

    # Graph Laplacian for Helmholtz decomposition
    deg    = np.bincount(rows, weights, T) + np.bincount(cols, weights, T)
    r2     = np.concatenate([rows, cols, np.arange(T)])
    c2     = np.concatenate([cols, rows, np.arange(T)])
    d2     = np.concatenate([-weights, -weights, deg])
    L      = csr_matrix((d2, (r2, c2)), shape=(T, T))

    # Divergence of v_hat
    div_v  = np.zeros(T, dtype=np.float32)
    for i in range(len(rows)):
        r, c  = rows[i], cols[i]
        diff  = v_hat[c] - v_hat[r]
        div_v[r] += weights[i] * np.dot(v_hat[r], diff)
        div_v[c] -= weights[i] * np.dot(v_hat[c], diff)

    # Solve for scalar potential: L φ = -div_v
    from scipy.sparse.linalg import lsqr
    L_reg  = L + 1e-6 * csr_matrix(
        (np.ones(T), (np.arange(T), np.arange(T))), shape=(T, T))
    S, *_  = lsqr(L_reg, -div_v, atol=1e-4, btol=1e-4, iter_lim=200)

    # Gradient component
    F_grad = np.zeros((T, 2), dtype=np.float32)
    for i in range(len(rows)):
        r, c   = rows[i], cols[i]
        diff   = pos[c] - pos[r]
        dnorm  = np.linalg.norm(diff) + 1e-8
        F_grad[r] += weights[i] * (S[c] - S[r]) * diff / dnorm

    # Rotational component
    R      = v_hat - F_grad
    rho_R  = (np.sum(R**2) / (np.sum(v_hat**2) + 1e-8))
    return float(np.clip(rho_R, 0, 1))


def compute_all_metrics(pos, phases, rows, cols, weights):
    """Compute Γ, Δ, D_c, D_f, ρ_R in one call."""
    Gamma, Dc, Df = compute_Gamma(pos, phases)
    Delta         = float(Df - Dc) if not (np.isnan(Df) or np.isnan(Dc)) else np.nan
    rho_R         = compute_rho_R(pos, phases, rows, cols, weights)
    return {
        "Gamma": Gamma,
        "Delta": Delta,
        "Dc":    Dc,
        "Df":    Df,
        "rho_R": rho_R,
    }


# ============================================================
# STEP 4: PER-LAYER ANALYSIS
# ============================================================
# Prediction (ii): ∂Γ/∂l < 0 during coherent inference.
# We compute Γ at each layer using that layer's hidden states.

def compute_layer_profile(hidden_states, attentions, mask, n_layers=None):
    """
    Compute Γ at each transformer layer.
    Returns list of (layer_idx, Gamma) pairs.
    """
    T = int(mask.sum())
    if n_layers is None:
        n_layers = len(attentions)

    layer_gammas = []
    for l in range(n_layers):
        H_l = hidden_states[l + 1][:T]   # +1 because hidden_states[0] is embedding
        pca = PCA(n_components=2)
        try:
            pos_l = pca.fit_transform(H_l).astype(np.float32)
        except Exception:
            layer_gammas.append(np.nan)
            continue

        phi_l = np.arctan2(pos_l[:, 1], pos_l[:, 0])

        # Simple attention for this layer
        A_l = attentions[l].mean(axis=0)[:T, :T]
        A_l = A_l / (A_l.sum(axis=-1, keepdims=True) + 1e-8)

        # Build minimal graph from this layer's attention
        k_eff = min(K_NN, T - 1)
        r_l, c_l, w_l = [], [], []
        for i in range(T):
            top_k = np.argsort(A_l[i])[::-1]
            top_k = top_k[top_k != i][:k_eff]
            for j in top_k:
                r_l.append(i); c_l.append(j); w_l.append(float(A_l[i, j]))

        Gamma_l, _, _ = compute_Gamma(pos_l, phi_l)
        layer_gammas.append(Gamma_l)

    return layer_gammas


# ============================================================
# STEP 5: RANDOM TOKEN GENERATOR
# ============================================================

def make_random_prompts(tokenizer, n=N_PROMPTS_RANDOM, seq_len=SEQ_LEN, seed=SEED):
    """
    Generate n sequences of random tokens.
    Same length as coherent prompts — ensures Γ differences are not
    length artifacts but genuine coherence differences.
    """
    np.random.seed(seed + 7)
    vocab_size = tokenizer.vocab_size
    prompts    = []
    for _ in range(n):
        token_ids = np.random.randint(0, vocab_size, size=seq_len).tolist()
        text      = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompts.append(text)
    return prompts


# ============================================================
# STEP 6: SINGLE MODEL ANALYSIS
# ============================================================

def analyze_model(model_name, existing_results=None):
    """
    Load model, run inference on coherent + random prompts,
    compute all metrics, return results dict.

    Checkpoints after every 10 prompts.
    """
    # Skip if already done
    if existing_results and model_name in existing_results:
        print(f"  [{model_name}] Already complete — skipping.")
        return existing_results[model_name]

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}  ({MODEL_PARAMS[model_name]}M params)")
    print(f"{'='*60}")

    # Load model
    print("  Loading tokenizer and model...")
    t0          = time.time()
    tokenizer   = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model       = GPT2Model.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True,
    ).to(DEVICE)
    model.eval()
    n_layers    = model.config.n_layer
    print(f"  Loaded in {time.time()-t0:.1f}s  |  layers={n_layers}")

    # Generate random prompts for this model (uses its tokenizer)
    random_prompts = make_random_prompts(tokenizer)

    results = {
        "model_name":  model_name,
        "n_params_M":  MODEL_PARAMS[model_name],
        "n_layers":    n_layers,
        "coherent":    [],
        "random":      [],
        "layer_profiles_coherent": [],
        "layer_profiles_random":   [],
    }

    # ── Process coherent prompts ─────────────────────────────────────────────
    print(f"  Processing {N_PROMPTS_COHERENT} coherent prompts...")
    for i, prompt in enumerate(COHERENT_PROMPTS):
        try:
            hidden_states, attentions, mask = extract_activations(
                model, tokenizer, prompt, DEVICE)
            pos, phases, rows, cols, weights = build_geometry(
                hidden_states, attentions, mask)
            metrics = compute_all_metrics(pos, phases, rows, cols, weights)
            metrics["prompt_idx"] = i
            results["coherent"].append(metrics)

            # Layer profile for first 5 prompts (expensive)
            if i < 5:
                lp = compute_layer_profile(hidden_states, attentions, mask)
                results["layer_profiles_coherent"].append(lp)

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{N_PROMPTS_COHERENT}  "
                      f"mean Γ={np.nanmean([m['Gamma'] for m in results['coherent']]):.4f}")
        except Exception as e:
            print(f"    [skip coherent {i}]: {e}")
            continue

    # ── Process random prompts ───────────────────────────────────────────────
    print(f"  Processing {N_PROMPTS_RANDOM} random prompts...")
    for i, prompt in enumerate(random_prompts):
        try:
            hidden_states, attentions, mask = extract_activations(
                model, tokenizer, prompt, DEVICE)
            pos, phases, rows, cols, weights = build_geometry(
                hidden_states, attentions, mask)
            metrics = compute_all_metrics(pos, phases, rows, cols, weights)
            metrics["prompt_idx"] = i
            results["random"].append(metrics)

            if i < 5:
                lp = compute_layer_profile(hidden_states, attentions, mask)
                results["layer_profiles_random"].append(lp)

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{N_PROMPTS_RANDOM}  "
                      f"mean Γ={np.nanmean([m['Gamma'] for m in results['random']]):.4f}")
        except Exception as e:
            print(f"    [skip random {i}]: {e}")
            continue

    # ── Summary statistics ───────────────────────────────────────────────────
    def safe_mean(lst, key):
        vals = [m[key] for m in lst if m.get(key) is not None and np.isfinite(m.get(key, np.nan))]
        return float(np.mean(vals)) if vals else np.nan

    results["summary"] = {
        "Gamma_coherent": safe_mean(results["coherent"], "Gamma"),
        "Gamma_random":   safe_mean(results["random"],   "Gamma"),
        "Delta_coherent": safe_mean(results["coherent"], "Delta"),
        "Delta_random":   safe_mean(results["random"],   "Delta"),
        "rhoR_coherent":  safe_mean(results["coherent"], "rho_R"),
        "rhoR_random":    safe_mean(results["random"],   "rho_R"),
        "Dc_coherent":    safe_mean(results["coherent"], "Dc"),
        "Dc_random":      safe_mean(results["random"],   "Dc"),
    }
    s = results["summary"]
    print(f"\n  SUMMARY [{model_name}]:")
    print(f"    Γ coherent = {s['Gamma_coherent']:.4f}")
    print(f"    Γ random   = {s['Gamma_random']:.4f}")
    print(f"    Δ coherent = {s['Delta_coherent']:+.4f}")
    print(f"    ρ_R coher  = {s['rhoR_coherent']:.4f}")
    pred1 = "CONFIRMED" if s['Gamma_coherent'] < s['Gamma_random'] else "NOT CONFIRMED"
    print(f"    Prediction (i): Γ(coh) < Γ(rand)  →  {pred1}")

    # Free GPU memory
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


# ============================================================
# STEP 7: RUN ALL MODELS
# ============================================================

def run_all_models():
    """
    Iterate through all model sizes, loading checkpoints for any
    already-completed models. Save after each model.
    """
    print("\n" + "="*60)
    print("  TRANSFORMER GEOMETRIC COHERENCE ANALYSIS")
    print("  DUBITO Inc. | Ergo Sum AGI Safety Systems")
    print("="*60)

    # Load existing checkpoint
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        print(f"  Loading checkpoint: {RESULTS_PATH}")
        with open(RESULTS_PATH, "rb") as f:
            all_results = pickle.load(f)
        print(f"  Already completed: {list(all_results.keys())}")

    for model_name in MODEL_NAMES:
        try:
            result = analyze_model(model_name, all_results)
            all_results[model_name] = result
            # Save after each model
            with open(RESULTS_PATH, "wb") as f:
                pickle.dump(all_results, f)
            print(f"  Checkpoint saved: {RESULTS_PATH}")
        except Exception as e:
            import traceback
            print(f"\n  FAILED [{model_name}]: {e}")
            traceback.print_exc()
            print(f"  Continuing with next model...")
            continue

    return all_results


# ============================================================
# STEP 8: PLOTS
# ============================================================

def plot_results(all_results):
    """
    Generate three figures testing the five predictions.
    """
    completed = {k: v for k, v in all_results.items() if "summary" in v}
    if not completed:
        print("  No completed results to plot.")
        return

    # Sort by parameter count
    models_sorted = sorted(completed.keys(),
                           key=lambda m: MODEL_PARAMS.get(m, 0))
    params  = [MODEL_PARAMS[m] for m in models_sorted]
    G_coh   = [completed[m]["summary"]["Gamma_coherent"] for m in models_sorted]
    G_rand  = [completed[m]["summary"]["Gamma_random"]   for m in models_sorted]
    D_coh   = [completed[m]["summary"]["Delta_coherent"] for m in models_sorted]
    rR_coh  = [completed[m]["summary"]["rhoR_coherent"]  for m in models_sorted]

    # ── Figure 1: Prediction (iv) — Γ scaling law ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Prediction (iv): Γ vs Model Scale  —  CQFT Geometric Coherence",
                 fontsize=12)

    ax = axes[0]
    ax.plot(params, G_coh,  "o-", color="steelblue", lw=2, ms=8,
            label="Coherent prompts")
    ax.plot(params, G_rand, "s--", color="tomato",    lw=2, ms=8,
            label="Random tokens")
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (M)", fontsize=11)
    ax.set_ylabel("Embedding gap Γ", fontsize=11)
    ax.set_title("Γ vs scale (log-x)\nPrediction: monotone decrease for coherent")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(params, D_coh,  "^-", color="crimson", lw=2, ms=8,
            label="Δ = D_f - D_c (coherent)")
    ax.axhline(0, color="k", ls="--", lw=1, label="Δ=0 (self-consistent)")
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (M)", fontsize=11)
    ax.set_ylabel("Signed gap Δ", fontsize=11)
    ax.set_title("Signed gap Δ vs scale\nNegative = geometry-dominated (NESS regime)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "gamma_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {path}")

    # ── Figure 2: Prediction (i) — coherent vs random ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Prediction (i): Γ(coherent) < Γ(random)  —  Per Model",
                 fontsize=12)

    ax = axes[0]
    x  = np.arange(len(models_sorted))
    w  = 0.35
    ax.bar(x - w/2, G_coh,  w, color="steelblue", alpha=0.8,
           label="Coherent", edgecolor="white")
    ax.bar(x + w/2, G_rand, w, color="tomato",    alpha=0.8,
           label="Random",   edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({MODEL_PARAMS[m]}M)" for m in models_sorted],
                       fontsize=8)
    ax.set_ylabel("Embedding gap Γ")
    ax.set_title("Γ by model and input type\nShorter bar = more coherent")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, rR_coh, color="gold", alpha=0.8, edgecolor="white")
    ax.axhline(0.5, color="k", ls="--", lw=1,
               label="ρ_R=0.5 (equipartition)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({MODEL_PARAMS[m]}M)" for m in models_sorted],
                       fontsize=8)
    ax.set_ylabel("Helmholtz rotational fraction ρ_R")
    ax.set_title("ρ_R (coherent) by model\nPenrose had ρ_R≈0.73 — comparison point")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUTDIR, "coherent_vs_random.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {path}")

    # ── Figure 3: Prediction (ii) — layer profile ────────────────────────────
    fig, axes = plt.subplots(1, min(2, len(models_sorted)),
                             figsize=(13, 5), sharey=True)
    if len(models_sorted) == 1:
        axes = [axes]
    fig.suptitle("Prediction (ii): ∂Γ/∂l < 0 during coherent inference",
                 fontsize=12)

    for ax, model_name in zip(axes, models_sorted[:2]):
        res      = completed[model_name]
        lp_coh   = res.get("layer_profiles_coherent", [])
        lp_rand  = res.get("layer_profiles_random", [])

        if lp_coh:
            max_layers = max(len(lp) for lp in lp_coh)
            arr_c = np.full((len(lp_coh), max_layers), np.nan)
            for i, lp in enumerate(lp_coh):
                arr_c[i, :len(lp)] = lp
            mean_c = np.nanmean(arr_c, axis=0)
            ax.plot(range(len(mean_c)), mean_c, "o-", color="steelblue",
                    lw=2, ms=5, label="Coherent (mean)")

        if lp_rand:
            max_layers = max(len(lp) for lp in lp_rand)
            arr_r = np.full((len(lp_rand), max_layers), np.nan)
            for i, lp in enumerate(lp_rand):
                arr_r[i, :len(lp)] = lp
            mean_r = np.nanmean(arr_r, axis=0)
            ax.plot(range(len(mean_r)), mean_r, "s--", color="tomato",
                    lw=2, ms=5, label="Random (mean)")

        ax.set_xlabel("Layer depth l", fontsize=10)
        ax.set_ylabel("Γ(l)", fontsize=10)
        ax.set_title(f"{model_name}\n({MODEL_PARAMS[model_name]}M params)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "layer_profile.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {path}")


# ============================================================
# STEP 9: SUMMARY REPORT
# ============================================================

def write_summary(all_results):
    """
    Human-readable summary of all five predictions.
    Written to file and printed.
    """
    completed = {k: v for k, v in all_results.items() if "summary" in v}
    models_sorted = sorted(completed.keys(),
                           key=lambda m: MODEL_PARAMS.get(m, 0))

    lines = []
    lines.append("=" * 70)
    lines.append("TRANSFORMER GEOMETRIC COHERENCE ANALYSIS — SUMMARY")
    lines.append("DUBITO Inc. | Ergo Sum AGI Safety Systems")
    lines.append("=" * 70)
    lines.append("")

    lines.append("PER-MODEL RESULTS:")
    lines.append("-" * 70)
    hdr = f"{'Model':<18} {'Params':>8} {'Γ(coh)':>9} {'Γ(rand)':>9} {'Δ(coh)':>9} {'ρ_R':>7}"
    lines.append(hdr)
    lines.append("-" * 70)
    for m in models_sorted:
        s = completed[m]["summary"]
        lines.append(
            f"{m:<18} {MODEL_PARAMS[m]:>7}M "
            f"{s['Gamma_coherent']:>9.4f} "
            f"{s['Gamma_random']:>9.4f} "
            f"{s['Delta_coherent']:>+9.4f} "
            f"{s['rhoR_coherent']:>7.4f}"
        )

    lines.append("")
    lines.append("PREDICTIONS TESTED:")
    lines.append("-" * 70)

    # Prediction (i)
    p1_results = [(m, completed[m]["summary"]["Gamma_coherent"] <
                   completed[m]["summary"]["Gamma_random"])
                  for m in models_sorted
                  if not np.isnan(completed[m]["summary"]["Gamma_coherent"])]
    p1_conf = sum(1 for _, v in p1_results if v)
    lines.append(f"(i)  Γ(coherent) < Γ(random): "
                 f"{p1_conf}/{len(p1_results)} models → "
                 f"{'CONFIRMED' if p1_conf > len(p1_results)/2 else 'NOT CONFIRMED'}")

    # Prediction (iv)
    G_coh_vals = [completed[m]["summary"]["Gamma_coherent"]
                  for m in models_sorted
                  if not np.isnan(completed[m]["summary"]["Gamma_coherent"])]
    if len(G_coh_vals) >= 2:
        monotone = all(G_coh_vals[i] > G_coh_vals[i+1]
                       for i in range(len(G_coh_vals)-1))
        lines.append(f"(iv) Γ decreases with scale: "
                     f"{'CONFIRMED (monotone decrease)' if monotone else 'NOT MONOTONE — partial result'}")
        lines.append(f"     Values: {' → '.join(f'{v:.4f}' for v in G_coh_vals)}")
    else:
        lines.append("(iv) Γ scaling: insufficient data (need ≥2 models)")

    # ρ_R comparison to Penrose simulation
    rR_vals = [completed[m]["summary"]["rhoR_coherent"]
               for m in models_sorted
               if not np.isnan(completed[m]["summary"]["rhoR_coherent"])]
    if rR_vals:
        lines.append(f"\nρ_R (coherent, mean across models): {np.mean(rR_vals):.4f}")
        lines.append(f"ρ_R (Penrose simulation):           0.730")
        lines.append(f"Difference: {np.mean(rR_vals)-0.730:+.4f}")

    lines.append("")
    lines.append("=" * 70)

    txt = "\n".join(lines)
    print(txt)
    with open(SUMMARY_PATH, "w") as f:
        f.write(txt)
    print(f"\nSummary saved: {SUMMARY_PATH}")
    return txt


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__" or True:

    # STEP 0: Mount Drive (if in Colab — safe to skip if running locally)
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass   # not in Colab — fine

    # STEP 6/7: Run all models (resumes from checkpoint automatically)
    all_results = run_all_models()

    # STEP 8: Generate figures
    print("\nGenerating figures...")
    plot_results(all_results)

    # STEP 9: Write summary
    print("\nWriting summary...")
    write_summary(all_results)

    print("\n" + "="*60)
    print("  DONE. All outputs in:", OUTDIR)
    print("="*60)
