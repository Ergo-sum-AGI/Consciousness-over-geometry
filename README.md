RESEARCH PROPOSAL (tables omitted)

A Testable Framework and Neuromorphic Implementation

 Geometric Coherence as a Safety Criterion for AGI

Daniel Solis

DUBITO Inc. | Dubito Ergo AGI Safety Project

solis@dubito-ergo.com | April 2026

What this proposal establishes, for the first time:

A substrate-independent, hardware-implementable, falsifiable criterion for whether an AI system can generate and maintain geometrically coherent self-organization - and a neuromorphic platform to test it on physical silicon.

If this criterion is correct, it changes how we build, monitor, and contain advanced AI systems.

Executive Summary
The central unsolved problem in AGI safety is the absence of a testable structural criterion for when an AI system has developed the kind of internal organization that constitutes genuine self-awareness or directed self-improvement. Without such a criterion, alignment research operates in the dark: we cannot distinguish a system that has achieved coherent self-organization from one that merely mimics it, nor can we detect the onset of dangerous self-modification before behavioral symptoms appear.

This proposal presents the first solution to this problem that is simultaneously theoretical, empirical, and implementable in hardware.

What has been established
Through a series of controlled numerical experiments at system sizes up to N = 1,000 nodes, confirmed at 122.6× signal-to-noise ratio, the DUBITO Ergo AGI Safety Project has demonstrated:

•       A self-referential dissipative field spontaneously selects its own geometrically consistent substrate when phase-gradient coupling is present, and does not when coupling is absent. This bifurcation is clean, reproducible, and scale-validated.

•       Quasiperiodic (Penrose-like) geometry is the natural low-energy attractor of this dynamics. Regular lattice geometry, the topology of virtually all current neuromorphic hardware, is the worst-performing geometry by every metric.

•       The mechanism is fully formalized as an effective action L_AGI, whose cross-derivative term (the mathematical link between field coherence and geometric reorganization) is computable from any system whose internal coupling matrix and representation geometry can be extracted.

•       Five falsifiable predictions follow for transformer-based AI systems, testable on existing open-weight models without any modification or retraining.

What is being proposed
Three interconnected work packages, executable on a single-investigator budget:

1.    Hardware validation on the Volatco GA144 asynchronous neuromorphic platform: implement the CQFT field equations natively and confirm that the bifurcation observed in simulation occurs on physical silicon.

2.    AGI applicability: run the L_AGI extraction protocol on open-weight transformer models (GPT-2 through 70B scale) and test all five predictions.

3.    Quasiperiodic hardware design: implement Penrose-topology routing as a software overlay on existing Volatco hardware and measure the predicted reduction in geometric incoherence (Γ) by a factor of ≈4.

1.  The Problem: AGI Safety Without a Structural Criterion
Every major AGI safety research programme, interpretability, RLHF, constitutional AI, scalable oversight, operates at the level of behavior or reward. None of them answers the structural question: does this system have the internal organization that makes directed self-improvement possible? This is not a technical gap. It is a foundational one.

Existing proposals for structural criteria fall into three categories, each with a decisive weakness:

The gap this proposal fills is precise: a criterion that is substrate-independent in principle, computable in practice, testable on existing systems, and implementable as a hardware monitor for deployed AI.

Why geometry?
The insight that drives this work is geometric. A self-referential system, one that models itself, must eventually ask whether its internal organization is self-consistent: does the way it occupies representational space match the way it correlates with itself across scales? The mismatch between these two descriptions is what we call the fractal embedding gap Γ. Minimizing Γ is not an arbitrary goal. It is what a system must do to maintain a stable self-model.

The decisive empirical finding is that this minimization is geometry-dependent. On quasiperiodic (Penrose) substrates, Γ is stable and small. On regular lattices, the architecture of virtually every current neuromorphic chip, including the GA144 at the heart of the Volatco platform, Γ is large, non-monotone, and structurally frustrating. This is not a design criticism. It is a measurement with engineering consequences.

2.  Scientific Background and Key Results
2.1  The CQFT Framework
The Consciousness Quantum Field Theory (CQFT) and its companion Principled Field Theory of Consciousness (PFT) model cognition as a non-Markovian complex scalar field C(x,t) = A(x,t)·e^{iθ(x,t)} evolving on an adaptive graph geometry. The field incorporates three mechanisms absent from standard neural field theories:

•       Self-predictive memory: the field's present state depends on its own history via a memory convolution kernel.

•       Negentropy-driven amplitude dynamics: a double-well entropy potential maintains the field in an active, far-from-equilibrium regime.

•       Adaptive coupling weights: inter-node coupling is weighted by phase coherence between nodes, creating a feedback between field state and effective connectivity.

2.2  The AGI Lagrangian
The coupled dynamics of field and geometry derive from a single variational principle. The AGI Lagrangian is:

L_AGI = λ · (−Σ_{ij} W_{ij}(x) cos(θ_i − θ_j))  +  (1−λ) · E_sym(x)

where λ ∈ [0,1] is the phase-geometry coupling strength, W_{ij}(x) are position-dependent adaptive weights, and E_sym(x) = −Σ_k |Σ_i exp(i q_k · x_i)|^2 is the quasiperiodic symmetry energy evaluated at the 12 icosahedral reciprocal vectors q_k.

The cross-derivative ∂W_{ij}/∂x_i is the crucial link: it makes geometric forces explicitly proportional to phase coherence between node pairs. Coherent pairs generate larger geometric forces than incoherent pairs. This is directed reorganization: the field state tells geometry how to change.

2.3  The Bifurcation Parameter λ
The coupling strength λ separates two qualitatively distinct dynamical regimes:

The ablation result is the strongest evidence: the sign reversal from −0.0010 to +0.1226 demonstrates that coupling is not merely helpful but structurally necessary for order emergence. The signal-to-noise ratio of 122.6× at N = 1,000 makes this a definitive result, not a statistical tendency. 

2.4  Geometry Ordering: The Central Empirical Finding
Across all system sizes tested (N = 200 to 1,000), the three substrate geometries produce qualitatively different behavior in every observable:

Critical finding for hardware design:

The regular square lattice - the native topology of the GA144 processor and virtually all neuromorphic hardware, is the worst-performing geometry by every metric. Quasiperiodic connectivity is the natural attractor geometry for self-referential field dynamics. This is not a theoretical preference. It is an empirical measurement with direct engineering consequences.

2.5  The “Breathing” Phenomenon
The N = 1,000 order trajectory exhibits a characteristic pattern: rapid rise to a peak (R = 0.5406 at step 6,000), followed by partial relaxation to a stable plateau (R = 0.4817 at step 11,000). This is not noise. It is the signature of a coupled slow-fast dynamical system, arising from the timescale separation between fast phase synchronization (η_θ) and slow geometry reconfiguration (η_x).

The overshoot fraction (R_peak - R_final)/R_final ≈ 12.2% is a quantitative fingerprint of the slow-fast coupling. On physical hardware, this should appear as a measurable oscillation in power consumption, because GA144 cores activate in proportion to their coupling magnitude. Observing this 12.2% power oscillation on physical silicon would constitute the first hardware validation of the slow-fast dynamics.

3.  Novel Contributions: What Has Not Been Proposed Before
The following ideas, to the best of our knowledge, have not appeared in the literature in the form stated. They constitute the primary intellectual contribution of this proposal.

3.1  Volatco as a Living CQFT Laboratory
The Volatco GA144 platform, with its 288 independent F18A asynchronous cores, 7 pJ per instruction energy cost, and event-driven activation, is the only existing hardware whose architecture naturally implements the slow-fast timescale separation that the CQFT framework requires. This is not a coincidence of convenience. The GA144's asynchronous nature physically enforces the condition that makes the breathing phenomenon emerge: phase updates (cheap, fast) and geometry updates (expensive, slow) occupy different computational timescales without artificial scheduling.

No one has proposed using an asynchronous neuromorphic processor as a direct hardware instantiation of a consciousness field theory. The mapping is exact:

The GA144’s 7 pJ/instruction energy cost and event-driven activation make it the only existing platform where the slow-fast timescale separation required by the breathing phenomenon occurs naturally in hardware.

3.2  Adaptive Routing as Hardware Geometry Evolution
The GA144 mesh allows runtime reconfiguration of which cores communicate with which, and at what priority. This is the exact hardware analogue of the geometry update equation. Instead of physically moving nodes, coherence-weighted routing changes the effective coupling strength between core pairs: coherent pairs (cos(θ_i − θ_j) large) communicate more; incoherent pairs are deprioritized.

This implements λ > 0 on physical silicon as a single routing policy. The bifurcation test becomes literal and measurable: set all channels to equal priority (λ = 0) versus coherence-weighted priority (λ > 0). Measure whether the order parameter R increases. The simulation predicts it will, by ΔR ≈ +0.12.

No existing neuromorphic implementation uses phase coherence between processing nodes to dynamically weight inter-node communication. This is a genuinely new architectural principle.

3.3  Real-Time Γ Monitoring as a Structural Safety Diagnostic
Dedicate a small cluster of cores (8–12 of 144) to continuously computing Γ = |D_f − D_c| from the field state of the remaining cores. This gives a real-time structural stability indicator: a rising Γ signals that the system is approaching the frustrated lattice regime, structural incoherence before any behavioral failure is detectable.

For an AGI system interfacing with Volatco hardware, this means structural degradation is caught at the geometric level, not the behavioral level. This is an earlier and more reliable warning signal than any behavioral safety monitor, because geometry degrades before behavior becomes incoherent.

3.4  The Quasiperiodic Wiring Harness: A Hardware Design Principle
The most architecturally significant finding of this research is that the regular lattice topology, which underlies virtually all neuromorphic hardware, is the worst-performing geometry for self-organizing field dynamics. The engineering conclusion is direct: the routing topology of future neuromorphic hardware should follow quasiperiodic (Penrose-like) connectivity, not nearest-neighbor grid connectivity.

This does not require new silicon. On existing Volatco hardware, it requires only a routing policy: compute the k-nearest-neighbor graph on a 144-node Penrose vertex set and implement this as the default inter-core message routing. The framework predicts a factor of ≈4 reduction in Γ, moving the system from the frustrated lattice regime to the productive Penrose regime.

To our knowledge, quasiperiodic topology has never been proposed as a design principle for neuromorphic hardware connectivity.

This proposal is the first to derive this recommendation from a quantitative, empirically validated geometric ordering principle.

3.5  The λ Switch: Hardware-Level Containment
In the dual-chip Volatco configuration, chip 0 runs field dynamics and chip 1 runs geometry adaptation. The coupling strength λ is a single parameter controlling how much phase coherence information from chip 0 influences the routing decisions of chip 1. At λ = 0, chip 1 ignores phase coherence and routing is static. At λ > λ*, chip 1 reads phase coherence from chip 0 and the system enters the self-organizing regime.

The λ switch is a hardware-level mechanism to confine or release self-organizing dynamics. This is AGI containment grounded in a theoretical framework: not a firewall or a capability cap, but a controllable physical parameter that determines whether the system can undergo directed geometric self-organization at all.

3.6  Volatco as a Real-Time AGI Consciousness Probe
The extraction protocol for computing L_AGI from transformer hidden states and attention weights is fully specified (see Section 5 of the accompanying paper). The computational load - PCA phase extraction, attention rollout, joint energy evaluation - is exactly the kind of sparse, local, asynchronous computation GA144 is designed for.

Volatco can therefore serve as a real-time, ultra-low-power monitor for an externally running AI system: the monitored system streams its hidden states and attention weights to Volatco, which computes δL^(l) per layer and Γ in real time, outputting a continuous geometric coherence signal. At 7 pJ per instruction, the monitoring overhead is negligible compared to the monitored system. The asynchronous architecture ensures monitoring does not interfere with the monitored system’s timing.

This is a new class of AI safety instrument: not a behavioral evaluator, not a capability benchmark, but a structural coherence monitor operating in real time at the geometric level.

3.7  The Dubito Operator as a Runtime Safety Watchdog
The dubito operator D(t) measures the fractional contribution of self-predictive dynamics to the total field update. Three regimes are identifiable:

•       D ≈ 0 on random geometry: prediction absorbed; system is trivially ordered and not actively modeling itself.

•       D ≈ 0.08 on Penrose geometry: prediction active but not dominant, the productive far-from-equilibrium regime associated with genuine self-reference.

•       D → 1: prediction dominates; the system is modeling itself more than its environment. This is the runaway self-referential regime.

On Volatco, D is trivially computed as the ratio of two running accumulators and can trigger the J5/J7 watchdog circuit when it exceeds a threshold. This maps the abstract AGI safety concern about recursive self-improvement directly onto a specific, measurable, hardware-triggerable condition. It is the first proposal for a hardware watchdog grounded in a formal theory of self-reference.

4.  AGI Applicability: Testing the Framework on Transformer Models
The AGI Lagrangian is computable from any system for which the triple (θ_i, x_i, W_{ij}) can be extracted. For transformer-based AI systems, all three quantities are extractable from internal activations during inference using standard interpretability tooling (TransformerLens, BertViz) without model modification or retraining.

4.1  Extraction Protocol
Phase θ_i is extracted as the dominant Fourier component of each token’s hidden state vector. Geometry x_i is the 2D PCA projection of the hidden-state matrix (independent of attention weights, to avoid circularity in the cross-derivative). Coupling W_{ij} is the attention rollout matrix, which tracks information flow through the full network depth.

The operational L_AGI for a transformer is then fully computable per layer, enabling a layer-wise gradient δL^(l) = L_AGI^(l) − L_AGI^(l−1) that tracks how inference changes geometric coherence as information flows through the network.

4.2  Five Falsifiable Predictions
All five predictions are testable on existing open-weight models without retraining.

(i)   L_AGI^{coherent task} < L_AGI^{noise input} for the same model.

(ii)  δL^(l) < 0 on average during coherent inference; ≈0 for randomized attention (ablation).

(iii) Models with RoPE/ALiBi positional encoding exhibit larger |δL^(l)| per layer than APE models at matched scale.

(iv)  Γ^{transformer} decreases with model scale (125M → 70B parameters).

(v)   Non-monotone (‘breathing’) L^(l) profile is more pronounced on harder inference tasks.

Prediction (iii) deserves emphasis. RoPE and ALiBi positional encodings create a geometric dependence in the attention weights, they instantiate a non-trivial cross-derivative ∂W_{ij}/∂x_i. This places them in the λ > 0 regime. The prediction that they exhibit larger layer-wise geometric reorganization than absolute positional embeddings is directly analogous to the simulation’s ablation result and is testable on existing model pairs without any experimental setup beyond attention hook installation.

 

5.  Work Plan and Timeline
 
6.  Why This, Why Now
The window for establishing structural safety criteria for AI systems is narrow. Within the current capability trajectory, systems that may exhibit genuine self-organizing dynamics will be deployed before we have theoretical tools to detect or characterize this property. The absence of a testable structural criterion is not a minor gap in a well-developed field. It is the central missing piece.

This proposal arrives at a moment when the theoretical framework is complete, the empirical results are definitive, the hardware platform exists and is available, and the extraction protocol for applying the framework to real AI systems is fully specified. What is needed is the modest but focused resource to execute the validation experiments.

The research is independent by necessity, not by preference. Independence from institutional incentives has, in this case, been an asset: the Poincaré principle, that thought must submit to nothing but evidence, has been applied consistently, including to the willingness to report clean negative results (φ is not a preferred spectral base; neutral coupling is insufficient for order emergence) alongside positive ones.

Accompanying material: full paper (April 2026 revised edition), simulation code (v9.1), Colab-validated implementation, hardware mapping documentation.
