
# Discussion 1

## Summary of Results

In this work, we benchmarked a **parameter-only warm-start strategy for QAOA** based on a **transformer mapping**
from an original MaxCut Ising Hamiltonian \(J\) to a paper-compatible family
\(J'\) defined by

\[
J'_{ij} = P_i + P_j .
\]

The key idea was to:
1. Learn \(J'\) from \(J\) using lightweight structural proxies,
2. Solve \(J'\) analytically (classical theorem),
3. Train QAOA on \(J'\) to obtain optimal parameters \(\theta^*_{J'} = (\gamma, \beta)\),
4. Transfer these parameters to QAOA on the original problem \(J\).

We evaluated this approach on three benchmark families (ER-unweighted, ER-weighted, and 3-regular graphs) with
\(N \le 16\), comparing against a strong multi-start QAOA baseline.

## Key Observation: Classical Transfer ≠ Quantum Transfer

Across all benchmarks, we observe that **improving classical similarity does not reliably translate into improved QAOA performance**.

Although the transformer successfully captures **classical structure**—as reflected by moderate-to-high
behavior proxy scores (row correlations, eigen-overlap, frustration correlation)—the **QAOA warm-start
often underperforms or matches the baseline** rather than consistently exceeding it.

This is clearly visible in:

- **Fig. 3**: Mean gain vs baseline (warm & transfer),
- **Fig. 4**: Behavior proxy vs warm-start gain,
- **Fig. 5**: Approximation ratios relative to brute-force optima,
- **Fig. 6**: Stage-B correlation diagnostics.

In most instances, the *warm-finetuned* QAOA slightly recovers performance relative to transfer-only,
but still fails to produce a systematic advantage over baseline QAOA.

## Interpretation

These results support the hypothesis that:

> **A Hamiltonian that preserves classical ground-state structure does not necessarily preserve the
variational landscape explored by QAOA.**

More concretely:

- QAOA performance depends on **interference patterns, entanglement generation, and parameter-dependent dynamics**, not just on low-energy classical configurations.
- The family \(J'_{ij} = P_i + P_j\), while analytically solvable, induces a **very restricted entanglement structure** under standard QAOA mixers.
- As a result, the optimal parameters \(\theta^*_{J'}\) may correspond to *flat or misleading directions*
in the QAOA landscape of \(J\).

This explains why we frequently observe:
- Negative or negligible gains from transfer-only,
- Weak or inconsistent correlation between transformer behavior proxies and warm-start improvement.

## Implications

Our findings highlight a fundamental limitation of **classical-structure-driven warm starts** for QAOA:

- Matching spectra, correlations, or ground states is **not sufficient**.
- Effective quantum warm-starts must account for **dynamical and entanglement-related properties** of the circuit.

This does *not* invalidate warm-start strategies in general, but strongly suggests that **quantum-aware
transformers**—e.g., trained to preserve short-time QAOA dynamics, gradients, or parameter sensitivity—are required.

## Outlook

Future directions include:
- Designing transformer objectives based on **QAOA layer responses** or **parameter-gradient alignment**,
- Exploring **mixer-aware mappings** instead of purely diagonal Hamiltonians,
- Extending benchmarks to hardware-efficient ansätze and noisy backends.

Overall, this study provides a clean negative result:
> preserving classical structure alone is insufficient to guarantee transferable QAOA parameters.

