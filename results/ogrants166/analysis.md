## Findings (ogrants166, N=166, model gpt-4o-mini-2024-07-18)

- **Reference system** `full`: 42.2% accuracy.
- A **single LLM call** reaches 62.1% (-19.9pp vs full, McNemar p=8.8e-05, sig.).
- **Retrieval-only k-NN** (no generation) reaches 67.5% (-25.3pp vs full).
- **Is it just more compute?** The full pipeline spends ~0.61× the tokens of self_consistency@30, yet scores 42.2% vs 62.1% (-19.9pp, McNemar p=8.8e-05 — distinguishable at matched budget).
- **Component contributions (full − ablated):**
    - Debate: -1.2pp (McNemar p=0.823803, n.s.)
    - Meta-review: -0.6pp (McNemar p=1.0, n.s.)
    - Guardrails: -1.2pp (McNemar p=0.831812, n.s.)
    - Retrieval: +1.8pp (McNemar p=0.728332, n.s.)
    - Debate+Meta+Guardrails (all extras): -1.2pp (McNemar p=0.831812, n.s.)
