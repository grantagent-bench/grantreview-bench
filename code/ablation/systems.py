"""Systems under test for the controlled ablation study.

Every system maps a proposal (an EvalRecord) to a prediction:

    {"system": str, "score": float|None, "verdict": str|None,
     "pred": "awarded"|"declined", "meta": {...}}

Compute (call/token counts) is measured by the caller via budget.meter_llm(),
so systems here contain no accounting code.

Families
--------
Baselines that ask "is it just more compute?":
  * single_call                 -- one prompt, one output, same base model
  * multi_criteria_single_call  -- criterion agents COLLAPSED into one prompt
  * self_consistency(n)         -- n independent samples, averaged / majority vote

Ablations of the deployed multi-agent pipeline (run_agentic with toggles):
  * full          -- retrieval + guardrails + debate + meta  (== deployed system)
  * no_retrieval  -- retrieval OFF
  * no_debate     -- debate OFF
  * no_meta       -- meta-review OFF
  * no_guardrails -- guardrails OFF
  * agents_only   -- agent panel + fusion only (no retrieval-aug reasoning extras)
  * debate_only   -- agent panel + debate only (no retrieval/guardrails/meta)

Retrieval-only (no generation LLM):
  * retrieval_knn -- leave-one-out k-NN over proposal embeddings (majority label)

The agentic path mirrors backend/api.py::process_review_run_agentic exactly:
agents (+inline guardrails) -> single panel debate -> fuse_results ->
meta_review -> fuse_results_agentic.  Toggles remove one component at a time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .budget import raw_chat


# ── shared helpers ────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _thresholds(profile: str) -> tuple[float, float]:
    from grantagent.profiles.loader import get_profile
    th = get_profile(profile).get("thresholds", {})
    return float(th.get("fund", 4.0)), float(th.get("borderline", 3.3))


def _verdict_from_score(score: float, profile: str) -> str:
    fund, border = _thresholds(profile)
    if score >= fund:
        return "Fund-Aligned"
    if score >= border:
        return "Borderline"
    return "Misaligned"


def _pred_from_verdict(verdict: str) -> str:
    # Matches the convention in run_full_evaluation.py / the paper.
    return "awarded" if verdict == "Fund-Aligned" else "declined"


def _proposal_text(sections: dict, limit: int = 8000) -> str:
    """Concatenate the substantive proposal sections for a single-shot prompt.

    Uses the SAME section vocabulary the deployed agents see
    (grantagent.agents.utils.build_submission_context) so the baseline is a fair
    "same input" comparison, but with a generous per-section budget because this
    is one call rather than a per-agent 600-char clip.  The large 'refs' blob is
    excluded (as the agents' submission context excludes it).
    """
    if not isinstance(sections, dict):
        return str(sections)[:limit]
    keys = ("title", "abstract", "intro", "methods", "results", "limitations", "budget")
    parts = []
    for key in keys:
        val = sections.get(key)
        if val:
            parts.append(f"{key.title()}: {val[:2500]}")
    blob = "\n\n".join(parts) if parts else json.dumps(sections)
    return blob[:limit]


def _query_text(sections: dict) -> str:
    if not isinstance(sections, dict):
        return str(sections)[:1500]
    return (sections.get("abstract") or sections.get("intro")
            or _proposal_text(sections, 1500))


@dataclass
class Context:
    """Shared, read-only run context passed to every system."""
    k: int = 8
    sc_n: int = 10
    sc_temperature: float = 0.8
    knn_k: int = 5
    _gindex: object = field(default=None, repr=False)
    _knn_cache: dict = field(default_factory=dict, repr=False)

    def gindex(self):
        if self._gindex is None:
            try:
                from grantagent.vector.index import GlobalIndex
                self._gindex = GlobalIndex(index_dir="./indices")
            except Exception:
                self._gindex = False  # sentinel: unavailable
        return self._gindex or None

    def retrieve(self, rec) -> list:
        gi = self.gindex()
        if gi is None:
            return []
        q = _query_text(rec.sections)
        try:
            return gi.query(q, k=self.k, grant_type=rec.profile) if q else []
        except Exception:
            return []


# ── baseline 1: single call ───────────────────────────────────────────────

def _single_call_score(rec, ctx, *, temperature: float, seed: int | None) -> Optional[float]:
    from grantagent.profiles.loader import get_profile
    criteria = ", ".join(get_profile(rec.profile).get("criteria", [])) or "overall quality"
    prompt = (
        f"You are reviewing a grant proposal for the {rec.profile} program.\n"
        f"Score the overall proposal on a 0-5 scale, weighing these criteria: {criteria}.\n\n"
        f"Proposal:\n{_proposal_text(rec.sections)}\n\n"
        'Respond with ONLY a JSON object: {"score_0_5": <float 0-5>}'
    )
    txt = raw_chat([{"role": "user", "content": prompt}],
                   temperature=temperature, seed=seed)
    data = _parse_json(txt)
    if not data:
        return None
    try:
        return max(0.0, min(5.0, float(data.get("score_0_5"))))
    except Exception:
        return None


def single_call(rec, ctx: Context) -> dict:
    score = _single_call_score(rec, ctx, temperature=0.0, seed=42)
    parse_fail = score is None
    if parse_fail:
        score = 2.5  # neutral fallback (also the heuristic/offline path)
    verdict = _verdict_from_score(score, rec.profile)
    # parse_fail marks rows where the model produced no parseable score, so
    # audits can distinguish a genuine 2.5 prediction from the fallback.
    return {"system": "single_call", "score": score, "verdict": verdict,
            "pred": _pred_from_verdict(verdict),
            "meta": {"parse_fail": True} if parse_fail else {}}


# ── baseline 2: criterion agents collapsed into one prompt ─────────────────

def multi_criteria_single_call(rec, ctx: Context) -> dict:
    from grantagent.agents.base import _build_react_agents
    from grantagent.fusion.arbiter import fuse_results

    names = [a.agent_name for a in _build_react_agents(rec.profile)]
    schema = ", ".join(f'"{n}": <float 0-5>' for n in names)
    prompt = (
        f"You are a grant review panel for the {rec.profile} program. In a single "
        f"pass, score the proposal on EACH criterion below (0-5 each).\n\n"
        f"Criteria: {', '.join(names)}\n\n"
        f"Proposal:\n{_proposal_text(rec.sections)}\n\n"
        f"Respond with ONLY a JSON object: {{{schema}}}"
    )
    txt = raw_chat([{"role": "user", "content": prompt}], temperature=0.0, seed=42)
    data = _parse_json(txt) or {}

    agent_results = []
    for n in names:
        try:
            s = max(0.0, min(5.0, float(data.get(n))))
        except Exception:
            s = 2.5
        agent_results.append({"agent": n, "score_0_5": s, "red_flags": [], "evidence": []})

    verdict = fuse_results(agent_results, rec.profile, doc_length=rec.text_len)
    v = verdict.get("verdict", "Misaligned")
    return {"system": "multi_criteria_single_call",
            "score": verdict.get("overall_weighted_score"), "verdict": v,
            "pred": _pred_from_verdict(v), "meta": {"by_agent": verdict.get("by_agent", {})}}


# ── baseline 3: self-consistency / multi-sample ────────────────────────────

def self_consistency(rec, ctx: Context) -> dict:
    scores, verdicts = [], []
    for i in range(ctx.sc_n):
        s = _single_call_score(rec, ctx, temperature=ctx.sc_temperature, seed=42 + i)
        if s is None:
            continue
        scores.append(s)
        verdicts.append(_verdict_from_score(s, rec.profile))
    if not scores:
        score = 2.5
        verdict = _verdict_from_score(score, rec.profile)
    else:
        score = sum(scores) / len(scores)                       # score averaging
        verdict = max(set(verdicts), key=verdicts.count)        # majority vote
    return {"system": f"self_consistency@{ctx.sc_n}", "score": score, "verdict": verdict,
            "pred": _pred_from_verdict(verdict),
            "meta": {"n_samples": len(scores), "score_std": _std(scores)}}


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ── ablations of the deployed agentic pipeline ─────────────────────────────

def run_agentic(rec, ctx: Context, *, retrieval=True, guardrails=True,
                debate=True, meta=True, debate_alpha: float = 0.0) -> dict:
    """Faithful, toggleable re-implementation of process_review_run_agentic.

    debate_alpha > 0 additionally WIRES the debate consensus score into fusion
    per the design-intent formula S' = S + alpha * (S_debate - S). The deployed
    system computes the consensus but never applies it (the disconnect analyzed
    in the paper's §6.6); this flag creates the causally-connected variant the
    debate ablation actually requires.
    """
    from grantagent.agents.base import _build_react_agents, AgentResult
    from grantagent.agents.guardrails import run_guardrails
    from grantagent.fusion.debate import run_debate, AgentPosition
    from grantagent.agents.meta_reviewer import run_meta_review
    from grantagent.fusion.arbiter import fuse_results, fuse_results_agentic
    from grantagent.profiles.loader import get_profile

    profile = rec.profile
    retrieved = ctx.retrieve(rec) if retrieval else []
    agents = _build_react_agents(profile)

    agent_results: list[dict] = []
    for agent in agents:
        try:
            result, trace = agent.run(rec.sections, retrieved, profile)
            rd = result.model_dump()
            if guardrails:
                gr = run_guardrails(rd, rec.sections, trace.to_dict())
                if gr.adjusted_confidence is not None:
                    rd["confidence"] = gr.adjusted_confidence
            agent_results.append(rd)
        except Exception as exc:
            agent_results.append(AgentResult(
                agent=agent.agent_name, score_0_5=0.0,
                red_flags=["agent_error"], evidence=[], notes=str(exc)).model_dump())

    debate_records: list[dict] = []
    if debate:
        positions = [AgentPosition(
            agent_name=r.get("agent", ""), score=float(r.get("score_0_5", 0)),
            confidence=float(r.get("confidence", 0.5)),
            reasoning=r.get("notes", "")[:400], red_flags=r.get("red_flags", []),
        ) for r in agent_results]
        if len(positions) >= 2:
            dr = run_debate(criterion="overall_panel", positions=positions,
                            max_rounds=2, convergence_threshold=1.0,
                            debate_trigger_threshold=1.5)
            debate_records.append(dr.to_dict())

    base = fuse_results(agent_results, profile, doc_length=rec.text_len)

    # Design-intent debate wiring: S' = S + alpha * (S_debate - S), then re-threshold.
    if debate_alpha and debate_records and not base.get("hard_flag_override"):
        d = next((d for d in debate_records
                  if d.get("debate_triggered") and d.get("final_score") is not None), None)
        if d is not None:
            w = float(base["overall_weighted_score"])
            adj = round(w + debate_alpha * (float(d["final_score"]) - w), 2)
            fund_t, border_t = _thresholds(profile)
            base["overall_weighted_score"] = adj
            base["verdict"] = ("Fund-Aligned" if adj >= fund_t
                               else "Borderline" if adj >= border_t else "Misaligned")
            base["p_fund"] = round(adj / 5.0, 2)
            base["debate_score_applied"] = True

    meta_review: dict = {}
    if meta:
        th = get_profile(profile).get("thresholds", {})
        meta_review = run_meta_review(
            agent_results=agent_results, debate_records=debate_records,
            profile=profile, weighted_score=base.get("overall_weighted_score", 0),
            verdict=base.get("verdict", "Misaligned"),
            fund_threshold=float(th.get("fund", 4.0)),
            borderline_threshold=float(th.get("borderline", 3.3)))

    if not debate_alpha:
        # Deployed path, bit-for-bit: fuse_results_agentic recomputes base fusion
        # from raw agent scores (which is why applied debate scores would be lost
        # on this path — the §6.6 disconnect).
        return fuse_results_agentic(
            agent_results, profile,
            debate_records=debate_records or None,
            meta_review=meta_review or None, doc_length=rec.text_len)

    # Connected path: replicate fuse_results_agentic's post-processing on the
    # adjusted base instead of letting it recompute (and thereby discard) the score.
    final = dict(base)
    if meta_review and not meta_review.get("agrees_with_verdict", True):
        rec_v = meta_review.get("recommended_verdict", final["verdict"])
        if rec_v in ("Fund-Aligned", "Borderline", "Misaligned"):
            final["original_verdict"] = final["verdict"]
            final["verdict"] = rec_v
            final["verdict_overridden"] = True
    debated = [d for d in debate_records if d.get("debate_triggered")] if debate_records else []
    final["debates_triggered"] = len(debated)
    if debated:
        final["debate_confidence"] = round(
            sum(d.get("final_confidence", 0.5) for d in debated) / len(debated), 2)
    if meta_review:
        final["meta_review_summary"] = meta_review.get("summary_statement", "")
        final["flag_for_human"] = meta_review.get("flag_for_human", False)
    return final


def _agentic_system(name, **toggles):
    def fn(rec, ctx: Context) -> dict:
        v = run_agentic(rec, ctx, **toggles)
        vd = v.get("verdict", "Misaligned")
        return {"system": name, "score": v.get("overall_weighted_score"),
                "verdict": vd, "pred": _pred_from_verdict(vd),
                "meta": {"debates_triggered": v.get("debates_triggered", 0),
                         "verdict_overridden": v.get("verdict_overridden", False),
                         "by_agent": v.get("by_agent", {})}}
    fn.__name__ = name
    return fn


full          = _agentic_system("full",          retrieval=True,  guardrails=True,  debate=True,  meta=True)
no_retrieval  = _agentic_system("no_retrieval",  retrieval=False, guardrails=True,  debate=True,  meta=True)
no_debate     = _agentic_system("no_debate",     retrieval=True,  guardrails=True,  debate=False, meta=True)
no_meta       = _agentic_system("no_meta",       retrieval=True,  guardrails=True,  debate=True,  meta=False)
no_guardrails = _agentic_system("no_guardrails", retrieval=True,  guardrails=False, debate=True,  meta=True)
agents_only   = _agentic_system("agents_only",   retrieval=True,  guardrails=False, debate=False, meta=False)
# Debate as the ONLY extra on top of the agent panel (no retrieval/guardrails/meta):
# isolates the debate component's marginal signal.  debate_alpha=0.3 (same as
# full_debate_connected) is REQUIRED here: without it the deployed fusion path
# recomputes from raw agent scores and discards the debate consensus (the §6.6
# disconnect), which would make this arm a no-op measurement of debate.
debate_only   = _agentic_system("debate_only",   retrieval=False, guardrails=False, debate=True,  meta=False,
                                debate_alpha=0.3)
# Debate CONNECTED to fusion (design-intent alpha=0.3) — the causally valid debate arm.
full_debate_connected = _agentic_system("full_debate_connected",
                                        retrieval=True, guardrails=True,
                                        debate=True, meta=True, debate_alpha=0.3)


# ── long-context control: single call over the (near-)full document ─────────

def single_call_fulldoc(rec, ctx: Context) -> dict:
    """Single-call baseline WITHOUT the pipeline's aggressive truncation.

    The deployed agents see ~600 chars/section; the standard single-call
    baseline matches that regime (~8k chars). This control feeds up to ~120k
    chars (~30k tokens) so the null result cannot be attributed to truncation.
    """
    from grantagent.profiles.loader import get_profile
    criteria = ", ".join(get_profile(rec.profile).get("criteria", [])) or "overall quality"
    if isinstance(rec.sections, dict):
        parts = [f"{k.title()}: {v[:40000]}"
                 for k in ("title", "abstract", "intro", "methods", "results",
                           "limitations", "budget")
                 if (v := rec.sections.get(k))]
        blob = "\n\n".join(parts)[:120000]
    else:
        blob = str(rec.sections)[:120000]
    prompt = (
        f"You are reviewing a grant proposal for the {rec.profile} program.\n"
        f"Score the overall proposal on a 0-5 scale, weighing these criteria: {criteria}.\n\n"
        f"Proposal:\n{blob}\n\n"
        'Respond with ONLY a JSON object: {"score_0_5": <float 0-5>}'
    )
    txt = raw_chat([{"role": "user", "content": prompt}], temperature=0.0, seed=42)
    data = _parse_json(txt)
    try:
        score = max(0.0, min(5.0, float(data.get("score_0_5"))))
    except Exception:
        score = 2.5
    verdict = _verdict_from_score(score, rec.profile)
    return {"system": "single_call_fulldoc", "score": score, "verdict": verdict,
            "pred": _pred_from_verdict(verdict), "meta": {"input_chars": len(blob)}}


# ── retrieval-only k-NN (no generation LLM) ────────────────────────────────

def retrieval_knn_fit(records, ctx: Context) -> dict:
    """Leave-one-out k-NN over local embeddings of the eval proposals.

    Tests whether "this proposal looks like funded/declined exemplars" alone
    recovers the funding label — i.e. whether retrieval is a hidden driver.
    Populates ctx._knn_cache: {record_id -> prediction}.  Zero LLM calls.
    """
    from grantagent.vector.embed import embed_texts

    ids = [r.id for r in records]
    labels = [r.label for r in records]
    texts = [_query_text(r.sections) for r in records]
    embs = embed_texts(texts)  # list[list[float]]

    def cos(a, b):
        num = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return num / (na * nb + 1e-9)

    k = ctx.knn_k
    for i, rid in enumerate(ids):
        sims = [(cos(embs[i], embs[j]), labels[j]) for j in range(len(ids)) if j != i]
        sims.sort(key=lambda t: t[0], reverse=True)
        top = sims[:k]
        awarded = sum(1 for _, lab in top if lab == "awarded")
        pred = "awarded" if awarded > len(top) / 2 else "declined"
        # pseudo-score: fraction of awarded neighbours, mapped to 0-5 for gap/AUC
        frac = awarded / max(len(top), 1)
        ctx._knn_cache[rid] = {"system": "retrieval_knn", "score": round(5 * frac, 3),
                               "verdict": None, "pred": pred,
                               "meta": {"neighbours": len(top), "awarded_frac": frac}}
    return ctx._knn_cache


def retrieval_knn(rec, ctx: Context) -> dict:
    if rec.id not in ctx._knn_cache:
        raise RuntimeError("retrieval_knn_fit must be called before retrieval_knn")
    return ctx._knn_cache[rec.id]


# ── full pipeline, reusing the already-computed deployed verdict (0 cost) ───

def full_stored(rec, ctx: Context) -> dict:
    """Score the stored full-system verdict from runs/paper_eval (no recompute).

    This is the actual deployed output on all 455 proposals — the most faithful
    "full system" number, and free.  Use `full` to RE-RUN the pipeline for a
    paired ablation on a subset.
    """
    v = rec.full_verdict or {}
    vd = v.get("verdict", "Misaligned")
    return {"system": "full_stored", "score": v.get("overall_weighted_score"),
            "verdict": vd, "pred": _pred_from_verdict(vd),
            "meta": {"debates_triggered": v.get("debates_triggered", 0),
                     "flags": v.get("flags", [])}}


# ── registry ───────────────────────────────────────────────────────────────

SYSTEMS = {
    "single_call": single_call,
    "multi_criteria_single_call": multi_criteria_single_call,
    "self_consistency": self_consistency,
    "retrieval_knn": retrieval_knn,
    "agents_only": agents_only,
    "debate_only": debate_only,
    "no_retrieval": no_retrieval,
    "no_debate": no_debate,
    "no_meta": no_meta,
    "no_guardrails": no_guardrails,
    "full": full,
    "full_debate_connected": full_debate_connected,
    "single_call_fulldoc": single_call_fulldoc,
    "full_stored": full_stored,
}

# Systems that need a batch fit step before per-record calls.
BATCH_FIT = {"retrieval_knn": retrieval_knn_fit}
