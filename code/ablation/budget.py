"""Compute accounting for the ablation study.

The paper's central question is whether the multi-agent *design* helps, or whether
any gain is just a consequence of spending more LLM compute.  To make that argument
we must measure compute, not assume it.  This module records, per proposal and per
system, the number of LLM calls and the number of tokens consumed.

- Call counts are exact (every entry point is instrumented).
- Token counts are exact when the provider returns usage (OpenAI does, via
  ``response.usage.total_tokens``); otherwise we fall back to a chars/4 estimate.

Usage:
    from .budget import METER, meter_llm, raw_chat
    with meter_llm():                 # patches llm_service.chat / chat_messages
        ... run a system ...
    calls, tokens = METER.snapshot()  # deltas are taken by the caller
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


def _est_tokens(*texts: str) -> int:
    return sum(len(t or "") for t in texts) // 4


@dataclass
class Meter:
    calls: int = 0
    tokens: int = 0
    exact_tokens: bool = True  # flips False if any call had to be estimated

    def add(self, *, calls: int = 1, tokens: int = 0, exact: bool = True) -> None:
        self.calls += calls
        self.tokens += tokens
        if not exact:
            self.exact_tokens = False

    def snapshot(self) -> tuple[int, int]:
        return self.calls, self.tokens

    def reset(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.exact_tokens = True


# Single process-wide meter shared by the patched service methods and raw_chat.
METER = Meter()


@contextlib.contextmanager
def meter_llm():
    """Patch the llm_service singleton so every chat/chat_messages call is counted.

    Wraps rather than replaces, so provider logic, retries and seeds are untouched.
    """
    from grantagent.service.llm import llm_service

    orig_chat = llm_service.chat
    orig_chat_messages = llm_service.chat_messages

    def chat(prompt, system=None):
        out = orig_chat(prompt, system=system)
        METER.add(calls=1, tokens=_est_tokens(prompt, system or "", out), exact=False)
        return out

    def chat_messages(messages, max_tokens=None, temperature=0.3):
        out = orig_chat_messages(messages, max_tokens=max_tokens, temperature=temperature)
        prompt_txt = "".join(m.get("content", "") for m in messages)
        METER.add(calls=1, tokens=_est_tokens(prompt_txt, out), exact=False)
        return out

    llm_service.chat = chat  # type: ignore[assignment]
    llm_service.chat_messages = chat_messages  # type: ignore[assignment]
    try:
        yield METER
    finally:
        llm_service.chat = orig_chat  # type: ignore[assignment]
        llm_service.chat_messages = orig_chat_messages  # type: ignore[assignment]


def raw_chat(messages, *, temperature: float = 0.0, seed: int | None = 42,
             max_tokens: int | None = None) -> str:
    """Direct provider call with explicit temperature/seed, counted in METER.

    Used by the single-call and self-consistency baselines, which need controllable
    sampling (the service hard-codes seed=42, which would collapse self-consistency).
    Returns assistant text; captures exact token usage when the provider reports it.
    """
    from grantagent.service.llm import llm_service

    if not llm_service.available():
        return ""
    mt = max_tokens or llm_service.max_tokens

    if llm_service.provider == "anthropic":
        import anthropic  # type: ignore
        import os

        system_blocks = [m["content"] for m in messages if m.get("role") == "system"]
        convo = [m for m in messages if m.get("role") != "system"]
        kwargs = {"model": llm_service.model, "max_tokens": mt,
                  "messages": convo, "temperature": temperature}
        if system_blocks:
            kwargs["system"] = "\n\n".join(system_blocks)
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        try:
            try:
                resp = client.messages.create(**kwargs)
            except Exception as exc:
                # Thinking-enabled models (e.g. fable-5) reject temperature != 1;
                # retry without the sampling override rather than failing silently.
                if "temperature" not in str(exc).lower():
                    raise
                kwargs.pop("temperature", None)
                resp = client.messages.create(**kwargs)
            blocks = resp.content or []
            text = "".join(b.text for b in blocks if getattr(b, "type", None) == "text")
            usage = getattr(resp, "usage", None)
            if usage is not None:
                tok = int(getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0))
                METER.add(calls=1, tokens=tok, exact=True)
            else:
                METER.add(calls=1, tokens=_est_tokens(*(m["content"] for m in messages), text), exact=False)
            return text
        except Exception:
            METER.add(calls=1, tokens=0, exact=False)
            return ""

    # OpenAI
    try:
        kwargs = {"model": llm_service.model, "messages": messages,
                  "max_tokens": mt, "temperature": temperature}
        if seed is not None:
            kwargs["seed"] = seed
        try:
            resp = llm_service.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Reasoning-model compat (gpt-5*/o*-series): retry once with
            # max_completion_tokens (reasoning headroom) and no temperature/seed.
            from grantagent.service.llm import _needs_completion_tokens_param
            if not _needs_completion_tokens_param(exc):
                raise
            resp = llm_service.client.chat.completions.create(
                model=llm_service.model, messages=messages,
                max_completion_tokens=max(mt, 4000))
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tok = getattr(usage, "total_tokens", None)
        if tok is not None:
            METER.add(calls=1, tokens=int(tok), exact=True)
        else:
            METER.add(calls=1, tokens=_est_tokens(*(m.get("content", "") for m in messages), text), exact=False)
        return text
    except Exception:
        METER.add(calls=1, tokens=0, exact=False)
        return ""
