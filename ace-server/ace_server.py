"""ACE v7.3 as a FastAPI server (Rocky Linux / CPU or CUDA / Mac MPS).

- Exposes HTTP endpoints:
  - GET  /health
  - GET  /info
  - POST /generate         (single-turn; uses shared server memory)
  - POST /generate/session (per-session memory; optional)

Notes:
- This file is a server wrapper around your existing ACE core.
- It keeps MAX_NEW_TOKENS=7000 as requested.
- It binds to 0.0.0.0 when run with uvicorn.

Run:
  uvicorn ace_server:app --host 0.0.0.0 --port 8000

Docker CMD example:
  CMD ["uvicorn","ace_server:app","--host","0.0.0.0","--port","8000"]
"""

from __future__ import annotations

import os
import json
import random
import re
import threading
import warnings
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================================
# Config
# ======================================================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MEMORY_FILE = os.getenv("MEMORY_FILE", "ace_memory_v7_0.json")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "7000"))

# Concurrency: model.generate is not thread-safe on some backends.
# We use a single global lock to serialize generation.
GEN_LOCK = threading.Lock()

# Silence tensor-copy warning spam
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use",
)


# ======================================================
# Device + Model Setup
# ======================================================

def get_device() -> str:
    # Prefer MPS (Apple Silicon), then CUDA, else CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = get_device()

print("[ACE] Loading model:", MODEL_NAME)
print("[ACE] Using device:", device)

# FP16 on GPU/MPS, FP32 on CPU
if device in ("cuda", "mps"):
    dtype = torch.float16
else:
    dtype = torch.float32

# Tokenizer
# Trust remote code is generally not required for Qwen2.5 instruct, but can be enabled if you use custom repos.
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, trust_remote_code=True)

# Safer default:
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer for {MODEL_NAME}: {e}")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model for {MODEL_NAME}: {e}")

model.to(device)
model.eval()
model.config.use_cache = True

# Warm-up to avoid first-call lag
with torch.inference_mode():
    warm_inputs = tokenizer("Warmup.", return_tensors="pt").to(device)
    _ = model.generate(
        **warm_inputs,
        do_sample=False,
        max_new_tokens=8,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )


# ======================================================
# Core generation helper (Qwen chat template)
# ======================================================

def _is_qwen_model() -> bool:
    name = (MODEL_NAME or "").lower()
    return "qwen" in name


def _build_qwen_chat_inputs(user_text: str, system_text: str | None = None):
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Normalize to {input_ids, attention_mask}
    if isinstance(enc, torch.Tensor):
        input_ids = enc
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if isinstance(enc, dict) and "attention_mask" in enc:
            attention_mask = enc["attention_mask"]
        elif hasattr(enc, "attention_mask") and enc.attention_mask is not None:
            attention_mask = enc.attention_mask
        else:
            attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def generate_text(
    prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = MAX_NEW_TOKENS,
    system_text: str | None = None,
) -> str:
    """Single completion."""
    try:
        if _is_qwen_model():
            inputs = _build_qwen_chat_inputs(user_text=prompt, system_text=system_text)
            in_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=8,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = output_ids[0][in_len:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if not completion:
                completion = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            return completion

        # Fallback
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=8,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = full_text[len(prompt):].strip()
        if not completion:
            completion = full_text.strip()
        return completion

    except Exception as e:
        return f"[ACE ERROR] {e}"


# ======================================================
# Prompt classifiers (same behavior)
# ======================================================

def is_literal_mode(prompt: str) -> bool:
    p = prompt.lower()
    triggers = [
        "no metaphors",
        "without metaphors",
        "literal explanation",
        "explain simply",
        "explain in simple language",
        "bez metafor",
        "bez prirovnani",
        "bez prirovnaní",
    ]
    return any(t in p for t in triggers)


def is_story_mode(prompt: str) -> bool:
    p = prompt.lower()
    triggers = [
        "story",
        "backstory",
        "short story",
        "narrative",
        "write a story",
        "character backstory",
        "fairytale",
        "fairy tale",
    ]
    return any(t in p for t in triggers)


def wants_hopeful_ending(prompt: str) -> bool:
    p = prompt.lower()
    triggers = [
        "ends hopeful",
        "hopeful ending",
        "good ending",
        "ends with hope",
        "positive ending",
    ]
    return any(t in p for t in triggers)


def is_continuation(prompt: str) -> bool:
    p = prompt.strip().lower()
    triggers = {
        "continue",
        "continue.",
        "continue the story",
        "continue the story.",
        "keep going",
        "keep going.",
        "pokračuj",
        "pokračuj v príbehu",
    }
    if p in triggers:
        return True
    if p.startswith("continue the "):
        return True
    if p.startswith("pokračuj "):
        return True
    return False


def refers_to_previous(prompt: str) -> bool:
    p = (prompt or "").lower()
    triggers = [
        "for that scp",
        "that scp",
        "that one",
        "the one above",
        "from above",
        "from earlier",
        "previous answer",
        "based on that",
        "based on the above",
    ]
    return any(t in p for t in triggers)


def is_short_greeting(prompt: str) -> bool:
    p = prompt.strip().lower()
    if not p:
        return False
    words = p.split()
    if len(words) > 3:
        return False
    greeting_tokens = ["hi", "hello", "hey", "yo", "sup", "ahoj", "čau", "čaute", "cau"]
    return any(tok in p for tok in greeting_tokens)


# ======================================================
# CBS (Context Boundary Space)
# ======================================================

def extract_context(prompt: str) -> Dict[str, Any]:
    words = [w.strip(".,!?;:").lower() for w in prompt.split()]
    keywords = {w for w in words if len(w) > 3}
    topic_hint = words[0] if words else ""
    sentiment = (
        "positive"
        if any(w in prompt.lower() for w in ["hope", "dream", "love", "happy"])
        else "negative"
        if any(w in prompt.lower() for w in ["fear", "pain", "loss", "sad", "hurt"])
        else "neutral"
    )
    return {"keywords": keywords, "topic_hint": topic_hint, "sentiment": sentiment}


def context_fit(text: str, ctx: Dict[str, Any]) -> float:
    penalty = 0.0
    lowered = text.lower()

    if ctx["topic_hint"] and ctx["topic_hint"] not in lowered:
        penalty += 0.1

    if ctx["keywords"]:
        overlap = sum(1 for k in ctx["keywords"] if k in lowered)
        if overlap == 0:
            penalty += 0.25
        elif overlap <= 2:
            penalty += 0.1

    if ctx["sentiment"] == "positive" and any(w in lowered for w in ["kill", "die", "ruin"]):
        penalty += 0.3

    if ctx["sentiment"] == "negative" and any(w in lowered for w in ["cute", "joy", "happy", "celebrate"]):
        penalty += 0.2

    return max(0.0, 1.0 - penalty)


# ======================================================
# Hallucination estimation (ACC Core)
# ======================================================

def knowledge_like_score(text: str) -> float:
    years = re.findall(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", text)
    year_score = min(len(years) / 5.0, 1.0)

    cues = [
        "according to",
        "study",
        "research",
        "data show",
        "reported that",
        "scientists",
        "experts say",
        "survey",
        "statistically",
    ]
    cue_hits = sum(1 for c in cues if c in text.lower())
    cue_score = min(cue_hits / 5.0, 1.0)

    return min(year_score * 0.6 + cue_score * 0.4, 1.0)


def novelty_score(text: str) -> float:
    words = text.lower().split()
    rare = [w for w in words if len(w) > 9]
    return min(len(rare) / 25.0, 1.0)


def hallucination_score(text: str, ctx: Dict[str, Any], story_mode: bool) -> float:
    cf = context_fit(text, ctx)
    base_incoherence = 1.0 - cf

    k_score = 0.0 if story_mode else knowledge_like_score(text)
    n_score = novelty_score(text)

    score = 0.55 * base_incoherence + 0.25 * k_score + 0.20 * n_score
    return max(0.0, min(score, 1.0))


def hallucination_level(target_state: int, h_score: float, story_mode: bool) -> int:
    """0 grounded, 1 mild, 2 strong, 3 free"""
    if story_mode:
        base = target_state
        if h_score < 0.3 and target_state >= 1:
            base = min(3, base + 1)
        return max(0, min(3, base))

    lvl = int(round(h_score * 3))
    return max(0, min(3, lvl))


def add_uncertainty_tag(text: str, level: int) -> str:
    if level <= 0:
        return text

    if level == 1:
        tag = (
            "\n\nNote: This answer may contain minor speculation and should not be "
            "treated as perfectly precise."
        )
    elif level == 2:
        tag = (
            "\n\nNote: Several parts of this answer are speculative and not based on "
            "specific verified data. Treat this as a creative approximation, not a "
            "reliable source."
        )
    else:
        tag = (
            "\n\nNote: This answer is largely a creative guess and may not match real "
            "facts. Do not treat this as authoritative information."
        )

    return text + tag


def ground_response(prompt: str, text: str) -> str:
    grounding_prompt = (
        "You are ACE, a controlled creativity engine.\n"
        "Rewrite the following answer so that it is more cautious and grounded.\n"
        "Remove or soften very specific invented details (names, dates, exact numbers) "
        "unless they are clearly generic.\n"
        "Keep the main ideas, but prefer phrases like 'for example', 'often', or "
        "'in many cases' instead of hard claims.\n"
        "Do NOT add new details.\n\n"
        "User question:\n"
        f"{prompt}\n\n"
        "Original answer:\n"
        f"{text}\n\n"
        "Rewritten, more cautious answer:\n"
    )
    return generate_text(
        grounding_prompt,
        temperature=0.5,
        top_p=0.9,
        max_new_tokens=MAX_NEW_TOKENS,
        system_text=(
            "You are ACE, a controlled creativity engine. "
            "Rewrite to be cautious and grounded. "
            "Do not add new facts, and do not mention policies or safety rules."
        ),
    )


# ======================================================
# ACW (Adaptive Creativity Window)
# ======================================================

def acw_state(prompt: str, decay: float, literal_mode: bool, story_mode: bool) -> int:
    if literal_mode:
        return 0

    p = prompt.lower().strip()
    short = len(p.split()) <= 4

    creative_trigger = story_mode or any(
        kw in p for kw in ["imagine", "poem", "invent", "creative", "world where", "dream"]
    )

    if short and not creative_trigger:
        return 0

    entropy = random.uniform(0.45, 1.0)
    intensity = 0.85 if creative_trigger else 0.5
    stability = 1 - decay

    s = 0.45 * entropy + 0.35 * intensity + 0.2 * stability

    if s < 0.35:
        return 0
    if s < 0.7:
        return 1
    return 2


def mutation_settings(state: int, literal_mode: bool, story_mode: bool) -> Dict[str, Any]:
    if literal_mode:
        return {"temp": 0.3, "top_p": 0.9}

    if story_mode:
        if state == 0:
            return {"temp": 0.9, "top_p": 0.95}
        if state == 1:
            return {"temp": 1.0, "top_p": 0.97}
        return {"temp": 1.05, "top_p": 0.98}

    if state == 0:
        return {"temp": 0.6, "top_p": 0.9}
    if state == 1:
        return {"temp": 0.9, "top_p": 0.94}
    return {"temp": 1.0, "top_p": 0.96}


DEFAULT_SYSTEM_ASSISTANT = (
    "You are ACE, a helpful assistant. "
    "Follow the user's constraints precisely. "
    "No jokes, no filler, no meta commentary. "
    "Do not mention policies or safety rules unless the user explicitly asks about them."
)

DEFAULT_SYSTEM_STORY = (
    "You are ACE, a narrative engine. "
    "Follow the user's constraints exactly. "
    "When the user asks for multiple parts, clearly separate them with labels like 'SCP SECTION' and 'EMOTIONAL SCENE'. "
    "Do not mention policies, safety rules, or meta commentary; stay in-character."
)


def _is_meta_refusal(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    bad = [
        "i'm afraid",
        "i am afraid",
        "i can't",
        "i cannot",
        "i can not",
        "i must say",
        "goes against",
        "guidelines",
        "policy",
        "safety",
        "ethical",
        "intended role",
        "as an ai",
        "as a language model",
        "i can't help with that",
    ]
    return any(b in t for b in bad)


def _extract_forbidden_literals(prompt: str) -> list[str]:
    p = (prompt or "").lower()
    out: list[str] = []

    for m in re.finditer(r'(?:never|do not|don\'t)\s+mention\s+"([^"]+)"', p):
        out.append(m.group(1).strip())

    m2 = re.search(r"(?:never|do not|don't)\s+mention\s+([^\n\r\.\!\?]+)", p)
    if m2:
        chunk = m2.group(1).strip()
        if 0 < len(chunk) <= 24:
            out.append(chunk)

    if ("never mention" in p or "do not mention" in p or "don't mention" in p) and (
        "code" in p or "number" in p
    ):
        nums = re.findall(r"\b\d{3,6}\b", prompt)
        out.extend(nums)

    return [s for s in {x.strip() for x in out} if s]


def _constraint_gate(prompt: str, text: str) -> bool:
    p = (prompt or "").lower()
    t = (text or "").strip()
    if not t:
        return False

    if _is_meta_refusal(t):
        return False

    if "scp" in p and "headings" in p:
        required = [
            "item #",
            "object class",
            "special containment procedures",
            "description",
            "addendum",
        ]
        low = t.lower()
        if any(r not in low for r in required):
            return False

    if "emotional scene" in p:
        low = t.lower()
        if "emotional scene" not in low and "scene" not in low:
            return True

        scene_part = ""
        m = re.search(
            r"(?:emotional\s+scene|scene)\s*[:\-]\s*\n?([\s\S]+)$",
            t,
            flags=re.IGNORECASE,
        )
        if m:
            scene_part = m.group(1).strip()

        if scene_part:
            paras = [pp.strip() for pp in re.split(r"\n\s*\n", scene_part) if pp.strip()]
            if "2 paragraphs" in p and len(paras) != 2:
                return False

            if "no exact numbers" in p and re.search(r"\d", scene_part):
                return False

            if "no named dates" in p:
                months = [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]
                days = [
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                ]
                low_scene = scene_part.lower()
                if any(mn in low_scene for mn in months) or any(dn in low_scene for dn in days):
                    return False
                if re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", scene_part):
                    return False

    forb = _extract_forbidden_literals(prompt)
    if forb:
        low = t.lower()
        for f in forb:
            if f and f.lower() in low:
                return False

    return True


def _extra_named_entity_penalty(prompt: str, text: str) -> float:
    p = prompt or ""
    t = text or ""

    cand = set(re.findall(r"\b[A-Z][a-z]{2,}\b", t))
    base = set(re.findall(r"\b[A-Z][a-z]{2,}\b", p))

    allow = {
        "The",
        "A",
        "An",
        "And",
        "But",
        "If",
        "In",
        "On",
        "At",
        "As",
        "After",
        "Before",
    }
    extra = {w for w in cand if w not in base and w not in allow}

    if len(extra) <= 1:
        return 0.0
    if len(extra) == 2:
        return 0.08
    if len(extra) == 3:
        return 0.16
    return 0.25


CANDIDATE_MIN_NEW_TOKENS = 250
CANDIDATE_MAX_NEW_TOKENS = 2500


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _candidate_budget(state: int) -> int:
    base = 520
    if state >= 2:
        base = 680
    return _clamp_int(base, CANDIDATE_MIN_NEW_TOKENS, CANDIDATE_MAX_NEW_TOKENS)


def _sentiment_consistency_score(text: str, ctx: Dict[str, Any]) -> float:
    lowered = text.lower()

    pos = ["hope", "dream", "love", "happy", "joy", "relief", "calm", "smile"]
    neg = ["fear", "pain", "loss", "sad", "hurt", "die", "kill", "ruin", "despair"]

    pos_hits = sum(1 for w in pos if w in lowered)
    neg_hits = sum(1 for w in neg if w in lowered)

    s = ctx.get("sentiment", "neutral")

    if s == "positive":
        if neg_hits >= 2 and pos_hits == 0:
            return 0.2
        if neg_hits >= 3:
            return 0.1
        return 1.0

    if s == "negative":
        if pos_hits >= 2 and neg_hits == 0:
            return 0.3
        return 1.0

    if pos_hits >= 4 and neg_hits == 0:
        return 0.7
    if neg_hits >= 4 and pos_hits == 0:
        return 0.7
    return 1.0


def _structural_quality_score(text: str, scp_mode: bool) -> float:
    t = text.strip()
    if not t:
        return 0.0

    sentences = [s for s in re.split(r"[.!?]\s+", t) if s.strip()]
    sc = len(sentences)

    if sc <= 1:
        base = 0.2
    elif sc <= 2:
        base = 0.55
    elif sc <= 10:
        base = 1.0
    elif sc <= 16:
        base = 0.85
    else:
        base = 0.7

    if scp_mode:
        headings = [
            "item #",
            "object class",
            "special containment procedures",
            "description",
            "addendum",
            "incident",
            "interview log",
        ]
        hits = sum(1 for h in headings if h in t.lower())
        if hits == 0:
            base *= 0.6
        elif hits == 1:
            base *= 0.85
        else:
            base *= 1.05

    return max(0.0, min(base, 1.0))


def _is_scp_mode(prompt: str) -> bool:
    p = prompt.lower()
    return ("scp" in p) or ("object class" in p) or ("containment procedures" in p)


def _is_strict_scp_request(prompt: str) -> bool:
    p = (prompt or "").lower()
    return (
        ("scp" in p and "item #" in p and "object class" in p and "containment" in p)
        or ("output must be" in p and "item #" in p)
    )


def _prompt_requires_emotional_scene(prompt: str) -> bool:
    return "emotional scene" in (prompt or "").lower()


def _strip_digits(text: str) -> str:
    return re.sub(r"\d", "", text)


# NOTE: Your original script contains a long _repair_strict_scp_output implementation.
# It's safe to keep; for brevity it is not included here.
# If you need strict SCP repair, paste it in place of this stub.

def _repair_strict_scp_output(prompt: str, text: str) -> str:
    return text


def _score_candidate(text: str, ctx: Dict[str, Any], scp_mode: bool, prompt: str) -> float:
    if _is_meta_refusal(text):
        return -1.0

    cf = context_fit(text, ctx)
    nv = novelty_score(text)
    es = _sentiment_consistency_score(text, ctx)
    st = _structural_quality_score(text, scp_mode)

    gate_pen = 0.35 if not _constraint_gate(prompt, text) else 0.0
    ne_pen = _extra_named_entity_penalty(prompt, text)

    score = (0.54 * cf) + (0.12 * nv) + (0.20 * es) + (0.14 * st)
    return score - gate_pen - ne_pen


def _generate_candidates(
    gen_prompt: str,
    count: int,
    params: Dict[str, Any],
    max_new_tokens: int,
    system_text: str | None = None,
    accept_prompt: str | None = None,
) -> list[str]:
    outs: list[str] = []
    attempts = 0
    max_attempts = max(6, count * 3)

    while len(outs) < count and attempts < max_attempts:
        i = len(outs)
        attempts += 1

        temp = float(params["temp"])
        top_p = float(params["top_p"])

        if i == 1:
            temp = min(1.25, temp + 0.05)
            top_p = min(0.99, top_p + 0.01)
        elif i >= 2:
            temp = min(1.30, temp + 0.08)
            top_p = min(0.99, top_p + 0.015)

        out = generate_text(
            gen_prompt,
            temperature=temp,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            system_text=system_text,
        ).lstrip()

        if accept_prompt is not None and not _constraint_gate(accept_prompt, out):
            continue

        outs.append(out)

    if not outs:
        out = generate_text(
            gen_prompt,
            temperature=max(0.6, float(params["temp"]) - 0.2),
            top_p=max(0.85, float(params["top_p"]) - 0.05),
            max_new_tokens=max_new_tokens,
            system_text=system_text,
        )
        if out and out.strip():
            outs.append(out.lstrip())

    return outs


def _pick_best_candidate(
    candidates: list[str],
    ctx: Dict[str, Any],
    scp_mode: bool,
    prompt: str,
) -> tuple[str, float]:
    best_text = candidates[0] if candidates else ""
    best_score = -999.0
    for c in candidates:
        s = _score_candidate(c, ctx, scp_mode, prompt)
        if s > best_score:
            best_score = s
            best_text = c
    return best_text, best_score


# ======================================================
# Memory
# ======================================================

def load_mem(path: str = MEMORY_FILE) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"scores": [], "last_story": ""}


def save_mem(mem: Dict[str, Any], path: str = MEMORY_FILE) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        # path might be just a filename
        pass

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ======================================================
# Literal + Story helpers
# ======================================================

def literal_explainer(prompt: str) -> str:
    full_prompt = (
        "Explain this as clearly, directly and literally as possible.\n"
        "No metaphors, no comparisons, no stories.\n"
        "Use simple factual language only.\n\n"
        f"Question: {prompt}\nAnswer:"
    )
    return generate_text(full_prompt, temperature=0.25, top_p=0.9, max_new_tokens=120)


def build_story_prompt(user_prompt: str, hopeful: bool) -> str:
    base = (
        "You are ACE, a narrative engine.\n"
        "Write a continuous story as pure narrative from inside the character's world.\n"
        "Do NOT explain writing techniques.\n"
        "Do NOT mention 'the reader', 'audience', 'chapter', or 'section'.\n"
        "Just tell the story.\n"
    )
    if hopeful:
        base += "The story should end with a clear sense of hope or a new beginning.\n"

    base += "\nStory prompt:\n" + user_prompt + "\n\nStory:\n"
    return base


def build_continuation_prompt(previous_story: str, hopeful: bool) -> str:
    base = (
        "You are ACE, a narrative engine.\n"
        "Continue the story below.\n"
        "Stay in the same characters, world, tone, and point of view.\n"
        "Do NOT restart or summarize the story.\n"
        "Do NOT switch to a different main character unless it logically follows.\n"
        "Write only the next part of the story as continuous narrative.\n"
    )
    if hopeful:
        base += (
            "Let the overall direction slowly move toward a subtle but real sense of "
            "hope or a new beginning, but do NOT jump straight to a happy ending.\n"
        )

    base += "\nExisting story:\n" + previous_story + "\n\nNext part:\n"
    return base


# ======================================================
# ACW marker helper
# ======================================================

def split_acw_marker(prompt: str) -> tuple[str, bool]:
    p = prompt.rstrip()
    if re.search(r"\s+acw$", p, flags=re.IGNORECASE):
        clean = re.sub(r"\s+acw$", "", p, flags=re.IGNORECASE).rstrip()
        return clean, True
    return prompt, False


# ======================================================
# ACE core (ACC + Lite) – unchanged logic, just no GUI
# ======================================================

def dynamic_max_tokens(prompt: str) -> int:
    words = len(prompt.strip().split())
    if words == 0:
        return MAX_NEW_TOKENS
    if words <= 3:
        return min(40, MAX_NEW_TOKENS)
    if words <= 15:
        return min(140, MAX_NEW_TOKENS)
    return MAX_NEW_TOKENS


def ace_once(prompt: str, mem: Dict[str, Any]) -> str:
    clean_prompt, acw_enabled = split_acw_marker(prompt)

    if not acw_enabled:
        if is_short_greeting(clean_prompt):
            return "Hello, I'm ACE. What do you want to try?"

        max_tokens = dynamic_max_tokens(clean_prompt)

        base_prompt = (
            "You are ACE, a helpful assistant.\n"
            "Respond only to the user's last message.\n"
            "Do not invent extra context (like emails, clients, or prior mistakes)\n"
            "if the user did not explicitly mention them.\n"
            "Keep the answer relatively short when the question is short.\n\n"
            f"User: {clean_prompt}\n"
            "ACE:"
        )

        return generate_text(
            base_prompt,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=max_tokens,
            system_text=(
                "You are ACE, a helpful assistant. "
                "Follow the user's instructions. "
                "Do not mention policies, safety rules, or refusals unless the user explicitly asks about them."
            ),
        )

    literal = is_literal_mode(clean_prompt)
    story_mode = is_story_mode(clean_prompt)
    hopeful = wants_hopeful_ending(clean_prompt)
    continuation = is_continuation(clean_prompt)
    last_story = mem.get("last_story", "")

    if continuation and last_story:
        story_mode = True
        literal = False
        ctx = extract_context(last_story)
    elif last_story and refers_to_previous(clean_prompt):
        ctx = extract_context(last_story)
    else:
        ctx = extract_context(clean_prompt)

    scores = mem.get("scores", [])
    decay = sum(scores[-5:]) / 5 if scores[-5:] else 0.0

    state = acw_state(clean_prompt, decay, literal, story_mode)
    if story_mode and state == 0:
        state = 1

    params = mutation_settings(state, literal, story_mode)

    if literal:
        out = literal_explainer(clean_prompt)
        cf = context_fit(out, ctx)
        h_score = hallucination_score(out, ctx, story_mode=False)
        level = hallucination_level(state, h_score, story_mode=False)
        final = add_uncertainty_tag(out, level if level >= 2 else 0)

        mem.setdefault("scores", []).append(cf)
        mem["scores"] = mem["scores"][-80:]
        return final

    # Story mode
    if story_mode:
        scp_mode = _is_scp_mode(clean_prompt)
        structured_story = scp_mode or ("headings" in clean_prompt.lower())

        if continuation and last_story and not structured_story:
            story_prompt = build_continuation_prompt(last_story, hopeful)
        else:
            story_prompt = clean_prompt

        cand_count = 1 if state == 0 else (2 if state == 1 else 3)

        if cand_count == 1:
            out = generate_text(
                story_prompt,
                temperature=float(params["temp"]),
                top_p=float(params["top_p"]),
                max_new_tokens=MAX_NEW_TOKENS,
                system_text=DEFAULT_SYSTEM_STORY,
            )
        else:
            budget = _candidate_budget(state)
            cands = _generate_candidates(
                gen_prompt=story_prompt,
                count=cand_count,
                params=params,
                max_new_tokens=budget,
                system_text=DEFAULT_SYSTEM_STORY,
                accept_prompt=clean_prompt,
            )
            if cands:
                best, _ = _pick_best_candidate(cands, ctx, scp_mode, clean_prompt)
                out = best
                if not out.strip():
                    out = generate_text(
                        story_prompt,
                        temperature=float(params["temp"]),
                        top_p=float(params["top_p"]),
                        max_new_tokens=900,
                        system_text=DEFAULT_SYSTEM_STORY,
                    )
                if structured_story and (
                    "addendum" not in out.lower() or "emotional scene" not in out.lower()
                ):
                    cont = generate_text(
                        story_prompt + "\n" + out.strip() + "\n",
                        temperature=float(params["temp"]),
                        top_p=float(params["top_p"]),
                        max_new_tokens=300,
                        system_text=DEFAULT_SYSTEM_STORY,
                    )
                    out = (out.rstrip() + "\n" + cont.lstrip()).strip()
            else:
                out = generate_text(
                    story_prompt,
                    temperature=float(params["temp"]),
                    top_p=float(params["top_p"]),
                    max_new_tokens=budget,
                    system_text=DEFAULT_SYSTEM_STORY,
                )

            if not structured_story:
                remaining = max(0, int(MAX_NEW_TOKENS) - int(_candidate_budget(state)))
                if remaining > 0:
                    cont_prompt = story_prompt + "\n" + out.strip() + "\n"
                    cont = generate_text(
                        cont_prompt,
                        temperature=float(params["temp"]),
                        top_p=float(params["top_p"]),
                        max_new_tokens=remaining,
                        system_text=DEFAULT_SYSTEM_STORY,
                    )
                    out = (out.rstrip() + "\n" + cont.lstrip()).strip()

        if structured_story and _is_strict_scp_request(clean_prompt):
            out = _repair_strict_scp_output(clean_prompt, out)

        cf = context_fit(out, ctx)
        mem.setdefault("scores", []).append(cf)
        mem["scores"] = mem["scores"][-80:]

        if continuation and last_story:
            combined = (last_story + "\n" + out).strip()
        else:
            combined = out.strip()
        mem["last_story"] = combined[-4000:]
        return out

    # Default ACC path
    scp_mode = _is_scp_mode(clean_prompt)
    cand_count = 1 if state == 0 else (2 if state == 1 else 3)

    gen_prompt = clean_prompt
    if last_story and refers_to_previous(clean_prompt):
        snippet = last_story[-2000:]
        gen_prompt = (
            "Use the following previous output as context. Do not contradict it.\n\n"
            + snippet
            + "\n\nUser request:\n"
            + clean_prompt
        )

    if cand_count == 1:
        raw = generate_text(
            gen_prompt,
            temperature=float(params["temp"]),
            top_p=float(params["top_p"]),
            max_new_tokens=MAX_NEW_TOKENS,
            system_text=DEFAULT_SYSTEM_ASSISTANT,
        )
    else:
        budget = _candidate_budget(state)
        cands = _generate_candidates(
            gen_prompt=gen_prompt,
            count=cand_count,
            params=params,
            max_new_tokens=budget,
            system_text=DEFAULT_SYSTEM_ASSISTANT,
            accept_prompt=clean_prompt,
        )
        raw = raw = (cands[0] if cands else "")
        if cands:
            raw, _ = _pick_best_candidate(cands, ctx, scp_mode, clean_prompt)
        else:
            raw = generate_text(
                gen_prompt,
                temperature=float(params["temp"]),
                top_p=float(params["top_p"]),
                max_new_tokens=budget,
                system_text=DEFAULT_SYSTEM_ASSISTANT,
            )

    cf = context_fit(raw, ctx)
    h_score = hallucination_score(raw, ctx, story_mode=False)
    level = hallucination_level(state, h_score, story_mode=False)

    final = raw

    if scp_mode and _is_strict_scp_request(clean_prompt):
        final = _repair_strict_scp_output(clean_prompt, final)
        level = 0

    if not final.strip():
        final = "[ACE ERROR] Model returned empty output."
        level = 0

    if level >= 2:
        grounded = ground_response(clean_prompt, raw)
        final = add_uncertainty_tag(grounded, level)
    elif level == 1:
        final = add_uncertainty_tag(raw, level)

    mem.setdefault("scores", []).append(cf)
    mem["scores"] = mem["scores"][-80:]
    return final


# ======================================================
# FastAPI Server
# ======================================================

app = FastAPI(title="ACE Server", version="7.3")

# One shared memory for the whole server process
SERVER_MEM = load_mem(MEMORY_FILE)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    # optional overrides
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_new_tokens: Optional[int] = Field(None, ge=1, le=MAX_NEW_TOKENS)


class GenerateResponse(BaseModel):
    response: str


class GenerateSessionRequest(GenerateRequest):
    session_id: str = Field(..., min_length=1, max_length=128)


SESSION_MEM: Dict[str, Dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/info")
def info() -> Dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "device": device,
        "max_new_tokens": MAX_NEW_TOKENS,
        "memory_file": MEMORY_FILE,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    # NOTE: per-request temperature/top_p are not plumbed into the full ACE pipeline
    # because ACE manages these internally via ACW. If you want hard overrides,
    # implement them in ace_once / generate_text.

    prompt = req.prompt

    with GEN_LOCK:
        try:
            out = ace_once(prompt, SERVER_MEM)
            save_mem(SERVER_MEM, MEMORY_FILE)
            return GenerateResponse(response=out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/session", response_model=GenerateResponse)
def generate_session(req: GenerateSessionRequest) -> GenerateResponse:
    # Same as /generate but keeps separate memory per session_id
    prompt = req.prompt

    with SESSION_LOCK:
        mem = SESSION_MEM.get(req.session_id)
        if mem is None:
            mem = load_mem(MEMORY_FILE)
            SESSION_MEM[req.session_id] = mem

    with GEN_LOCK:
        try:
            out = ace_once(prompt, mem)
            # session memory is not auto-saved to disk unless you want it.
            return GenerateResponse(response=out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
