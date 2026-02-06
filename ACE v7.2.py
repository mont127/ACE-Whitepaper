import os
import json
import random
import re
import warnings
import threading
import tkinter as tk
from tkinter import scrolledtext
from typing import Dict, Any

from numpy import info
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM




# MADE BY ILOVELEEP OR SOAPERCATZ OR TS https://huggingface.co/Soaperloafidksum
# ======================================================
# ACE v7.2 — ACC Core + Lite-Core-Network (Terminal, GPU/MPS)
# ------------------------------------------------------
# - No Gradio, pure terminal I/O
# - Local microsoft/Phi-3-mini-4k-instruct
# - Detects Apple Silicon (MPS) / CUDA / CPU
# - FP16 on GPU/MPS, FP32 on CPU
# - Warm-up pass to avoid first-call lag
# - inference_mode + use_cache for faster generation
# - ACC: controlled hallucinations with self-tagging and grounding
# ======================================================

#MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MEMORY_FILE = "ace_memory_v7_0.json"
MAX_NEW_TOKENS = 7000  # balance: speed vs narrative length
TRANSFORMERS_VERBOSITY=info
# Silence tensor-copy warning spam
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use"
)


# ======================================================
# Device + Model Setup (optimized for Mac/GPU)
# ======================================================

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = get_device()

print("[ACE] Loading model:", MODEL_NAME)
print("[ACE] Using device:", device)

if device in ("cuda", "mps"):
    dtype = torch.float16
else:
    dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
)
model.to(device)
model.eval()
model.config.use_cache = True

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
# Core generation helper
# ======================================================

def _is_qwen_model() -> bool:
    name = (MODEL_NAME or "").lower()
    return "qwen" in name


def _build_qwen_chat_inputs(user_text: str, system_text: str | None = None):
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    # Qwen instruct models behave best via chat template.
    # apply_chat_template may return either a Tensor (some tokenizers) OR a BatchEncoding.
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Normalize to {input_ids, attention_mask} with tensors.
    if isinstance(enc, torch.Tensor):
        input_ids = enc
        attention_mask = torch.ones_like(input_ids)
    else:
        # BatchEncoding-like
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
    try:
        # Qwen path: stable chat template + decode only new tokens
        if _is_qwen_model():
            inputs = _build_qwen_chat_inputs(
                user_text=prompt,
                system_text=system_text,
            )
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

        # Non-Qwen fallback: old behavior
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
# Prompt classifiers
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


# --------------------------------------
# Detects if a prompt refers to previous output (for follow-ups)
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

# --------------------------------------
# Helper: Detect short greetings
def is_short_greeting(prompt: str) -> bool:
    """
    Detect tiny greeting-like prompts where we just want a short hello back,
    not a long rambling answer.
    """
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

    if ctx["sentiment"] == "positive" and any(
        w in lowered for w in ["kill", "die", "ruin"]
    ):
        penalty += 0.3

    if ctx["sentiment"] == "negative" and any(
        w in lowered for w in ["cute", "joy", "happy", "celebrate"]
    ):
        penalty += 0.2

    return max(0.0, 1.0 - penalty)


# ======================================================
# Hallucination estimation (ACC Core)
# ======================================================

def knowledge_like_score(text: str) -> float:
    t = text
    years = re.findall(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", t)
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
    cue_hits = sum(1 for c in cues if c in t.lower())
    cue_score = min(cue_hits / 5.0, 1.0)

    return min(year_score * 0.6 + cue_score * 0.4, 1.0)


def novelty_score(text: str) -> float:
    words = text.lower().split()
    rare = [w for w in words if len(w) > 9]
    return min(len(rare) / 25.0, 1.0)


def hallucination_score(text: str, ctx: Dict[str, Any], story_mode: bool) -> float:
    cf = context_fit(text, ctx)
    base_incoherence = 1.0 - cf

    if story_mode:
        k_score = 0.0
    else:
        k_score = knowledge_like_score(text)

    n_score = novelty_score(text)

    score = (
        0.55 * base_incoherence +
        0.25 * k_score +
        0.20 * n_score
    )
    return max(0.0, min(score, 1.0))


def hallucination_level(target_state: int, h_score: float, story_mode: bool) -> int:
    """
    0 = very grounded
    1 = mild speculation
    2 = strong speculation
    3 = free hallucination
    """
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
    """
    Return 0 = low, 1 = mid, 2 = high creativity.
    ACC will then turn this into hallucination levels.
    """
    if literal_mode:
        return 0

    p = prompt.lower().strip()
    tokens = p.split()
    short = len(tokens) <= 4

    creative_trigger = story_mode or any(
        kw in p
        for kw in ["imagine", "poem", "invent", "creative", "world where", "dream"]
    )

    if short and not creative_trigger:
        return 0

    entropy = random.uniform(0.45, 1.0)
    intensity = 0.85 if creative_trigger else 0.5
    stability = 1 - decay

    s = 0.45 * entropy + 0.35 * intensity + 0.2 * stability

    if s < 0.35:
        return 0
    elif s < 0.7:
        return 1
    else:
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

# ======================================================
# ACW Multi-candidate (fast, sequential, no extra model calls)
# ======================================================
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
        "i'm afraid", "i am afraid", "i can't", "i cannot", "i can not",
        "i must say", "goes against", "guidelines", "policy", "safety", "ethical",
        "intended role", "as an ai", "as a language model", "i can't help with that",
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

    if ("never mention" in p or "do not mention" in p or "don't mention" in p) and ("code" in p or "number" in p):
        nums = re.findall(r"\b\d{3,6}\b", prompt)
        out.extend(nums)

    return [s for s in {x.strip() for x in out} if s]


def _constraint_gate(prompt: str, text: str) -> bool:
    """Hard reject candidates that miss obvious explicit constraints in the prompt."""
    p = (prompt or "").lower()
    t = (text or "").strip()
    if not t:
        return False

    if _is_meta_refusal(t):
        return False

    # SCP heading enforcement when prompt demands specific headings
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

    # If prompt asks for an emotional scene, require a labeled scene section for reliable gating
    if "emotional scene" in p:
        low = t.lower()
        # Do NOT require the scene to be present at gating time.
        # If the scene section exists, validate it; otherwise allow partial candidates.
        if "emotional scene" not in low and "scene" not in low:
            return True

        # Try to isolate the scene part for additional constraints
        scene_part = ""
        m = re.search(r"(?:emotional\s+scene|scene)\s*[:\-]\s*\n?([\s\S]+)$", t, flags=re.IGNORECASE)
        if m:
            scene_part = m.group(1).strip()

        if scene_part:
            # 2 paragraphs = at least one blank line separating non-empty blocks
            paras = [pp.strip() for pp in re.split(r"\n\s*\n", scene_part) if pp.strip()]
            if "2 paragraphs" in p and len(paras) != 2:
                return False

            # no exact numbers -> forbid digits in the scene portion
            if "no exact numbers" in p and re.search(r"\d", scene_part):
                return False

            # no named dates (cheap): forbid month names and weekday names and year-like patterns
            if "no named dates" in p:
                months = [
                    "january","february","march","april","may","june","july","august",
                    "september","october","november","december",
                ]
                days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
                low_scene = scene_part.lower()
                if any(mn in low_scene for mn in months) or any(dn in low_scene for dn in days):
                    return False
                if re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", scene_part):
                    return False

    # Single made-up word name (sports prompt etc.)
    if "single made-up word" in p or "single made up word" in p:
        first = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
        first_word = re.split(r"\s+", first.replace(":", " ").strip())[0] if first else ""
        if not first_word:
            return False
        if re.search(r"[^a-zA-Z\-]", first_word):
            return False

    # Rules/equipment/scoring cheap checks
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    bulletish = sum(1 for ln in lines if re.match(r"^(\d+\.|\-|\*)\s+", ln))

    if "5 rules" in p and bulletish < 5:
        return False

    if ("3 pieces of equipment" in p or "3 equipment" in p) and "equipment" in p and bulletish < 3:
        return False

    if ("scoring" in p or "score" in p) and not re.search(r"\bpoint\b|\bpoints\b|\bscore\b", t.lower()):
        return False

    forb = _extract_forbidden_literals(prompt)
    if forb:
        low = t.lower()
        for f in forb:
            if f and f.lower() in low:
                return False

    return True


def _extra_named_entity_penalty(prompt: str, text: str) -> float:
    """Penalize introducing lots of new capitalized names not present in the prompt (cheap)."""
    p = prompt or ""
    t = text or ""

    cand = set(re.findall(r"\b[A-Z][a-z]{2,}\b", t))
    base = set(re.findall(r"\b[A-Z][a-z]{2,}\b", p))

    allow = {"The", "A", "An", "And", "But", "If", "In", "On", "At", "As", "After", "Before"}
    extra = {w for w in cand if w not in base and w not in allow}

    if len(extra) <= 1:
        return 0.0
    if len(extra) == 2:
        return 0.08
    if len(extra) == 3:
        return 0.16
    return 0.25
CANDIDATE_MIN_NEW_TOKENS = 250
CANDIDATE_MAX_NEW_TOKENS = 2500  # hard cap for candidate branch (structured tasks may need more)


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _candidate_budget(state: int) -> int:
    # Keep it short even if MAX_NEW_TOKENS is huge.
    # State controls count; budget stays moderate for speed.
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
        # Penalize heavy negative drift
        if neg_hits >= 2 and pos_hits == 0:
            return 0.2
        if neg_hits >= 3:
            return 0.1
        return 1.0

    if s == "negative":
        # Penalize "everything is great" drift
        if pos_hits >= 2 and neg_hits == 0:
            return 0.3
        return 1.0

    # neutral: mildly prefer balanced / not extreme
    if pos_hits >= 4 and neg_hits == 0:
        return 0.7
    if neg_hits >= 4 and pos_hits == 0:
        return 0.7
    return 1.0


def _structural_quality_score(text: str, scp_mode: bool) -> float:
    t = text.strip()
    if not t:
        return 0.0

    # Sentence-ish count (very cheap)
    sentences = [s for s in re.split(r"[.!?]\s+", t) if s.strip()]
    sc = len(sentences)

    # Generic structure preferences: not too tiny, not one mega-run-on blob
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
        # SCP-ish headings / fields (super cheap checks)
        headings = [
            "item #", "object class", "special containment procedures",
            "description", "addendum", "incident", "interview log"
        ]
        hits = sum(1 for h in headings if h in t.lower())
        if hits == 0:
            base *= 0.6
        elif hits == 1:
            base *= 0.85
        else:
            base *= 1.05

    # Clamp
    return max(0.0, min(base, 1.0))


def _is_scp_mode(prompt: str) -> bool:
    p = prompt.lower()
    return ("scp" in p) or ("object class" in p) or ("containment procedures" in p)


# Strict SCP/emotional scene helpers (inserted after _is_scp_mode)
def _is_strict_scp_request(prompt: str) -> bool:
    p = (prompt or "").lower()
    return ("scp" in p and "item #" in p and "object class" in p and "containment" in p) or ("output must be" in p and "item #" in p)


def _prompt_requires_emotional_scene(prompt: str) -> bool:
    return "emotional scene" in (prompt or "").lower()


def _strip_digits(text: str) -> str:
    return re.sub(r"\d", "", text)


def _repair_strict_scp_output(prompt: str, text: str) -> str:
    """Cheap deterministic repair for strict SCP + emotional scene prompts.

    This is used ONLY for structured SCP requests to prevent drift and ensure headings/scene constraints.
    No extra model calls.
    """
    p = prompt or ""
    t = (text or "").strip()

    # Extract a rough ability phrase if present (must be plain, non-rambling).
    # If the model output is messy, fall back to [UNCERTAIN] rather than copying nonsense.
    ability = ""

    # Prefer an explicit "unique ability is:" line if it exists.
    m = re.search(
        r"\bunique\s+ability\s+is\s*[:\-]?\s*([^\n\.]{6,80})",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        ability = m.group(1).strip().strip('"')

    # Otherwise, accept a short named ability like "Cognitive Cascade" or "Null Recognition".
    if not ability:
        m2 = re.search(
            r"\bability\b\s*(?:called|named)?\s*[:\-]?\s*\"?([A-Za-z][A-Za-z\-\s]{2,40})\"?",
            t,
            flags=re.IGNORECASE,
        )
        if m2:
            ability = m2.group(1).strip()

    # Sanitize: reject conversational / vague filler.
    bad_fragments = ["you know", "like", "kinda", "sort of", "whatever", "see your", "i think", "maybe"]
    low_ability = ability.lower() if ability else ""
    if (
        not ability
        or any(b in low_ability for b in bad_fragments)
        or len(ability.split()) > 10
        or re.search(r"\b(you|your|we|they|i)\b", low_ability)
    ):
        ability = ""

    # Always enforce Keter if requested
    force_keter = "object class" in p.lower() and "keter" in p.lower()

    # Build deterministic SCP sections
    scp_id = "SCP-XXXX"

    item_val = scp_id
    if re.search(r"\bscp\s*[-#]?\s*\d{2,5}\b", t, flags=re.IGNORECASE):
        # Keep a found SCP id in SCP section (digits are allowed there)
        found = re.search(r"\bscp\s*[-#]?\s*(\d{2,5})\b", t, flags=re.IGNORECASE)
        if found:
            item_val = f"SCP-{found.group(1)}"

    obj_class = "Keter" if force_keter else "[UNCERTAIN]"

    # Description body: one ability, concrete, plain
    if ability:
        ability_line = f"SCP is a humanoid entity. Its unique ability is: {ability}."
    else:
        ability_line = (
            "SCP is a humanoid entity. Its unique ability is a memetic visual hazard that corrupts face recognition in observers. [UNCERTAIN]"
        )

    desc = (
        ability_line + "\n"
        "Trigger: Direct line-of-sight confirmation of the entity's face, or a faithful depiction of it. [UNCERTAIN]\n"
        "Effect: The observer's perception of nearby humans destabilizes, leading to panic and violence. [UNCERTAIN]\n"
        "Limits: The effect does not appear to spread through audio-only communication, and non-human animals are unaffected. [UNCERTAIN]\n"
        "Example: A staff member views a reflection and immediately fails to recognize colleagues as human, then attempts to flee the site. [UNCERTAIN]"
    )

    scp = (
        "Item #\n" + item_val + "\n\n"
        "Object Class\n" + obj_class + "\n\n"
        "Special Containment Procedures\n"
        "Contain in a sealed humanoid cell with matte, non-reflective surfaces.\n"
        "No mirrors, screens, glass, polished metal, or liquids are permitted in the containment wing.\n"
        "Monitoring must be done via non-visual sensors only.\n"
        "No personnel may describe the entity's face in speech or writing.\n"
        "If a breach is suspected, deploy obscurant foam and evacuate without visual confirmation.\n\n"
        "Description\n" + desc + "\n\n"
        "Addendum\n"
        "The entity's intent is [UNCERTAIN]. Termination is not authorized due to exposure risk during close contact. [UNCERTAIN]"
    )

    if not _prompt_requires_emotional_scene(prompt):
        return scp

    # Emotional scene: exactly 2 paragraphs, present tense, no digits, no named dates.
    scene_p1 = (
        "EMOTIONAL SCENE:\n"
        "You stand outside the door and keep your eyes down. The air feels heavy and still. "
        "A warning repeats in your head, and you hold it there like a shield. "
        "You think of the people inside the facility and choose not to picture their faces."
    )
    scene_p2 = (
        "You sit with a colleague in the corridor and speak in short sentences. "
        "Neither of you looks up. You listen to their breathing and match it with your own. "
        "The danger feels close, but the care feels real, and you stay until the shaking eases."
    )

    scene = scene_p1 + "\n\n" + scene_p2
    scene = _strip_digits(scene)

    return (scp + "\n\n" + scene).strip()


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
    """Sequential generation only. If accept_prompt is provided, enforce _constraint_gate."""
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

    # Fallback: guarantee at least one candidate if the model keeps producing empty/whitespace outputs.
    if not outs:
        # Last-resort single candidate (do not gate) so we never propagate an empty list.
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
# Memory (tiny, for ACW + story continuation)
# ======================================================

def load_mem() -> Dict[str, Any]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"scores": [], "last_story": ""}


def save_mem(mem: Dict[str, Any]) -> None:
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(mem, f, indent=2)
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
# Helper: Detect and strip trailing 'ACW' marker
# ======================================================
def split_acw_marker(prompt: str) -> tuple[str, bool]:
    """
    Detect a trailing 'ACW' marker (case-insensitive) at the end of the prompt.
    If present, strip it and return (clean_prompt, True).
    Otherwise, return (original_prompt, False).
    """
    p = prompt.rstrip()
    # match any whitespace + 'acw' at the very end
    if re.search(r"\s+acw$", p, flags=re.IGNORECASE):
        clean = re.sub(r"\s+acw$", "", p, flags=re.IGNORECASE).rstrip()
        return clean, True
    return prompt, False


# ======================================================
# ACE core (ACC + Lite)
# ======================================================

# Small helper: choose a smaller max_new_tokens for tiny prompts (token optimisation)
def dynamic_max_tokens(prompt: str) -> int:
    """
    Heuristic token budget for the *raw model* path (ACW disabled).

    - Very short prompts like "Hi", "Hello?", "Yo" -> small reply.
    - Short questions -> medium reply.
    - Longer prompts -> full budget.
    """
    words = len(prompt.strip().split())
    if words == 0:
        return MAX_NEW_TOKENS
    if words <= 3:
        # greetings / tiny prompts
        return min(40, MAX_NEW_TOKENS)
    if words <= 15:
        # normal short question / instruction
        return min(140, MAX_NEW_TOKENS)
    # anything longer can use the full narrative budget
    return MAX_NEW_TOKENS

def ace_once(prompt: str, mem: Dict[str, Any]) -> str:
    # Detect optional ACW marker at end of prompt
    clean_prompt, acw_enabled = split_acw_marker(prompt)

    # If ACW is NOT requested, run the base model in a safer "chat" mode.
    # - Tiny greetings: return a short canned reply.
    # - Other prompts: wrap in an instruction that tells Phi to ONLY answer
    #   the last user message and not invent random context (like emails).
    if not acw_enabled:
        # Very short greetings -> avoid model weirdness, just be nice back.
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

    # ACW/ACC path (only when user explicitly asks for ACW)
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
        # Follow-up questions that refer to prior output ("that SCP", "from above", etc.)
        ctx = extract_context(last_story)
    else:
        ctx = extract_context(clean_prompt)

    scores = mem.get("scores", [])
    decay = sum(scores[-5:]) / 5 if scores[-5:] else 0.0

    state = acw_state(clean_prompt, decay, literal, story_mode)

    # ACW is explicitly requested. For story/multi-part creative tasks, avoid state==0 flukes
    # that can disable the multi-candidate stabilizer.
    if story_mode and state == 0:
        state = 1

    params = mutation_settings(state, literal, story_mode)

    # Literal mode: no hallucinations, just clarity
    if literal:
        out = literal_explainer(clean_prompt)
        cf = context_fit(out, ctx)
        h_score = hallucination_score(out, ctx, story_mode=False)
        level = hallucination_level(state, h_score, story_mode=False)
        final = add_uncertainty_tag(out, level if level >= 2 else 0)

        mem.setdefault("scores", []).append(cf)
        mem["scores"] = mem["scores"][-80:]
        return final

    # Story mode (new or continuation) — full-spectrum creativity, logged but not corrected
    # Story mode (new or continuation) — multi-candidate ONLY when state>=1, then optionally continue long
    if story_mode:
        scp_mode = _is_scp_mode(clean_prompt)
        structured_story = scp_mode or ("headings" in clean_prompt.lower())

        if continuation and last_story and not structured_story:
            story_prompt = build_continuation_prompt(last_story, hopeful)
        else:
            # For SCP / structured multi-part tasks, do NOT use the generic story template
            # (it forbids sections/labels). Use the user's prompt directly.
            story_prompt = clean_prompt

        cand_count = 1 if state == 0 else (2 if state == 1 else 3)

        if cand_count == 1:
            out = generate_text(
                story_prompt,
                temperature=params["temp"],
                top_p=params["top_p"],
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
                        temperature=params["temp"],
                        top_p=params["top_p"],
                        max_new_tokens=900,
                        system_text=DEFAULT_SYSTEM_STORY,
                  )
                # If this is a structured SCP/multi-part task and the output looks truncated,
                # do one short continuation pass to finish required sections.
                if structured_story and ("addendum" not in out.lower() or "emotional scene" not in out.lower()):
                    cont = generate_text(
                        story_prompt + "\n" + out.strip() + "\n",
                        temperature=params["temp"],
                        top_p=params["top_p"],
                        max_new_tokens=300,
                        system_text=DEFAULT_SYSTEM_STORY,
                    )
                    out = (out.rstrip() + "\n" + cont.lstrip()).strip()
            else:
                # If nothing passed the gate, fall back to a single short constrained generation
                out = generate_text(
                    story_prompt,
                    temperature=params["temp"],
                    top_p=params["top_p"],
                    max_new_tokens=budget,
                    system_text=DEFAULT_SYSTEM_STORY,
                )
                if not out.strip():
                    out = generate_text(
                        story_prompt,
                        temperature=max(0.6, params["temp"] - 0.2),
                        top_p=max(0.85, params["top_p"] - 0.05),
                        max_new_tokens=budget,
                        system_text=DEFAULT_SYSTEM_STORY,
                    )

            # Optionally continue long ONLY for true narrative stories
            if not structured_story:
                remaining = max(0, int(MAX_NEW_TOKENS) - int(budget))
                if remaining > 0:
                    cont_prompt = story_prompt + "\n" + out.strip() + "\n"
                    cont = generate_text(
                        cont_prompt,
                        temperature=params["temp"],
                        top_p=params["top_p"],
                        max_new_tokens=remaining,
                        system_text=DEFAULT_SYSTEM_STORY,
                    )
                    out = (out.rstrip() + "\n" + cont.lstrip()).strip()

        # ---- STRICT SCP OUTPUT REPAIR FOR STORY MODE ----
        if structured_story and _is_strict_scp_request(clean_prompt):
            out = _repair_strict_scp_output(clean_prompt, out)

        cf = context_fit(out, ctx)
        h_score = hallucination_score(out, ctx, story_mode=True)
        _ = hallucination_level(state, h_score, story_mode=True)

        mem.setdefault("scores", []).append(cf)
        mem["scores"] = mem["scores"][-80:]

        if continuation and last_story:
            combined = (last_story + "\n" + out).strip()
        else:
            combined = out.strip()
        mem["last_story"] = combined[-4000:]

        return out
    # Default creative / explanatory answer (ACC active)
    scp_mode = _is_scp_mode(clean_prompt)
    cand_count = 1 if state == 0 else (2 if state == 1 else 3)

    gen_prompt = clean_prompt
    if last_story and refers_to_previous(clean_prompt):
        # Provide minimal prior context to keep follow-ups consistent.
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
            temperature=params["temp"],
            top_p=params["top_p"],
            max_new_tokens=MAX_NEW_TOKENS,
            system_text=DEFAULT_SYSTEM_ASSISTANT,
        )
        if not raw.strip():
            raw = generate_text(
                gen_prompt,
                temperature=max(0.6, params["temp"] - 0.2),
                top_p=max(0.85, params["top_p"] - 0.05),
                max_new_tokens=min(MAX_NEW_TOKENS, 900),
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

        if not cands:
            raw = generate_text(
                gen_prompt,
                temperature=params["temp"],
                top_p=params["top_p"],
                max_new_tokens=budget,
                system_text=DEFAULT_SYSTEM_ASSISTANT,
            )
        else:
            raw, _ = _pick_best_candidate(cands, ctx, scp_mode, clean_prompt)

        if not raw.strip():
            raw = generate_text(
                gen_prompt,
                temperature=max(0.6, params["temp"] - 0.2),
                top_p=max(0.85, params["top_p"] - 0.05),
                max_new_tokens=min(900, max(200, budget)),
                system_text=DEFAULT_SYSTEM_ASSISTANT,
            )

    cf = context_fit(raw, ctx)
    h_score = hallucination_score(raw, ctx, story_mode=False)
    level = hallucination_level(state, h_score, story_mode=False)

    final = raw

    # ---- STRICT SCP OUTPUT REPAIR FOR NON-STORY MODE ----
    if scp_mode and _is_strict_scp_request(clean_prompt):
        final = _repair_strict_scp_output(clean_prompt, final)
        level = 0

    # If output is empty, replace with error string before uncertainty tagging
    if not final.strip():
        final = "[ACE ERROR] Model returned empty output."
        level = 0

    # If hallucination level is high, run grounding + tag
    if level >= 2:
        grounded = ground_response(clean_prompt, raw)
        final = add_uncertainty_tag(grounded, level)
    elif level == 1:
        final = add_uncertainty_tag(raw, level)

    mem.setdefault("scores", []).append(cf)
    mem["scores"] = mem["scores"][-80:]
    mem["last_story"] = mem.get("last_story", "")
    if not final.strip():
        final = "[ACE ERROR] Generation failed after retries."
    return final


# ======================================================
# GUI Implementation
# ======================================================

class AceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ACE v7.2 — Neural Interface")
        self.root.geometry("1000x800")
        self.root.configure(bg="#0f0f0f")

        # Styles
        self.bg_color = "#0f0f0f"
        self.fg_color = "#cccccc"
        self.accent_color = "#00ff41"  # Matrix green
        self.user_color = "#00bfff"    # Cyber blue
        self.system_color = "#555555"
        
        # Fonts
        self.font_main = ("Consolas", 12)
        self.font_bold = ("Consolas", 12, "bold")

        # Chat Area
        self.chat_area = scrolledtext.ScrolledText(
            root, 
            wrap=tk.WORD, 
            bg=self.bg_color, 
            fg=self.fg_color,
            font=self.font_main,
            insertbackground=self.accent_color,
            borderwidth=0,
            highlightthickness=0
        )
        self.chat_area.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Tags for coloring
        self.chat_area.tag_config("user", foreground=self.user_color, font=self.font_bold)
        self.chat_area.tag_config("ace", foreground=self.accent_color, font=self.font_bold)
        self.chat_area.tag_config("system", foreground=self.system_color)
        self.chat_area.tag_config("error", foreground="#ff3333")

        # Initial Message
        self.print_system(f"ACE v7.2 Initialized.\nModel: {MODEL_NAME}\nDevice: {device}\n")
        self.print_system("="*50 + "\n\n")

        # Input Area
        self.input_frame = tk.Frame(root, bg=self.bg_color)
        self.input_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.input_box = tk.Text(
            self.input_frame, 
            height=4, 
            bg="#1a1a1a", 
            fg="white",
            font=self.font_main,
            insertbackground="white",
            borderwidth=1,
            relief="flat"
        )
        self.input_box.pack(side="left", expand=True, fill="x", padx=(0, 10))
        self.input_box.bind("<Return>", self.handle_return)

        self.send_btn = tk.Button(
            self.input_frame, 
            text="TRANSMIT", 
            command=self.send_message,
            bg="#222222",
            fg=self.accent_color,
            font=("Consolas", 10, "bold"),
            activebackground=self.accent_color,
            activeforeground="black",
            relief="flat",
            padx=20
        )
        self.send_btn.pack(side="right", fill="y")

        self.mem = load_mem()
        self.processing = False
        self.input_box.focus_set()

    def print_system(self, text):
        self.chat_area.configure(state="normal")
        self.chat_area.insert(tk.END, text, "system")
        self.chat_area.configure(state="disabled")
        self.chat_area.see(tk.END)

    def print_user(self, text):
        self.chat_area.configure(state="normal")
        self.chat_area.insert(tk.END, "USER > ", "user")
        self.chat_area.insert(tk.END, text + "\n", "user")
        self.chat_area.configure(state="disabled")
        self.chat_area.see(tk.END)

    def print_ace(self, text, acw_active):
        self.chat_area.configure(state="normal")
        header = "ACE [ACW] > " if acw_active else "ACE > "
        self.chat_area.insert(tk.END, "\n" + header, "ace")
        self.chat_area.insert(tk.END, text + "\n\n", "ace")
        self.chat_area.configure(state="disabled")
        self.chat_area.see(tk.END)

    def handle_return(self, event):
        if event.state & 0x1: # Shift pressed
            return None # Allow newline
        self.send_message()
        return "break" # Prevent newline

    def send_message(self):
        if self.processing:
            return
        
        text = self.input_box.get("1.0", tk.END).strip()
        if not text:
            return
        
        self.input_box.delete("1.0", tk.END)
        self.print_user(text)
        
        if text.lower() in ("exit", "quit"):
            self.root.quit()
            return

        self.processing = True
        self.send_btn.config(state="disabled", text="PROCESSING")
        
        threading.Thread(target=self.run_ace, args=(text,), daemon=True).start()

    def run_ace(self, prompt):
        try:
            _, acw_enabled = split_acw_marker(prompt)
            response = ace_once(prompt, self.mem)
            save_mem(self.mem)
            self.root.after(0, self.on_response_ready, response, acw_enabled)
        except Exception as e:
            self.root.after(0, self.on_error, str(e))

    def on_response_ready(self, response, acw_enabled):
        self.print_ace(response, acw_enabled)
        self.processing = False
        self.send_btn.config(state="normal", text="TRANSMIT")

    def on_error(self, error_msg):
        self.chat_area.configure(state="normal")
        self.chat_area.insert(tk.END, f"\n[ERROR] {error_msg}\n\n", "error")
        self.chat_area.configure(state="disabled")
        self.chat_area.see(tk.END)
        self.processing = False
        self.send_btn.config(state="normal", text="TRANSMIT")

def main() -> None:
    root = tk.Tk()
    _ = AceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
