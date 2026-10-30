"""QuizForge AI — AI quiz & flashcard generator for students.

Single-file Streamlit app. Run with:  streamlit run app.py
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from docx import Document
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_BALANCED = "openai/gpt-oss-120b"      # primary
MODEL_FAST = "openai/gpt-oss-20b"           # cheap / fast
MODEL_FALLBACK = "llama-3.3-70b-versatile"  # automatic fallback

MODEL_LABELS: dict[str, str] = {
    "Balanced": MODEL_BALANCED,
    "Fast": MODEL_FAST,
}

DIFFICULTIES = ["Easy", "Medium", "Hard"]
QUESTION_COUNTS = [5, 10, 15, 20]
STYLES = ["Conceptual", "Definition", "Application"]
LANGUAGES = ["English", "Urdu-Roman"]

MIN_NOTES_CHARS = 120
MAX_NOTES_CHARS = 12_000
READING_WORDS_PER_MINUTE = 200

TEMPERATURE = 0.4
REQUEST_TIMEOUT = 90.0
MAX_RETRIES = 1

KEY_MISSING_MESSAGE = (
    "No Groq API key found. Add GROQ_API_KEY to your Streamlit secrets or your "
    "environment, then reload the app."
)

SYSTEM_PROMPT = """You are QuizForge, an exam-writing engine for students.
Return ONLY a single valid JSON object. No markdown fences, no preamble, no commentary.

The JSON must match exactly this schema:
{
  "topic": "short topic name",
  "questions": [
    {
      "id": 1,
      "question": "string",
      "options": ["A text", "B text", "C text", "D text"],
      "answer_index": 0,
      "explanation": "2-3 sentences explaining why the answer is correct and why one tempting wrong option is wrong",
      "difficulty": "easy"
    }
  ],
  "flashcards": [
    {"front": "short term or question", "back": "concise answer"}
  ]
}

Rules:
- Exactly 4 options per question, all non-empty, all distinct.
- answer_index is an integer 0-3 pointing at the correct option.
- Vary the position of the correct answer across questions.
- "difficulty" must be one of "easy", "medium", "hard".
- Base every question strictly on the supplied notes. Never invent facts.
- Output valid JSON only."""


class QuizForgeError(Exception):
    """A failure that has already been translated into a human message."""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    """Create every session key once so reruns never lose progress."""
    defaults: dict[str, Any] = {
        "quiz_data": None,
        "current_index": 0,
        "user_answers": [],
        "answered": False,
        "quiz_finished": False,
        "score": 0,
        "started_at": None,
        "finished_at": None,
        "flash_index": 0,
        "flash_pos": 0,
        "flash_known": set(),
        "flash_order": [],
        "flash_flipped": False,
        "flash_filter": False,
        "notes_text": "",
        "difficulty": "Medium",
        "num_questions": 10,
        "style": "Conceptual",
        "language": "English",
        "model_choice": "Balanced",
        "uploaded_name": "",
        "allow_trim": False,
        "last_error": "",
        "quiz_meta": None,
        "pending_notes": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_progress() -> None:
    """Clear answering state only — never touches notes or settings."""
    st.session_state.current_index = 0
    st.session_state.user_answers = []
    st.session_state.answered = False
    st.session_state.quiz_finished = False
    st.session_state.score = 0
    st.session_state.started_at = None
    st.session_state.finished_at = None


def reset_flashcards() -> None:
    st.session_state.flash_index = 0
    st.session_state.flash_pos = 0
    st.session_state.flash_known = set()
    st.session_state.flash_flipped = False
    st.session_state.flash_order = []


# ---------------------------------------------------------------------------
# API key, client and connection status
# ---------------------------------------------------------------------------


def load_api_key() -> str | None:
    """Read the key from Streamlit secrets, then the environment, else None."""
    try:
        secret = st.secrets["GROQ_API_KEY"]
    except Exception:  # no secrets.toml, or key absent — both are fine
        secret = None
    if secret and str(secret).strip():
        return str(secret).strip()

    env_value = os.environ.get("GROQ_API_KEY")
    if env_value and env_value.strip():
        return env_value.strip()
    return None


def fingerprint(value: str) -> str:
    """Short, non-reversible token used as a cache key (never the raw key)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def _cached_client(key_fingerprint: str) -> Groq:
    """Build the SDK client once per key."""
    key = load_api_key()
    if not key:
        raise QuizForgeError(KEY_MISSING_MESSAGE)
    return Groq(api_key=key, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)


def get_client() -> Groq | None:
    key = load_api_key()
    if not key:
        return None
    return _cached_client(fingerprint(key))


@st.cache_resource(show_spinner=False)
def connection_status(key_fingerprint: str) -> tuple[bool, str]:
    """One cheap call per key: is the API reachable and the key accepted?"""
    client = get_client()
    if client is None:
        return False, "API Key Missing"
    try:
        client.models.list()
        return True, "API Connected"
    except (AuthenticationError, PermissionDeniedError):
        return False, "API Key Invalid"
    except Exception:
        return False, "API Unreachable"


def api_ready() -> tuple[bool, str]:
    key = load_api_key()
    if not key:
        return False, "API Key Missing"
    return connection_status(fingerprint(key))


# ---------------------------------------------------------------------------
# File extraction
# ---------------------------------------------------------------------------


def extract_text_from_file(filename: str, data: bytes) -> str:
    """Pull readable text out of a .txt / .pdf / .docx upload."""
    name = (filename or "").lower()
    try:
        if name.endswith(".txt"):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1", errors="ignore")
        elif name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        elif name.endswith(".docx"):
            document = Document(io.BytesIO(data))
            parts = [p.text for p in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text.strip() for cell in row.cells))
            text = "\n".join(parts)
        else:
            raise QuizForgeError("Unsupported file type. Please upload a .txt, .pdf or .docx file.")
    except QuizForgeError:
        raise
    except Exception as exc:
        raise QuizForgeError("Could not read this file, please paste the text instead.") from exc

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 40:
        raise QuizForgeError("Could not read this file, please paste the text instead.")
    return text


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

STYLE_HINTS = {
    "Conceptual": "test understanding of ideas and the relationships between them",
    "Definition": "test precise recall of terms, definitions and key facts",
    "Application": "put each question in a short scenario where the student applies the concept",
}


def build_prompt(notes: str, count: int, difficulty: str, style: str, language: str) -> str:
    """Compose the user message describing the task."""
    if language == "Urdu-Roman":
        language_line = (
            "Roman Urdu — write every question, option, explanation and flashcard in "
            "natural Roman Urdu (Urdu words in Latin letters, e.g. \"Tashreeh kya hai?\")"
        )
    else:
        language_line = "English"

    flashcard_target = max(4, round(count * 0.8))
    return f"""Create a study quiz from the notes below.

- Number of questions: exactly {count}
- Difficulty: {difficulty}
- Question style: {style} — {STYLE_HINTS.get(style, STYLE_HINTS['Conceptual'])}
- Language for all text: {language_line}
- Flashcards: about {flashcard_target} cards covering the key terms and facts
- Every question needs 4 distinct options and a 2-3 sentence explanation that also
  says why one tempting wrong option is wrong.
- "difficulty" per question: "easy", "medium" or "hard".
- Use only facts present in the notes.

NOTES:
\"\"\"
{notes}
\"\"\""""


# ---------------------------------------------------------------------------
# Parsing + validation (model output is untrusted)
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()


def parse_json_safely(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON recovery: raw → de-fenced → brace slice → comma repair."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()

    for candidate in (text, _strip_fences(text)):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            continue

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        sliced = text[start : end + 1]
        for candidate in (sliced, re.sub(r",(\s*[}\]])", r"\1", sliced)):
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue
    return None


def _clean_str(value: Any, limit: int = 2000) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _validate_question(item: Any) -> dict[str, Any] | None:
    """Return a normalised question, or None if it is unusable."""
    if not isinstance(item, dict):
        return None

    question = _clean_str(item.get("question"), 600)
    explanation = _clean_str(item.get("explanation"), 1400)
    raw_options = item.get("options")
    if not question or not explanation:
        return None
    if not isinstance(raw_options, list) or len(raw_options) != 4:
        return None

    options = [_clean_str(option, 400) for option in raw_options]
    if any(not option for option in options):
        return None
    if len({option.lower() for option in options}) != 4:
        return None

    answer_index = item.get("answer_index")
    if isinstance(answer_index, bool):
        return None
    try:
        answer_index = int(answer_index)
    except (TypeError, ValueError):
        return None
    if not 0 <= answer_index <= 3:
        return None

    difficulty = _clean_str(item.get("difficulty"), 12).lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"

    return {
        "question": question,
        "options": options,
        "answer_index": answer_index,
        "explanation": explanation,
        "difficulty": difficulty,
    }


def validate_quiz(payload: Any, expected_count: int) -> tuple[dict[str, Any] | None, str]:
    """Validate and normalise the whole payload. Returns (quiz, warning_message)."""
    if not isinstance(payload, dict):
        return None, "The model response was not a JSON object."

    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        return None, "No questions were found in the model response."

    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = 0
    for item in raw_questions:
        question = _validate_question(item)
        if question is None:
            dropped += 1
            continue
        key = re.sub(r"\W+", "", question["question"].lower())[:120]
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        questions.append(question)

    if not questions:
        return None, "Every generated question was malformed. Please try again."
    if len(questions) > expected_count:
        questions = questions[:expected_count]
    for position, question in enumerate(questions, start=1):
        question["id"] = position

    notes: list[str] = []
    if dropped or len(questions) < expected_count:
        notes.append(
            f"{len(questions)} of {expected_count} questions were usable — "
            f"{dropped} malformed item(s) were dropped."
        )

    cards: list[dict[str, str]] = []
    raw_cards = payload.get("flashcards")
    if isinstance(raw_cards, list):
        for card in raw_cards:
            if not isinstance(card, dict):
                continue
            front = _clean_str(card.get("front"), 300)
            back = _clean_str(card.get("back"), 700)
            if front and back:
                cards.append({"front": front, "back": back})
    if not cards:
        cards = [
            {
                "front": question["question"],
                "back": f"{question['options'][question['answer_index']]} — {question['explanation']}",
            }
            for question in questions
        ]
        notes.append("Flashcards were rebuilt from the questions because none came back usable.")

    topic = _clean_str(payload.get("topic"), 120) or "Study notes"
    return (
        {"topic": topic, "questions": questions, "flashcards": cards},
        " ".join(notes),
    )


# ---------------------------------------------------------------------------
# Groq calls
# ---------------------------------------------------------------------------


def _friendly_error(exc: Exception) -> str:
    """Translate SDK exceptions into one human sentence."""
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return "API key invalid or missing. Check GROQ_API_KEY and try again."
    if isinstance(exc, RateLimitError):
        return "Groq speed limit reached, please wait a few seconds and retry."
    if isinstance(exc, APITimeoutError):
        return "The request timed out. Network issue, try again."
    if isinstance(exc, APIConnectionError):
        return "Network issue, try again."
    if isinstance(exc, InternalServerError):
        return "Groq had a server error. Network issue, try again in a moment."
    if isinstance(exc, NotFoundError):
        return "That model is no longer available. Please try again."
    if isinstance(exc, BadRequestError):
        return "Groq rejected the request. Try a smaller question count and retry."
    return "Something went wrong while contacting Groq. Please try again."


def _is_model_missing(exc: Exception) -> bool:
    """True when the model ID is unknown or decommissioned."""
    if isinstance(exc, NotFoundError):
        return True
    message = str(exc).lower()
    markers = (
        "model_not_found",
        "decommission",
        "does not exist",
        "no longer supported",
        "model not found",
        "unknown model",
    )
    return any(marker in message for marker in markers)


def _request(client: Groq, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    """Single chat completion. Falls back to plain text mode if JSON mode is rejected."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "timeout": REQUEST_TIMEOUT,
    }
    try:
        response = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except BadRequestError as exc:
        lowered = str(exc).lower()
        if "response_format" in lowered or "json_object" in lowered:
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise QuizForgeError("The model returned an empty response. Please try again.")
    return content


def _request_with_fallback(
    client: Groq, model: str, messages: list[dict[str, str]], max_tokens: int
) -> tuple[str, str, str]:
    """Run the call, retrying once on the fallback model if the model is gone."""
    try:
        return _request(client, model, messages, max_tokens), model, ""
    except QuizForgeError:
        raise
    except Exception as exc:
        if _is_model_missing(exc) and model != MODEL_FALLBACK:
            try:
                raw = _request(client, MODEL_FALLBACK, messages, max_tokens)
            except Exception as fallback_exc:
                raise QuizForgeError(_friendly_error(fallback_exc)) from fallback_exc
            notice = f"{model} is unavailable right now, so {MODEL_FALLBACK} answered instead."
            return raw, MODEL_FALLBACK, notice
        raise QuizForgeError(_friendly_error(exc)) from exc


def call_llm(
    notes: str, count: int, difficulty: str, style: str, language: str, model: str
) -> dict[str, Any]:
    """Ask the model for a quiz and return the raw parsed payload plus metadata."""
    client = get_client()
    if client is None:
        raise QuizForgeError(KEY_MISSING_MESSAGE)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(notes, count, difficulty, style, language)},
    ]
    max_tokens = min(8000, 700 + count * 320)

    started = time.time()
    raw, model_used, notice = _request_with_fallback(client, model, messages, max_tokens)
    payload = parse_json_safely(raw)
    notices = [notice] if notice else []

    if payload is None:
        repair_messages = messages + [
            {"role": "assistant", "content": raw[:6000]},
            {
                "role": "user",
                "content": (
                    "That response was not valid JSON. Return ONLY the JSON object exactly as "
                    "specified, starting with { and ending with }. No markdown fences, no commentary."
                ),
            },
        ]
        raw, model_used, repair_notice = _request_with_fallback(
            client, model_used, repair_messages, max_tokens
        )
        if repair_notice:
            notices.append(repair_notice)
        payload = parse_json_safely(raw)
        notices.append("The first response was malformed, so it was repaired once.")

    if payload is None:
        raise QuizForgeError(
            "The model did not return usable JSON this time. Press Try again — it usually "
            "works on the second attempt."
        )

    return {
        "payload": payload,
        "model": model_used,
        "elapsed": time.time() - started,
        "notices": notices,
    }


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
  --qf-bg:#0B1220;
  --qf-bg-soft:#0F172A;
  --qf-card:rgba(23,32,53,.72);
  --qf-border:rgba(148,163,184,.18);
  --qf-text:#E6EDF7;
  --qf-muted:#94A3B8;
  --qf-violet:#7C3AED;
  --qf-cyan:#22D3EE;
  --qf-green:#22C55E;
  --qf-red:#F43F5E;
  --qf-radius:18px;
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 620px at 6% -10%, rgba(124,58,237,.22), transparent 60%),
    radial-gradient(900px 520px at 96% 2%, rgba(34,211,238,.16), transparent 58%),
    var(--qf-bg);
  color:var(--qf-text);
  font-family:'Inter','Segoe UI',system-ui,sans-serif;
}

/* --- chrome --- */
header[data-testid="stHeader"]{background:transparent;height:0;}
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"],
#MainMenu, footer {display:none !important;}
.block-container{max-width:1100px;margin:0 auto;padding-top:1.4rem;padding-bottom:3rem;}
h1,h2,h3,h4,h5{font-family:'Plus Jakarta Sans',sans-serif;color:var(--qf-text);}

/* --- hero --- */
.qf-hero{
  display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:1rem;
  padding:1.6rem 1.8rem;margin-bottom:1.4rem;border-radius:22px;
  border:1px solid var(--qf-border);
  background:linear-gradient(135deg, rgba(124,58,237,.20), rgba(34,211,238,.10) 60%, rgba(15,23,42,.85));
  box-shadow:0 24px 60px -32px rgba(124,58,237,.75);
  animation:qf-in .45s ease both;
}
.qf-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:2.35rem;line-height:1.1;
  margin:.35rem 0 .4rem;font-weight:800;letter-spacing:-.02em;}
.qf-grad{background:linear-gradient(100deg,var(--qf-violet),var(--qf-cyan));
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.qf-sub{margin:0;color:var(--qf-muted);font-size:1rem;max-width:46ch;}
.qf-chip{display:inline-block;padding:.28rem .7rem;border-radius:999px;font-size:.74rem;
  font-weight:600;letter-spacing:.04em;text-transform:uppercase;
  border:1px solid rgba(124,58,237,.45);background:rgba(124,58,237,.16);color:#D7C6FF;}
.qf-powered{display:inline-block;padding:.42rem .9rem;border-radius:999px;font-size:.82rem;
  border:1px solid rgba(34,211,238,.35);background:rgba(34,211,238,.10);color:#BFEFFB;}
.qf-powered b{color:#fff;}

/* --- cards --- */
.qf-card{
  background:var(--qf-card);border:1px solid var(--qf-border);border-radius:var(--qf-radius);
  padding:1.15rem 1.25rem;backdrop-filter:blur(14px);
  box-shadow:0 18px 40px -30px rgba(0,0,0,.9);margin-bottom:1rem;
  animation:qf-in .35s ease both;
}
.qf-card-title{font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1rem;margin:0 0 .55rem;}
.qf-label{font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:.95rem;
  margin:.2rem 0 .5rem;display:block;}
.qf-muted{color:var(--qf-muted);font-size:.86rem;margin:0;}
.qf-row{display:flex;flex-wrap:wrap;gap:.5rem;margin:.35rem 0 0;}
.qf-stat{padding:.4rem .7rem;border-radius:10px;background:rgba(148,163,184,.10);
  border:1px solid var(--qf-border);font-size:.8rem;color:#CBD5E1;}
.qf-stat b{color:#fff;}
.qf-sep{height:1px;background:var(--qf-border);margin:1rem 0;border:0;}

/* --- sidebar --- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0B1220 0%, #0F172A 100%);
  border-right:1px solid var(--qf-border);
}
[data-testid="stSidebar"] *{font-family:'Inter',sans-serif;}
.qf-side-title{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1rem;
  margin-bottom:.4rem;color:#fff;}
.qf-pill{display:inline-flex;align-items:center;gap:.45rem;padding:.4rem .75rem;
  border-radius:999px;font-size:.8rem;font-weight:600;margin-bottom:.6rem;}
.qf-pill .dot{width:8px;height:8px;border-radius:50%;display:inline-block;}
.qf-pill.ok{background:rgba(34,197,94,.14);border:1px solid rgba(34,197,94,.45);color:#8EF0B4;}
.qf-pill.ok .dot{background:var(--qf-green);box-shadow:0 0 0 3px rgba(34,197,94,.18);}
.qf-pill.bad{background:rgba(244,63,94,.14);border:1px solid rgba(244,63,94,.45);color:#FFB3C1;}
.qf-pill.bad .dot{background:var(--qf-red);box-shadow:0 0 0 3px rgba(244,63,94,.18);}
.qf-pill.warn{background:rgba(234,179,8,.14);border:1px solid rgba(234,179,8,.45);color:#FCE38A;}
.qf-pill.warn .dot{background:#EAB308;}

/* --- widgets --- */
.stTextArea textarea{
  background:rgba(15,23,42,.78) !important;color:var(--qf-text) !important;
  border-radius:14px !important;border:1px solid var(--qf-border) !important;
  font-family:'Inter',sans-serif !important;font-size:.94rem !important;
  line-height:1.65 !important;padding:.9rem 1rem !important;
}
.stTextArea textarea::placeholder{color:#64748B !important;}
.stTextArea textarea:focus{border-color:rgba(124,58,237,.75) !important;
  box-shadow:0 0 0 3px rgba(124,58,237,.18) !important;}
div[data-baseweb="select"] > div{
  background:rgba(15,23,42,.78) !important;border:1px solid var(--qf-border) !important;
  border-radius:12px !important;color:var(--qf-text) !important;
}
div[data-baseweb="popover"] li{background:#111B2E !important;color:var(--qf-text) !important;}

.stButton > button, .stDownloadButton > button{
  border-radius:12px !important;border:1px solid var(--qf-border) !important;
  background:rgba(148,163,184,.10) !important;color:var(--qf-text) !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;font-weight:600 !important;
  padding:.55rem 1rem !important;
  transition:transform .15s ease, box-shadow .2s ease, border-color .2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform:translateY(-1px);border-color:rgba(124,58,237,.65) !important;
  box-shadow:0 12px 26px -16px rgba(124,58,237,.95) !important;
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(110deg,var(--qf-violet),var(--qf-cyan)) !important;
  color:#08101F !important;border:none !important;
  box-shadow:0 14px 30px -18px rgba(124,58,237,.95) !important;
}
.stButton > button:disabled{opacity:.45 !important;}

.stProgress > div > div > div{background:rgba(148,163,184,.16) !important;border-radius:99px;}
.stProgress > div > div > div > div{
  background:linear-gradient(90deg,var(--qf-violet),var(--qf-cyan)) !important;
  border-radius:99px;}
.qf-progress-label{font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;
  font-size:.85rem;color:#CBD5E1;margin-bottom:.35rem;letter-spacing:.02em;}

[data-testid="stFileUploaderDropzone"]{
  background:rgba(124,58,237,.07) !important;
  border:1.5px dashed rgba(124,58,237,.55) !important;border-radius:14px !important;
  padding:1rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover{background:rgba(124,58,237,.12) !important;}
[data-testid="stFileUploader"] small{color:var(--qf-muted) !important;}

/* --- tabs --- */
.stTabs [data-baseweb="tab-list"]{
  gap:.35rem;background:rgba(15,23,42,.62);padding:.42rem;border-radius:14px;
  border:1px solid var(--qf-border);
}
.stTabs [data-baseweb="tab"]{
  border-radius:10px;padding:.5rem 1rem;color:var(--qf-muted);
  font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:.92rem;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(115deg,var(--qf-violet),var(--qf-cyan)) !important;
  color:#08101F !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}

[data-testid="stExpander"]{
  border:1px solid var(--qf-border) !important;border-radius:14px !important;
  background:rgba(148,163,184,.05) !important;margin-bottom:.5rem;
}
[data-testid="stExpander"] summary{font-family:'Inter',sans-serif;font-weight:600;font-size:.9rem;}

/* --- quiz question --- */
.qf-question{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.28rem;font-weight:700;
  line-height:1.45;margin:.9rem 0 1.1rem;color:#F4F8FF;}
.qf-opt-list{display:flex;flex-direction:column;gap:.6rem;margin:.3rem 0 .9rem;}
.qf-opt{display:flex;align-items:center;gap:.75rem;padding:.8rem 1rem;border-radius:14px;
  border:1px solid var(--qf-border);background:rgba(148,163,184,.07);
  transition:border-color .18s ease, background .18s ease, transform .18s ease;}
.qf-opt:hover{transform:translateX(2px);border-color:rgba(124,58,237,.55);}
.qf-opt-key{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.8rem;
  width:26px;height:26px;min-width:26px;display:flex;align-items:center;justify-content:center;
  border-radius:8px;background:rgba(124,58,237,.20);color:#D7C6FF;}
.qf-opt-text{flex:1;font-size:.95rem;line-height:1.5;color:#E2E8F0;}
.qf-opt.correct{border-color:rgba(34,197,94,.7);background:rgba(34,197,94,.14);}
.qf-opt.correct .qf-opt-key{background:rgba(34,197,94,.3);color:#C9FBD9;}
.qf-opt.wrong{border-color:rgba(244,63,94,.7);background:rgba(244,63,94,.13);}
.qf-opt.wrong .qf-opt-key{background:rgba(244,63,94,.3);color:#FFD5DD;}
.qf-tag{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
  padding:.2rem .5rem;border-radius:999px;white-space:nowrap;}
.qf-tag.ok{background:rgba(34,197,94,.2);color:#8EF0B4;border:1px solid rgba(34,197,94,.4);}
.qf-tag.bad{background:rgba(244,63,94,.2);color:#FFB3C1;border:1px solid rgba(244,63,94,.4);}

.qf-verdict{display:inline-block;font-weight:700;font-size:.88rem;padding:.4rem .8rem;
  border-radius:10px;margin-bottom:.7rem;}
.qf-verdict.ok{background:rgba(34,197,94,.14);color:#8EF0B4;border:1px solid rgba(34,197,94,.45);}
.qf-verdict.bad{background:rgba(244,63,94,.14);color:#FFB3C1;border:1px solid rgba(244,63,94,.45);}
.qf-explain{border-left:3px solid var(--qf-violet);background:rgba(124,58,237,.10);
  border-radius:0 12px 12px 0;padding:.85rem 1rem;font-size:.92rem;line-height:1.6;color:#DDE7F5;}
.qf-explain b{color:#fff;}

/* --- results --- */
.qf-ring{width:150px;height:150px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;margin:.4rem auto;box-shadow:0 0 40px -18px rgba(124,58,237,.9);}
.qf-ring-inner{width:118px;height:118px;border-radius:50%;background:#0B1220;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.1rem;}
.qf-ring-num{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.7rem;font-weight:800;color:#fff;}
.qf-ring-sub{font-size:.78rem;color:var(--qf-muted);}

/* --- flashcard --- */
.qf-flip{perspective:1500px;width:100%;max-width:660px;margin:.4rem auto 1rem;height:320px;}
.qf-flip-inner{position:relative;width:100%;height:100%;transform-style:preserve-3d;
  transition:transform .65s cubic-bezier(.4,.2,.2,1);}
.qf-flip-inner.is-flipped{transform:rotateY(180deg);}
.qf-face{position:absolute;inset:0;backface-visibility:hidden;-webkit-backface-visibility:hidden;
  border-radius:20px;border:1px solid var(--qf-border);padding:1.5rem;
  display:flex;flex-direction:column;justify-content:center;gap:.6rem;overflow:auto;
  box-shadow:0 26px 60px -34px rgba(0,0,0,.95);}
.qf-face.front{background:linear-gradient(150deg, rgba(124,58,237,.24), rgba(11,18,32,.96));}
.qf-face.back{transform:rotateY(180deg);
  background:linear-gradient(150deg, rgba(34,211,238,.20), rgba(11,18,32,.96));}
.qf-face-tag{font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;
  color:var(--qf-muted);}
.qf-face-text{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.12rem;line-height:1.5;
  font-weight:600;color:#F1F5FF;}
.qf-flip-note{text-align:center;color:var(--qf-muted);font-size:.8rem;margin-bottom:.6rem;}

/* --- empty state --- */
.qf-empty{border:1px dashed rgba(148,163,184,.32);border-radius:var(--qf-radius);
  padding:2.2rem 1.5rem;text-align:center;background:rgba(148,163,184,.04);margin:.6rem 0;}
.qf-empty-icon{font-size:2rem;display:block;margin-bottom:.5rem;}
.qf-empty-title{font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;
  font-size:1.05rem;color:#fff;margin-bottom:.3rem;}
.qf-empty-text{color:var(--qf-muted);font-size:.9rem;margin:0;}

.qf-foot{color:#64748B;font-size:.76rem;text-align:center;margin-top:1.6rem;line-height:1.6;}
.qf-note{color:#94A3B8;font-size:.82rem;margin:.4rem 0 0;}

.qf-steps{margin:.4rem 0 .8rem;padding-left:1.1rem;color:#DDE7F5;font-size:.9rem;line-height:1.75;}
.qf-steps li{margin-bottom:.25rem;}
.qf-pre{margin:.2rem 0 .7rem;padding:.75rem .9rem;border-radius:12px;
  background:rgba(11,18,32,.85);border:1px solid var(--qf-border);color:#8EF0B4;
  font-family:'SFMono-Regular',Consolas,monospace;font-size:.82rem;overflow-x:auto;}

@keyframes qf-in{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}

@media (max-width:760px){
  .block-container{padding-left:1rem;padding-right:1rem;}
  .qf-hero{padding:1.2rem 1.1rem;}
  .qf-title{font-size:1.75rem;}
  .qf-flip{height:380px;}
  .qf-question{font-size:1.1rem;}
  .qf-ring{width:130px;height:130px;}
  .qf-ring-inner{width:102px;height:102px;}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# === CHUNK MARKER ===

# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------

LETTERS = ["A", "B", "C", "D"]


def esc(value: Any) -> str:
    """Escape text for safe HTML embedding."""
    return html.escape(str(value), quote=True)


def empty_state(icon: str, title: str, text: str) -> None:
    st.markdown(
        f'<div class="qf-empty"><span class="qf-empty-icon">{icon}</span>'
        f'<div class="qf-empty-title">{esc(title)}</div>'
        f'<p class="qf-empty-text">{esc(text)}</p></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="qf-card"><div class="qf-card-title">{esc(title)}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return cleaned[:40] or "quizforge-quiz"


def format_duration(seconds: float | None) -> str:
    if not seconds or seconds < 0:
        return "0s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m {total % 60}s"


def render_hero() -> None:
    st.markdown(
        """
<div class="qf-hero">
  <div>
    <span class="qf-chip">AI study workspace</span>
    <div class="qf-title">Quiz<span class="qf-grad">Forge</span> AI</div>
    <p class="qf-sub">Turn your class notes into exam-ready quizzes and flashcards in seconds.</p>
  </div>
  <div class="qf-powered">Powered by <b>Groq</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


def drain_pending_notes() -> None:
    """Apply text extracted from an upload before any widget claims its key."""
    pending = st.session_state.get("pending_notes")
    if pending is not None:
        st.session_state.notes_text = pending
        st.session_state.pending_notes = None
        st.session_state.allow_trim = len(pending) <= MAX_NOTES_CHARS


def onboarding_panel() -> None:
    st.markdown(
        """
<div class="qf-card">
  <div class="qf-card-title">🔑 Connect your Groq API key</div>
  <ol class="qf-steps">
    <li>Create a free key at <b>console.groq.com/keys</b> .</li>
    <li>Add it to <b>.streamlit/secrets.toml</b> beside <b>app.py</b> (create both if missing).</li>
    <li>Reload this page — the sidebar pill turns green and generation unlocks.</li>
  </ol>
  <pre class="qf-pre">GROQ_API_KEY = "gsk_your_key_here"</pre>
  <p class="qf-note">Deploying to Streamlit Cloud? Add the same line under Settings → Secrets,
  then reboot the app. Never commit the real key to GitHub.</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def pill(label: str, state: str) -> str:
    return f'<div class="qf-pill {state}"><span class="dot"></span>{esc(label)}</div>'


def render_sidebar(ready: bool, status_label: str) -> None:
    with st.sidebar:
        st.markdown('<div class="qf-side-title">⚙️ Settings</div>', unsafe_allow_html=True)

        if ready:
            st.markdown(pill(status_label, "ok"), unsafe_allow_html=True)
        elif status_label == "API Key Missing":
            st.markdown(pill(status_label, "bad"), unsafe_allow_html=True)
        elif status_label == "API Key Invalid":
            st.markdown(pill(status_label, "bad"), unsafe_allow_html=True)
        else:
            st.markdown(pill(status_label, "warn"), unsafe_allow_html=True)

        st.selectbox("Difficulty", DIFFICULTIES, key="difficulty")
        st.selectbox("Number of questions", QUESTION_COUNTS, key="num_questions")
        st.selectbox("Question style", STYLES, key="style")
        st.selectbox("Output language", LANGUAGES, key="language")
        st.selectbox("Model", list(MODEL_LABELS.keys()), key="model_choice")
        st.caption(f"Model ID: {MODEL_LABELS[st.session_state.model_choice]}")

        st.markdown('<hr class="qf-sep" />', unsafe_allow_html=True)

        with st.expander("How to use"):
            st.markdown(
                "1. Paste notes in **📝 Create** (or upload a file).\n"
                "2. Pick difficulty, count, style and language here.\n"
                "3. Press **Generate Quiz**, check the summary card.\n"
                "4. Answer in **🎯 Quiz** — the explanation appears after each pick.\n"
                "5. Revise in **🃏 Flashcards**, then export in **📤 Export**."
            )

        with st.expander("Privacy"):
            st.markdown(
                "Notes are sent to Groq's API only while a quiz is generating. "
                "Nothing is stored on a server — your quiz lives in this browser session."
            )

        if st.button("Clear everything", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ("difficulty", "num_questions", "style", "language", "model_choice"):
                    del st.session_state[key]
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 1 — Create
# ---------------------------------------------------------------------------


def generate_quiz(ready: bool) -> None:
    """Validate input, call the model, validate output, store the quiz."""
    notes = st.session_state.notes_text.strip()
    count = int(st.session_state.num_questions)

    if not ready:
        st.session_state.last_error = KEY_MISSING_MESSAGE
        st.rerun()
        return

    if len(notes) < MIN_NOTES_CHARS:
        st.warning(
            f"Please paste at least {MIN_NOTES_CHARS} characters of notes — "
            "a few sentences about one topic works best."
        )
        return

    if len(notes) > MAX_NOTES_CHARS and not st.session_state.allow_trim:
        st.warning(
            f"Your notes are {len(notes):,} characters. The model works best under "
            f"{MAX_NOTES_CHARS:,}. Trim the text to the key section, or shorten it here."
        )
        if st.button(f"Use the first {MAX_NOTES_CHARS:,} characters", use_container_width=True):
            st.session_state.pending_notes = notes[:MAX_NOTES_CHARS]
            st.session_state.allow_trim = True
            st.rerun()
        return

    st.session_state.last_error = ""
    model = MODEL_LABELS[st.session_state.model_choice]

    try:
        with st.status("Analysing your notes…", expanded=False) as status:
            status.update(label=f"Writing {count} questions…", state="running")
            result = call_llm(
                notes=notes[:MAX_NOTES_CHARS],
                count=count,
                difficulty=st.session_state.difficulty,
                style=st.session_state.style,
                language=st.session_state.language,
                model=model,
            )
            status.update(label="Checking the answers…", state="running")
            quiz, warning = validate_quiz(result["payload"], count)
            if quiz is None:
                st.session_state.last_error = warning
                status.update(label="Generation failed", state="error")
                st.rerun()
            meta = {
                "model": result["model"],
                "elapsed": result["elapsed"],
                "notices": result["notices"],
                "warning": warning,
                "requested": count,
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
            status.update(label="Quiz ready", state="complete")
    except QuizForgeError as exc:
        st.session_state.last_error = str(exc)
        st.rerun()
        return
    except Exception as exc:  # genuinely unexpected — never white-screen
        print(f"[quizforge] unexpected generation error: {type(exc).__name__}")
        st.session_state.last_error = (
            "Something unexpected interrupted the generation. Please press Try again."
        )
        st.rerun()
        return

    st.session_state.quiz_data = quiz
    st.session_state.quiz_meta = meta
    reset_progress()
    reset_flashcards()
    st.toast(f"Quiz ready — {len(quiz['questions'])} questions on {quiz['topic']}", icon="✅")
    st.rerun()


def render_create_tab(ready: bool) -> None:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
        if st.button("Try again", key="retry_generate"):
            st.session_state.last_error = ""
            st.rerun()

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        st.markdown('<div class="qf-label">Paste your notes here</div>', unsafe_allow_html=True)
        st.text_area(
            "Paste your notes here",
            key="notes_text",
            height=300,
            label_visibility="collapsed",
            placeholder="Paste lesson notes, a textbook summary, lecture slides or an article…",
        )

        uploaded = st.file_uploader(
            "Or upload a file",
            type=["txt", "pdf", "docx"],
            key="uploader",
            help="Text, PDF or Word documents. Files stay in your browser session.",
        )
        if uploaded is not None:
            signature = f"{uploaded.name}:{uploaded.size}"
            if signature != st.session_state.uploaded_name:
                try:
                    text = extract_text_from_file(uploaded.name, uploaded.getvalue())
                    st.session_state.uploaded_name = signature
                    st.session_state.pending_notes = text
                    st.toast(f"Loaded {uploaded.name}", icon="📄")
                    st.rerun()
                except QuizForgeError as exc:
                    st.error(str(exc))

        notes = st.session_state.notes_text or ""
        words = len(notes.split())
        minutes = max(1, round(words / READING_WORDS_PER_MINUTE)) if words else 0
        st.markdown(
            '<div class="qf-row">'
            f'<span class="qf-stat">Words <b>{words:,}</b></span>'
            f'<span class="qf-stat">Characters <b>{len(notes):,}</b></span>'
            f'<span class="qf-stat">Read time <b>~{minutes} min</b></span>'
            f'<span class="qf-stat">Limit <b>{MAX_NOTES_CHARS:,}</b></span>'
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        if not ready:
            onboarding_panel()

        summary = "".join(
            [
                f'<span class="qf-stat">Difficulty <b>{esc(st.session_state.difficulty)}</b></span>',
                f'<span class="qf-stat">Questions <b>{st.session_state.num_questions}</b></span>',
                f'<span class="qf-stat">Style <b>{esc(st.session_state.style)}</b></span>',
                f'<span class="qf-stat">Language <b>{esc(st.session_state.language)}</b></span>',
                f'<span class="qf-stat">Model <b>{esc(st.session_state.model_choice)}</b></span>',
            ]
        )
        card(
            "Your setup",
            f'<div class="qf-row">{summary}</div>'
            '<p class="qf-note">Change any of these in the sidebar. Settings are saved '
            "across the session, so you can generate a different mix any time.</p>",
        )

        if ready:
            if st.button("Generate Quiz", type="primary", use_container_width=True):
                generate_quiz(ready)
        else:
            st.button("Generate Quiz", type="primary", use_container_width=True, disabled=True)
            st.caption("Add a Groq API key to unlock generation.")

        if st.session_state.quiz_data:
            st.caption("Generating again replaces the current quiz and resets your answers.")

    quiz = st.session_state.quiz_data
    meta = st.session_state.quiz_meta or {}
    if not quiz:
        st.markdown('<hr class="qf-sep" />', unsafe_allow_html=True)
        empty_state(
            "🧩",
            "No quiz yet",
            "Paste at least a paragraph of notes on the left, then press Generate Quiz.",
        )
        return

    st.markdown('<hr class="qf-sep" />', unsafe_allow_html=True)
    cards = len(quiz["flashcards"])
    facts = "".join(
        [
            f'<span class="qf-stat">Questions <b>{len(quiz["questions"])}</b></span>',
            f'<span class="qf-stat">Flashcards <b>{cards}</b></span>',
            f'<span class="qf-stat">Difficulty <b>{esc(st.session_state.difficulty)}</b></span>',
            f'<span class="qf-stat">Model <b>{esc(meta.get("model", "—"))}</b></span>',
            f'<span class="qf-stat">Built in <b>{format_duration(meta.get("elapsed"))}</b></span>',
        ]
    )
    card(
        f"Topic detected: {quiz['topic']}",
        f'<div class="qf-row">{facts}</div>',
    )

    if meta.get("warning"):
        st.warning(meta["warning"])
    for notice in meta.get("notices", []):
        st.info(notice)

    action_left, action_right = st.columns(2)
    with action_left:
        if st.button("Start Quiz", type="primary", use_container_width=True):
            reset_progress()
            st.toast("Quiz reset — open the 🎯 Quiz tab to begin.", icon="🎯")
            st.rerun()
    with action_right:
        if st.button("Regenerate", use_container_width=True):
            generate_quiz(ready)


# ---------------------------------------------------------------------------
# Tab 2 — Quiz
# ---------------------------------------------------------------------------


def ensure_answers(total: int) -> list[int | None]:
    answers = st.session_state.user_answers
    while len(answers) < total:
        answers.append(None)
    return answers


def finish_quiz(quiz: dict[str, Any]) -> None:
    questions = quiz["questions"]
    answers = ensure_answers(len(questions))
    st.session_state.score = sum(
        1 for i, question in enumerate(questions) if answers[i] == question["answer_index"]
    )
    st.session_state.finished_at = time.time()
    st.session_state.quiz_finished = True


def build_verdict(quiz: dict[str, Any], answers: list[int | None]) -> str:
    """Plain-language study verdict derived from the attempt (no extra API call)."""
    questions = quiz["questions"]
    total = len(questions)
    correct = sum(
        1 for i, question in enumerate(questions) if answers[i] == question["answer_index"]
    )
    pct = round(correct / total * 100) if total else 0

    buckets: dict[str, list[int]] = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}
    for i, question in enumerate(questions):
        bucket = buckets.get(question["difficulty"], buckets["medium"])
        bucket[1] += 1
        if answers[i] == question["answer_index"]:
            bucket[0] += 1

    scored = {
        name: (hit / seen if seen else None)
        for name, (hit, seen) in buckets.items()
        if seen
    }
    ordered = sorted(scored.items(), key=lambda item: item[1] or 0)
    head = f"You answered {correct} of {total} correctly ({pct}%)."
    if len(ordered) < 2:
        return head + " Solid work — try a higher difficulty next round."
    weakest, weakest_rate = ordered[0]
    strongest, strongest_rate = ordered[-1]
    if strongest_rate == weakest_rate:
        return head + " Even performance across difficulty levels."
    return (
        head
        + f" Strongest on {strongest} questions ({round(strongest_rate * 100)}%), "
        + f"shakiest on {weakest} ones ({round(weakest_rate * 100)}%). "
        + f"Review the {weakest} items before your next attempt."
    )


def review_block(quiz: dict[str, Any], answers: list[int | None]) -> None:
    for i, question in enumerate(quiz["questions"], start=1):
        chosen = answers[i - 1] if i - 1 < len(answers) else None
        is_right = chosen == question["answer_index"]
        mark = "✅" if is_right else "❌"
        your_text = (
            question["options"][chosen] if isinstance(chosen, int) and 0 <= chosen <= 3 else "Not answered"
        )
        with st.expander(f"{mark}  Q{i}. {question['question'][:90]}"):
            st.markdown(f"**Your answer:** {your_text}")
            st.markdown(f"**Correct answer:** {question['options'][question['answer_index']]}")
            st.markdown(f"**Why:** {question['explanation']}")
            st.caption(f"Difficulty: {question['difficulty'].title()}")


def render_results(quiz: dict[str, Any]) -> None:
    questions = quiz["questions"]
    answers = ensure_answers(len(questions))
    total = len(questions)
    score = st.session_state.score
    pct = round(score / total * 100) if total else 0
    elapsed = None
    if st.session_state.started_at and st.session_state.finished_at:
        elapsed = st.session_state.finished_at - st.session_state.started_at

    ring = (
        f"background:conic-gradient(from 210deg, #7C3AED 0%, #22D3EE {pct}%,"
        f" rgba(148,163,184,.18) {pct}% 100%);"
    )
    st.markdown(
        f'<div class="qf-card"><div class="qf-card-title">Your result</div>'
        f'<div class="qf-ring" style="{ring}"><div class="qf-ring-inner">'
        f'<div class="qf-ring-num">{score}/{total}</div>'
        f'<div class="qf-ring-sub">{pct}%</div></div></div>'
        f'<p class="qf-note" style="text-align:center;">Time taken: '
        f"{format_duration(elapsed)}</p></div>",
        unsafe_allow_html=True,
    )
    st.progress(pct / 100, text=f"Score: {pct}%")

    verdict = build_verdict(quiz, answers)
    state = "ok" if pct >= 60 else "bad"
    st.markdown(
        f'<div class="qf-card"><div class="qf-card-title">Study verdict</div>'
        f'<div class="qf-verdict {state}">{pct}% correct</div>'
        f'<p class="qf-note">{esc(verdict)}</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Per-question review")
    review_block(quiz, answers)

    csv_data = build_csv(quiz, answers)
    first, second, third = st.columns(3)
    with first:
        if st.button("Retake Quiz", type="primary", use_container_width=True):
            reset_progress()
            st.rerun()
    with second:
        if st.button("New Quiz", use_container_width=True):
            st.session_state.quiz_data = None
            st.session_state.quiz_meta = None
            reset_progress()
            reset_flashcards()
            st.toast("Cleared — paste new notes in the Create tab.", icon="🧹")
            st.rerun()
    with third:
        st.download_button(
            "Download results (CSV)",
            data=csv_data,
            file_name=f"quizforge-results-{slugify(quiz['topic'])}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_quiz_tab() -> None:
    quiz = st.session_state.quiz_data
    if not quiz:
        empty_state(
            "🎯",
            "No quiz loaded",
            "Generate a quiz in the Create tab first — it takes about ten seconds.",
        )
        return

    questions = quiz["questions"]
    total = len(questions)

    if st.session_state.quiz_finished:
        render_results(quiz)
        return

    index = max(0, min(st.session_state.current_index, total - 1))
    st.session_state.current_index = index
    answers = ensure_answers(total)

    if st.session_state.started_at is None:
        st.session_state.started_at = time.time()

    question = questions[index]
    question_on = index + 1
    answered = st.session_state.answered

    st.markdown(
        f'<div class="qf-progress-label">Question {question_on} of {total}'
        f'<span style="float:right;color:#94A3B8;font-weight:600;">'
        f"{question['difficulty'].title()}</span></div>",
        unsafe_allow_html=True,
    )
    st.progress(question_on / total)

    st.markdown(f'<div class="qf-question">{esc(question["question"])}</div>', unsafe_allow_html=True)

    if not answered:
        for i, option in enumerate(question["options"]):
            if st.button(
                f"{LETTERS[i]}.   {option}",
                key=f"opt_{index}_{i}",
                use_container_width=True,
            ):
                answers[index] = i
                st.session_state.user_answers = answers
                st.session_state.answered = True
                st.rerun()
    else:
        chosen = answers[index]
        rows: list[str] = []
        for i, option in enumerate(question["options"]):
            classes = "qf-opt"
            tag = ""
            if i == question["answer_index"]:
                classes += " correct"
                tag = '<span class="qf-tag ok">Correct</span>'
            if i == chosen and chosen != question["answer_index"]:
                classes += " wrong"
                tag = '<span class="qf-tag bad">Your pick</span>'
            elif i == chosen:
                tag = '<span class="qf-tag ok">Your pick</span>'
            rows.append(
                f'<div class="{classes}"><span class="qf-opt-key">{LETTERS[i]}</span>'
                f'<span class="qf-opt-text">{esc(option)}</span>{tag}</div>'
            )
        st.markdown(f'<div class="qf-opt-list">{"".join(rows)}</div>', unsafe_allow_html=True)

        is_right = chosen == question["answer_index"]
        headline = "Correct" if is_right else "Not quite"
        st.markdown(
            f'<div class="qf-explain"><b>{headline}.</b> {esc(question["explanation"])}</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        if index < total - 1:
            if st.button("Next Question →", type="primary", use_container_width=True):
                st.session_state.current_index = index + 1
                st.session_state.answered = False
                st.rerun()
        else:
            if st.button("Finish Quiz", type="primary", use_container_width=True):
                finish_quiz(quiz)
                st.rerun()

        if st.button("Restart this quiz", key="restart_midway", use_container_width=True):
            reset_progress()
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — Flashcards
# ---------------------------------------------------------------------------


def render_flashcards_tab() -> None:
    quiz = st.session_state.quiz_data
    if not quiz or not quiz.get("flashcards"):
        empty_state(
            "🃏",
            "No flashcards yet",
            "Generate a quiz and its flashcards appear here automatically — no second API call.",
        )
        return

    cards = quiz["flashcards"]
    if len(st.session_state.flash_order) != len(cards):
        st.session_state.flash_order = list(range(len(cards)))
    order: list[int] = st.session_state.flash_order
    known: set[int] = st.session_state.flash_known

    review_only = st.toggle("Review only the cards I did not know", key="flash_filter")
    pool = [i for i in order if i not in known] if review_only else list(order)

    if not pool:
        empty_state("🎉", "Every card is marked as known", "Turn off the review filter to see the whole deck again.")
        if st.button("Reset known cards", use_container_width=True):
            st.session_state.flash_known = set()
            st.session_state.flash_pos = 0
            st.rerun()
        return

    position = max(0, min(st.session_state.flash_pos, len(pool) - 1))
    st.session_state.flash_pos = position
    card_index = pool[position]
    current = cards[card_index]
    flipped = st.session_state.flash_flipped

    st.markdown(
        f'<div class="qf-flip-note">Card {position + 1} of {len(pool)}'
        f" &middot; {len(known)} of {len(cards)} marked known</div>",
        unsafe_allow_html=True,
    )
    flip_class = "qf-flip-inner is-flipped" if flipped else "qf-flip-inner"
    st.markdown(
        f'<div class="qf-flip"><div class="{flip_class}">'
        f'<div class="qf-face front"><span class="qf-face-tag">Question</span>'
        f'<div class="qf-face-text">{esc(current["front"])}</div></div>'
        f'<div class="qf-face back"><span class="qf-face-tag">Answer</span>'
        f'<div class="qf-face-text">{esc(current["back"])}</div></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    row1, row2 = st.columns(2)
    with row1:
        if st.button("Flip card", type="primary", use_container_width=True):
            st.session_state.flash_flipped = not flipped
            st.rerun()
    with row2:
        if st.button("Shuffle", use_container_width=True):
            shuffled = list(order)
            random.shuffle(shuffled)
            st.session_state.flash_order = shuffled
            st.session_state.flash_pos = 0
            st.session_state.flash_flipped = False
            st.toast("Deck shuffled — your answers are untouched.", icon="🔀")
            st.rerun()

    row3, row4 = st.columns(2)
    with row3:
        if st.button("← Previous", use_container_width=True):
            st.session_state.flash_pos = (position - 1) % len(pool)
            st.session_state.flash_flipped = False
            st.rerun()
    with row4:
        if st.button("Next →", use_container_width=True):
            st.session_state.flash_pos = (position + 1) % len(pool)
            st.session_state.flash_flipped = False
            st.rerun()

    row5, row6 = st.columns(2)
    with row5:
        if st.button("I knew it", use_container_width=True):
            updated = set(known)
            updated.add(card_index)
            st.session_state.flash_known = updated
            st.session_state.flash_pos = min(position + 1, len(pool) - 1)
            st.session_state.flash_flipped = False
            st.rerun()
    with row6:
        if st.button("Show again", use_container_width=True):
            updated = set(known)
            updated.discard(card_index)
            st.session_state.flash_known = updated
            st.session_state.flash_pos = min(position + 1, len(pool) - 1)
            st.session_state.flash_flipped = False
            st.rerun()


# ---------------------------------------------------------------------------
# Export builders
# ---------------------------------------------------------------------------


def build_csv(quiz: dict[str, Any], answers: list[int | None]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "question",
            "option_a",
            "option_b",
            "option_c",
            "option_d",
            "correct_option",
            "correct_answer",
            "your_answer",
            "is_correct",
            "difficulty",
            "explanation",
        ]
    )
    for i, question in enumerate(quiz["questions"]):
        chosen = answers[i] if i < len(answers) else None
        writer.writerow(
            [
                i + 1,
                question["question"],
                *question["options"],
                LETTERS[question["answer_index"]],
                question["options"][question["answer_index"]],
                question["options"][chosen] if isinstance(chosen, int) else "",
                "yes" if chosen == question["answer_index"] else ("no" if chosen is not None else ""),
                question["difficulty"],
                question["explanation"],
            ]
        )
    return buffer.getvalue()


def build_json(quiz: dict[str, Any], answers: list[int | None], meta: dict[str, Any]) -> str:
    payload = {
        "topic": quiz["topic"],
        "generated_at": meta.get("generated_at"),
        "model": meta.get("model"),
        "questions": [
            {
                "id": i + 1,
                "question": question["question"],
                "options": question["options"],
                "answer_index": question["answer_index"],
                "correct_answer": question["options"][question["answer_index"]],
                "your_answer": answers[i] if i < len(answers) and isinstance(answers[i], int) else None,
                "explanation": question["explanation"],
                "difficulty": question["difficulty"],
            }
            for i, question in enumerate(quiz["questions"])
        ],
        "flashcards": quiz["flashcards"],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_markdown(quiz: dict[str, Any]) -> str:
    lines = [f"# {quiz['topic']}", "", f"_{len(quiz['questions'])} questions · QuizForge AI_", ""]
    for i, question in enumerate(quiz["questions"], start=1):
        lines.append(f"## Q{i}. {question['question']}")
        lines.append("")
        for letter, option in zip(LETTERS, question["options"]):
            marker = "**" if LETTERS[question["answer_index"]] == letter else ""
            lines.append(f"- {letter}. {marker}{option}{marker}")
        lines.append("")
        lines.append(f"**Answer:** {LETTERS[question['answer_index']]}")
        lines.append("")
        lines.append(f"**Explanation:** {question['explanation']}")
        lines.append("")
    lines.append("## Flashcards")
    lines.append("")
    for flashcard in quiz["flashcards"]:
        lines.append(f"- **{flashcard['front']}** — {flashcard['back']}")
    return "\n".join(lines) + "\n"


def build_anki(quiz: dict[str, Any]) -> str:
    lines = [f"{flashcard['front']}\t{flashcard['back']}" for flashcard in quiz["flashcards"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tab 4 — Export
# ---------------------------------------------------------------------------


def render_export_tab() -> None:
    quiz = st.session_state.quiz_data
    if not quiz:
        empty_state(
            "📤",
            "Nothing to export yet",
            "Generate a quiz first — CSV, JSON, Markdown and Anki text all appear here.",
        )
        return

    meta = st.session_state.quiz_meta or {}
    answers = st.session_state.user_answers
    base = slugify(quiz["topic"])

    card(
        "Export your material",
        '<div class="qf-row">'
        f'<span class="qf-stat">Questions <b>{len(quiz["questions"])}</b></span>'
        f'<span class="qf-stat">Flashcards <b>{len(quiz["flashcards"])}</b></span>'
        f'<span class="qf-stat">Generated <b>{esc(meta.get("generated_at", "—"))}</b></span>'
        "</div>",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "CSV",
            data=build_csv(quiz, answers),
            file_name=f"{base}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "JSON",
            data=build_json(quiz, answers, meta),
            file_name=f"{base}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "Markdown",
            data=build_markdown(quiz),
            file_name=f"{base}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown('<hr class="qf-sep" />', unsafe_allow_html=True)
    st.markdown("### Anki-style flashcards")
    st.caption("Two columns separated by a tab. Paste into a .txt file and import it into Anki.")
    anki = build_anki(quiz)
    st.code(anki, language="text")
    st.download_button(
        "Download Anki text (.txt)",
        data=anki,
        file_name=f"{base}-anki.txt",
        mime="text/plain",
    )


# ---------------------------------------------------------------------------
# Footer + entry point
# ---------------------------------------------------------------------------


def render_footer() -> None:
    meta = st.session_state.quiz_meta or {}
    if not meta:
        st.markdown(
            '<div class="qf-foot">QuizForge AI · notes in, quizzes out · built with Streamlit + Groq</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        '<div class="qf-foot">'
        f"Model: {esc(meta.get('model', '—'))} · generated in "
        f"{format_duration(meta.get('elapsed'))} · {esc(meta.get('generated_at', ''))}"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="QuizForge AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    inject_css()

    ready, status_label = api_ready()

    render_hero()
    render_sidebar(ready, status_label)
    drain_pending_notes()

    tab_create, tab_quiz, tab_flash, tab_export = st.tabs(
        ["📝 Create", "🎯 Quiz", "🃏 Flashcards", "📤 Export"]
    )
    with tab_create:
        render_create_tab(ready)
    with tab_quiz:
        render_quiz_tab()
    with tab_flash:
        render_flashcards_tab()
    with tab_export:
        render_export_tab()

    render_footer()


try:
    main()
except Exception as unexpected:  # last-resort guard: never white-screen
    print(f"[quizforge] fatal error: {type(unexpected).__name__}: {unexpected}")
    st.error(
        "QuizForge hit an unexpected problem while rendering. "
        "Reload the page, or press Clear everything in the sidebar."
    )


# ---------------------------------------------------------------------------
# Deployment — GitHub → Streamlit Community Cloud
#
# 1. Create a public GitHub repo and add app.py plus requirements.txt.
# 2. Add a .gitignore that excludes secrets.toml and .env (create those files
#    locally yourself — never commit them).
# 3. Go to share.streamlit.io → "New app" → pick the repo, branch and app.py.
# 4. Advanced settings → Secrets → paste (TOML, exact key name):
#        GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxx"
# 5. Deploy. After changing any secret, use "Reboot app" so the new value loads.
# 6. Never commit the real API key to GitHub — anyone can read a public repo.
# ---------------------------------------------------------------------------
