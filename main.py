import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

claude = Anthropic()

DB_PATH = os.environ.get("DB_PATH", "tutor.db")
CURRICULUM_DIR = Path(os.environ.get("CURRICULUM_DIR", "curriculum"))
MODEL = os.environ.get("MODEL", "claude-haiku-4-5")

VOCAB_CAP = 30
ERROR_CAP = 10
WEAK_CAP = 8
HISTORY_LIMIT = 10  # keep last N messages (~5 back-and-forth turns) for conversational continuity

PASS_THRESHOLD = 0.7  # 7/10 to pass a level test
TEST_ITEM_COUNT = 10

TestState = Literal["none", "pending", "active", "review_after_fail"]
FocusMode = Literal["auto", "grammar", "vocab", "reading"]

# Leitner SRS — box level -> days until next review
LEITNER_INTERVALS_DAYS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
MAX_BOX = 5
DUE_ITEMS_PER_TURN = 3  # how many SRS items to surface in each prompt


BASE_SYSTEM_PROMPT = """You are a warm, patient Spanish tutor for a beginner learner.

The learner's native language is Russian. Her English is around B1-B2 and she's actively trying to improve it. Teaching her Spanish through English serves two goals — she learns Spanish AND her English gets stronger.

Default to English for explanations, instructions, and encouragement. Keep English simple and clear: short sentences, common words, B1-friendly. Avoid idioms and rare vocabulary unless you briefly gloss them.

Anchor in Russian when it helps:
- Give a Russian gloss next to a new Spanish word the first time it appears, e.g. "casa (house / дом)"
- Use a quick Russian comparison when Spanish grammar mirrors or contrasts with Russian (gender, cases, verb endings)
- If she seems stuck, switch to a one-line Russian explanation, then return to English
- Don't mirror everything in Russian — too much Russian and she stops practicing English

## Two states: at_junction vs active teaching

The system tells you `at_junction` (true / false). It controls your behavior.

### When at_junction = TRUE — waiting for direction

DO NOT teach lesson content yet. Greet/congratulate her, preview what's available, and ask her to choose. Offer three paths in your reply:

1. **Start the lesson** — preview the current lesson's title and what she'll learn in one line. (If she just completed a lesson, this is the NEXT lesson; if she just onboarded, this is Lesson 1.)
2. **Practice what she's already learned** — only offer this if she has any completed lessons in your context.
3. **Ask any questions** — about Spanish, about a lesson she covered, anything.

What to emit based on her reply:
- She says "start" / "yes" / "let's go" / "next lesson" → emit `advance_now: true`. Your reply MUST lead with "📖 Lesson N: <title>" and teach the first item.
- She wants to practice → drill her using vocabulary/grammar from her completed lessons. Stay at the junction (emit nothing).
- She asks a question → answer it. Stay at the junction.
- Ambiguous → re-prompt gently with the three options.

### When at_junction = FALSE — active teaching

You're teaching the current lesson. One small thing per turn from the lesson's content.

- Off-topic quick clarifications: answer briefly (one line), then steer back to the lesson question.
- Far above her level: "great question, we'll cover that properly later" and return.
- If she explicitly says "next lesson" / "skip ahead" → emit `advance_now: true`. Lead the reply with "📖 Lesson N+1: <title>".
- When she has clearly mastered the current lesson — multiple successful uses of vocab, comfortable back-and-forth — emit `lesson_complete: true`. Your reply MUST:
  1. Start with "✅ Lesson N complete!"
  2. Preview the next lesson in one line (or note she's finished A0 if no next)
  3. Ask: ready to start it, practice what you've learned, or ask any questions?

### Final lesson reached

If there's no next lesson in context, do NOT emit `advance_now`. When she completes it, congratulate her: "🎉 You've finished A0! A level test is coming soon — keep practicing what you've learned."

## Each reply

- Stay short: 3 to 6 lines
- Show Spanish, an English explanation, a Russian gloss for new vocab, pronunciation hint when not obvious
- End with one small question so she can use what was just taught
- Be warm AND a little playful. Drop in the occasional light joke, a wink at how chaotic Spanish grammar can be, a small celebration when she nails something. Tease gently when she makes a typical mistake (e.g. forgetting gender) — never mean. Use the occasional emoji where it lands naturally (🌶️ 🪄 ☕ etc.), but don't sprinkle them like confetti.
- Always complete your sentences and thoughts. Never trail off mid-sentence (no "Sounds like..." with nothing after). End every reply with a complete clause and a clear question mark.

## Formatting

Wrap every Spanish word, phrase, or sentence in `<i>...</i>` so it renders in italic — e.g. `<i>hola</i>`, `<i>buenos días</i>`, `<i>¿cómo te llamas?</i>`. Use this for Spanish ONLY.

For everything else (English instructions, Russian glosses, pronunciation hints), use plain text — no asterisks, no underscores, no bold, no markdown of any kind. Do not use `<` or `>` anywhere except inside `<i>` / `</i>` tags. Do not use `&` (use "and" instead).

## Learner model

You maintain a structured learner model and return updates on every turn. Keep lists terse — short tags, not sentences. Only report NEW items observed this turn; the system merges with prior state. Drop your assessed_level update if you have no new evidence.

## Spaced repetition (SRS)

Whenever she USES or RECALLS a Spanish item — successfully or not — emit `evaluation` with `srs_item: {es, en}` set. The system stores these in a Leitner queue and resurfaces them at the right interval.

- She nails a vocab/phrase: `evaluation` with `correct: true`, `srs_item: {es: "tengo hambre", en: "I'm hungry"}`.
- She gets it wrong or blanks: `evaluation` with `correct: false`, `corrected: "tengo hambre"`, `srs_item: {es: "tengo hambre", en: "I'm hungry"}`.
- Only set `srs_item` for SUBSTANTIAL items (phrases, content vocab, idioms, verb forms). Skip for filler like 'sí', 'no', 'gracias' — those don't need tracking.

When the system passes you "## SRS — items due for review now", weave 1–2 of them naturally into THIS turn as a quick recap before continuing the lesson. Don't make it a separate quiz — just slip it in. ('Quick recap — how do you say "X" in Spanish?')"""


ONBOARDING_QUESTION = """Hi! I'm your Spanish tutor. Before we start, I'd like to know three things:

1. What should I call you?
2. Why are you learning Spanish?
3. Which languages do you already speak, and how well? (e.g. "Russian native, English B2")

You can answer in any language or mix them — no pressure. Just write whatever feels natural!"""


ONBOARDING_PARSE_SYSTEM = """You parse a new learner's onboarding reply into structured fields.

- name: what to call them (use their stated name; otherwise their Telegram first name verbatim)
- goal: why they want to learn Spanish, in one short phrase
- native_lang: ISO 639-1 code (e.g. "ru", "en", "es")
- known_languages: list of {lang, level}. lang is ISO 639-1. level is CEFR ("A1"–"C2") or "native". Skip the native language itself.
- self_reported_spanish_level: CEFR. If they don't mention Spanish, use "A0".

Be charitable: if the reply is short or vague, fill in reasonable defaults rather than failing."""


class KnownLanguage(BaseModel):
    lang: str
    level: str


class OnboardingResult(BaseModel):
    name: str
    goal: str
    native_lang: str
    known_languages: list[KnownLanguage]
    self_reported_spanish_level: str


class LearnerModelUpdate(BaseModel):
    assessed_level: Optional[str] = Field(
        None, description="CEFR level you assess for the learner, only if you have new evidence this turn"
    )
    new_vocab: list[str] = Field(
        default_factory=list, description="Spanish words/phrases the learner just encountered or successfully used"
    )
    new_weak_grammar: list[str] = Field(
        default_factory=list, description="Grammar areas the learner is struggling with (short tags like 'gender-agreement')"
    )
    new_errors: list[str] = Field(
        default_factory=list, description="Specific mistakes observed this turn (short descriptions)"
    )
    current_topic: Optional[str] = Field(None, description="Short label for what you're currently teaching")


class SrsItem(BaseModel):
    es: str = Field(description="Canonical Spanish word or phrase")
    en: str = Field(description="Short English gloss (used as the learner-facing meaning)")


class Evaluation(BaseModel):
    attempted: str = Field(description="What the learner tried to say (Spanish verbatim)")
    correct: bool
    what_was_wrong: Optional[str] = Field(None, description="One short sentence on the mistake, if any")
    corrected: Optional[str] = Field(None, description="The corrected version if wrong")
    tag: str = Field(description="Short topic tag, e.g. 'ser-vs-estar', 'tener-idioms', 'preterite-irregular'")
    test_item_idx: Optional[int] = Field(None, description="Set ONLY when grading a test item; the idx of the item being graded")
    srs_item: Optional[SrsItem] = Field(
        None,
        description="Set this when the evaluation should be tracked in the SRS queue — i.e. it's about a substantial vocab/phrase she should remember long-term. Skip for trivia like 'sí', 'no', 'gracias'.",
    )


class TestItem(BaseModel):
    idx: int
    type: Literal["translate_to_en", "translate_to_es", "fill_blank", "short_response", "multiple_choice"]
    prompt: str = Field(description="What you show the learner")
    correct_answer: str = Field(description="The canonical correct answer")
    options: list[str] = Field(default_factory=list, description="3-4 options when type is multiple_choice; empty otherwise")
    tag: str = Field(description="Short topic tag for tracking weaknesses")


class TestPlan(BaseModel):
    items: list[TestItem]


class TutorTurn(BaseModel):
    reply: str
    learner_model_update: LearnerModelUpdate
    lesson_complete: bool = Field(
        False,
        description="True ONLY when the learner has just demonstrated mastery of the current lesson (multiple successful uses of vocab, comfortable back-and-forth). The system will mark the lesson completed and put you at a junction (waiting for direction). Your reply must start with '✅ Lesson N complete!' and offer her: start next lesson, practice, or ask questions.",
    )
    advance_now: bool = Field(
        False,
        description="True when the learner clearly wants to START or CONTINUE a lesson NOW (e.g. 'yes', 'let's go', 'start it', 'next lesson please'). When true, your reply must lead with '📖 Lesson N: <title>' and teach the first item. During test_state=pending, this STARTS THE TEST instead.",
    )
    evaluation: Optional[Evaluation] = Field(
        None,
        description="REQUIRED during test_state=active when the learner has just answered a test item — set test_item_idx and grade. Optional otherwise.",
    )
    request_retest: bool = Field(
        False,
        description="True ONLY during test_state=review_after_fail when the learner has demonstrated recovery OR clearly asks to retry the test.",
    )
    send_image: bool = Field(
        False,
        description="Set true when the current lesson has a grammar reference card AND introducing it visually would help right now (e.g., when teaching its grammar topic for the first time, or when she asks for a visual). The system will send the lesson's image alongside your reply. Don't spam — once per lesson is plenty.",
    )


def load_curriculum() -> dict[str, dict]:
    curriculum = {}
    for path in sorted(CURRICULUM_DIR.glob("*/*.json")):
        with path.open(encoding="utf-8") as f:
            lesson = json.load(f)
        curriculum[lesson["id"]] = lesson
    return curriculum


CURRICULUM = load_curriculum()
LESSON_ORDER = sorted(CURRICULUM.keys())
FIRST_LESSON = LESSON_ORDER[0] if LESSON_ORDER else None


def next_lesson_id(current: str) -> Optional[str]:
    try:
        idx = LESSON_ORDER.index(current)
    except ValueError:
        return None
    return LESSON_ORDER[idx + 1] if idx + 1 < len(LESSON_ORDER) else None


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                telegram_id INTEGER PRIMARY KEY,
                name TEXT,
                goal TEXT,
                native_lang TEXT,
                known_languages TEXT,
                self_reported_spanish_level TEXT,
                assessed_spanish_level TEXT,
                onboarded INTEGER NOT NULL DEFAULT 0,
                streak_days INTEGER NOT NULL DEFAULT 0,
                last_active_date TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS learner_state (
                user_id INTEGER PRIMARY KEY REFERENCES users(telegram_id) ON DELETE CASCADE,
                current_lesson TEXT,
                at_junction INTEGER NOT NULL DEFAULT 1,
                completed_lessons TEXT NOT NULL DEFAULT '[]',
                learner_model TEXT NOT NULL DEFAULT '{}',
                test_state TEXT NOT NULL DEFAULT 'none',
                test_items TEXT NOT NULL DEFAULT '[]',
                test_responses TEXT NOT NULL DEFAULT '[]',
                focus_mode TEXT NOT NULL DEFAULT 'auto',
                last_updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user_id ON conversation_turns(user_id, id);

            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
                lesson_id TEXT,
                attempted TEXT,
                correct INTEGER NOT NULL,
                tag TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id, created_at);

            CREATE TABLE IF NOT EXISTS srs_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
                es TEXT NOT NULL,
                en TEXT NOT NULL,
                box_level INTEGER NOT NULL DEFAULT 1,
                due_at TEXT NOT NULL,
                last_reviewed_at TEXT,
                times_correct INTEGER NOT NULL DEFAULT 0,
                times_wrong INTEGER NOT NULL DEFAULT 0,
                tag TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, es)
            );
            CREATE INDEX IF NOT EXISTS idx_srs_due ON srs_items(user_id, due_at);
            """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(learner_state)").fetchall()}
        if "mode" in cols:
            conn.execute("ALTER TABLE learner_state DROP COLUMN mode")
        if "at_junction" not in cols:
            conn.execute("ALTER TABLE learner_state ADD COLUMN at_junction INTEGER NOT NULL DEFAULT 1")
        if "completed_lessons" not in cols:
            conn.execute("ALTER TABLE learner_state ADD COLUMN completed_lessons TEXT NOT NULL DEFAULT '[]'")
        if "test_state" not in cols:
            conn.execute("ALTER TABLE learner_state ADD COLUMN test_state TEXT NOT NULL DEFAULT 'none'")
        if "test_items" not in cols:
            conn.execute("ALTER TABLE learner_state ADD COLUMN test_items TEXT NOT NULL DEFAULT '[]'")
        if "test_responses" not in cols:
            conn.execute("ALTER TABLE learner_state ADD COLUMN test_responses TEXT NOT NULL DEFAULT '[]'")
        if "focus_mode" not in cols:
            conn.execute("ALTER TABLE learner_state ADD COLUMN focus_mode TEXT NOT NULL DEFAULT 'auto'")
        user_cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "streak_days" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN streak_days INTEGER NOT NULL DEFAULT 0")
        if "last_active_date" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN last_active_date TEXT")


def get_user(tg_id: int) -> Optional[sqlite3.Row]:
    with db() as conn:
        return conn.execute("SELECT * FROM users WHERE telegram_id = ?", (tg_id,)).fetchone()


def ensure_user(tg_id: int, first_name: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (telegram_id, name) VALUES (?, ?)",
            (tg_id, first_name),
        )
        conn.execute(
            "INSERT OR IGNORE INTO learner_state (user_id) VALUES (?)",
            (tg_id,),
        )


def save_onboarding(tg_id: int, result: OnboardingResult) -> None:
    with db() as conn:
        conn.execute(
            """
            UPDATE users SET
                name = ?,
                goal = ?,
                native_lang = ?,
                known_languages = ?,
                self_reported_spanish_level = ?,
                assessed_spanish_level = ?,
                onboarded = 1,
                last_seen_at = datetime('now')
            WHERE telegram_id = ?
            """,
            (
                result.name,
                result.goal,
                result.native_lang,
                json.dumps([kl.model_dump() for kl in result.known_languages], ensure_ascii=False),
                result.self_reported_spanish_level,
                result.self_reported_spanish_level,
                tg_id,
            ),
        )
        conn.execute(
            "UPDATE learner_state SET current_lesson = ?, at_junction = 1, completed_lessons = '[]' WHERE user_id = ?",
            (FIRST_LESSON, tg_id),
        )


def get_learner_state(tg_id: int) -> tuple[Optional[str], bool, list[str], dict]:
    """Returns (current_lesson_id, at_junction, completed_lessons, learner_model)."""
    with db() as conn:
        row = conn.execute(
            "SELECT current_lesson, at_junction, completed_lessons, learner_model FROM learner_state WHERE user_id = ?",
            (tg_id,),
        ).fetchone()
        if not row:
            return None, True, [], {}
        return (
            row["current_lesson"],
            bool(row["at_junction"]),
            json.loads(row["completed_lessons"]),
            json.loads(row["learner_model"]),
        )


def get_focus_mode(tg_id: int) -> str:
    with db() as conn:
        row = conn.execute(
            "SELECT focus_mode FROM learner_state WHERE user_id = ?", (tg_id,)
        ).fetchone()
        return row["focus_mode"] if row else "auto"


def set_focus_mode(tg_id: int, mode: str) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET focus_mode = ?, last_updated_at = datetime('now') WHERE user_id = ?",
            (mode, tg_id),
        )


def set_current_lesson(tg_id: int, lesson_id: str) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET current_lesson = ?, last_updated_at = datetime('now') WHERE user_id = ?",
            (lesson_id, tg_id),
        )


def set_at_junction(tg_id: int, value: bool) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET at_junction = ?, last_updated_at = datetime('now') WHERE user_id = ?",
            (1 if value else 0, tg_id),
        )


def mark_lesson_completed(tg_id: int, lesson_id: str) -> None:
    _, _, completed, _ = get_learner_state(tg_id)
    if lesson_id in completed:
        return
    completed.append(lesson_id)
    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET completed_lessons = ?, last_updated_at = datetime('now') WHERE user_id = ?",
            (json.dumps(completed), tg_id),
        )


def save_turn(tg_id: int, role: str, content: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO conversation_turns (user_id, role, content) VALUES (?, ?, ?)",
            (tg_id, role, content),
        )


def get_recent_history(tg_id: int, limit: int = HISTORY_LIMIT) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversation_turns WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (tg_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def trim_history(tg_id: int, keep: int = HISTORY_LIMIT) -> None:
    with db() as conn:
        conn.execute(
            """
            DELETE FROM conversation_turns
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT id FROM conversation_turns
                  WHERE user_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (tg_id, tg_id, keep),
        )


def get_test_state(tg_id: int) -> tuple[str, list[dict], list[dict]]:
    with db() as conn:
        row = conn.execute(
            "SELECT test_state, test_items, test_responses FROM learner_state WHERE user_id = ?",
            (tg_id,),
        ).fetchone()
        if not row:
            return "none", [], []
        return row["test_state"], json.loads(row["test_items"]), json.loads(row["test_responses"])


def set_test_state(tg_id: int, state: TestState) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET test_state = ?, last_updated_at = datetime('now') WHERE user_id = ?",
            (state, tg_id),
        )


def save_test_items(tg_id: int, items: list[dict]) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET test_items = ?, test_responses = '[]' WHERE user_id = ?",
            (json.dumps(items, ensure_ascii=False), tg_id),
        )


def append_test_response(tg_id: int, response: dict) -> list[dict]:
    _, _, responses = get_test_state(tg_id)
    responses.append(response)
    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET test_responses = ? WHERE user_id = ?",
            (json.dumps(responses, ensure_ascii=False), tg_id),
        )
    return responses


def clear_test(tg_id: int) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET test_state = 'none', test_items = '[]', test_responses = '[]' WHERE user_id = ?",
            (tg_id,),
        )


def update_streak(tg_id: int) -> int:
    """Bump or reset the streak based on last_active_date. Returns the new streak count."""
    today = datetime.now(timezone.utc).date().isoformat()
    with db() as conn:
        row = conn.execute(
            "SELECT streak_days, last_active_date FROM users WHERE telegram_id = ?",
            (tg_id,),
        ).fetchone()
        if not row:
            return 0
        last = row["last_active_date"]
        if last == today:
            return row["streak_days"]  # already counted today
        if last:
            last_date = datetime.fromisoformat(last).date()
            today_date = datetime.fromisoformat(today).date()
            gap = (today_date - last_date).days
            new_streak = row["streak_days"] + 1 if gap == 1 else 1
        else:
            new_streak = 1
        conn.execute(
            "UPDATE users SET streak_days = ?, last_active_date = ? WHERE telegram_id = ?",
            (new_streak, today, tg_id),
        )
        return new_streak


def get_progress_stats(tg_id: int) -> dict:
    """Aggregate everything /progress wants in one helper, easier to test."""
    row = get_user(tg_id)
    if not row:
        return {}
    current_lesson_id, _, completed_lessons, _ = get_learner_state(tg_id)
    srs_stats = get_srs_stats(tg_id)
    by_level: dict[str, dict[str, int]] = {}
    for lid in LESSON_ORDER:
        lvl = lid.split("/")[0]
        by_level.setdefault(lvl, {"total": 0, "done": 0})
        by_level[lvl]["total"] += 1
        if lid in completed_lessons:
            by_level[lvl]["done"] += 1
    with db() as conn:
        attempts_total = conn.execute(
            "SELECT COUNT(*) c FROM attempts WHERE user_id = ?", (tg_id,)
        ).fetchone()["c"]
        attempts_correct = conn.execute(
            "SELECT COUNT(*) c FROM attempts WHERE user_id = ? AND correct = 1", (tg_id,)
        ).fetchone()["c"]
    return {
        "name": row["name"],
        "level": row["assessed_spanish_level"] or row["self_reported_spanish_level"],
        "current_lesson_id": current_lesson_id,
        "current_lesson_title": CURRICULUM.get(current_lesson_id, {}).get("title", "—"),
        "streak_days": row["streak_days"] or 0,
        "by_level": by_level,
        "attempts_total": attempts_total,
        "attempts_correct": attempts_correct,
        "srs": srs_stats,
    }


def log_attempt(tg_id: int, lesson_id: Optional[str], attempted: str, correct: bool, tag: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO attempts (user_id, lesson_id, attempted, correct, tag) VALUES (?, ?, ?, ?, ?)",
            (tg_id, lesson_id, attempted, 1 if correct else 0, tag),
        )


def upsert_srs_item(tg_id: int, es: str, en: str, tag: Optional[str], correct: bool) -> None:
    """Add or update an SRS item per Leitner rules."""
    now = datetime.now(timezone.utc)
    es_clean = es.strip()
    if not es_clean:
        return
    with db() as conn:
        row = conn.execute(
            "SELECT box_level, times_correct, times_wrong FROM srs_items WHERE user_id = ? AND es = ?",
            (tg_id, es_clean),
        ).fetchone()
        if row:
            if correct:
                new_box = min(row["box_level"] + 1, MAX_BOX)
                times_c = row["times_correct"] + 1
                times_w = row["times_wrong"]
            else:
                new_box = 1
                times_c = row["times_correct"]
                times_w = row["times_wrong"] + 1
            interval = LEITNER_INTERVALS_DAYS[new_box]
            due_at = (now + timedelta(days=interval)).isoformat()
            conn.execute(
                """UPDATE srs_items SET box_level = ?, due_at = ?, last_reviewed_at = ?,
                       times_correct = ?, times_wrong = ?, tag = COALESCE(?, tag)
                   WHERE user_id = ? AND es = ?""",
                (new_box, due_at, now.isoformat(), times_c, times_w, tag, tg_id, es_clean),
            )
        else:
            box = 2 if correct else 1
            interval = LEITNER_INTERVALS_DAYS[box]
            due_at = (now + timedelta(days=interval)).isoformat()
            conn.execute(
                """INSERT INTO srs_items (user_id, es, en, box_level, due_at, last_reviewed_at,
                       times_correct, times_wrong, tag)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tg_id, es_clean, en.strip(), box, due_at, now.isoformat(),
                    1 if correct else 0, 0 if correct else 1, tag,
                ),
            )


def get_due_srs_items(tg_id: int, limit: int = DUE_ITEMS_PER_TURN) -> list[dict]:
    """Items where due_at <= now, ordered by struggle then box level."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        rows = conn.execute(
            """SELECT es, en, box_level, tag, times_wrong, times_correct FROM srs_items
               WHERE user_id = ? AND due_at <= ?
               ORDER BY times_wrong DESC, box_level ASC, due_at ASC
               LIMIT ?""",
            (tg_id, now_iso, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_srs_stats(tg_id: int) -> dict:
    now_iso = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) c FROM srs_items WHERE user_id = ?", (tg_id,)
        ).fetchone()["c"]
        due = conn.execute(
            "SELECT COUNT(*) c FROM srs_items WHERE user_id = ? AND due_at <= ?",
            (tg_id, now_iso),
        ).fetchone()["c"]
        by_box = conn.execute(
            "SELECT box_level, COUNT(*) c FROM srs_items WHERE user_id = ? GROUP BY box_level",
            (tg_id,),
        ).fetchall()
    return {
        "total_items": total,
        "due_now": due,
        "by_box": {r["box_level"]: r["c"] for r in by_box},
    }


def is_last_lesson_of_level(lesson_id: Optional[str]) -> bool:
    if not lesson_id:
        return False
    level = lesson_id.split("/")[0]
    level_lessons = [lid for lid in LESSON_ORDER if lid.startswith(f"{level}/")]
    return bool(level_lessons) and lesson_id == level_lessons[-1]


TEST_GEN_SYSTEM = """You generate an end-of-level review test for a Spanish learner.

Output exactly 10 mixed items covering the level's vocabulary and grammar. Use a MIX of types — do not make them all the same:
- translate_to_en: a Spanish phrase to translate to English
- translate_to_es: an English phrase to translate to Spanish
- fill_blank: an incomplete sentence with one blank to fill
- short_response: an open question with a short factual answer (e.g. "How do you say 'I'm hungry' in Spanish?")
- multiple_choice: a question with 3-4 plausible options, exactly one correct

For each item:
- idx: 0 through 9
- type: one of the 5 above
- prompt: what the learner sees (keep instructions in English; the target language is Spanish)
- correct_answer: the canonical expected answer (a short phrase)
- options: list of 3-4 strings if type is multiple_choice, else empty
- tag: a short topic identifier from the level's grammar/vocab (e.g. "ser-vs-estar", "preterite-irregular", "gender-agreement")

Spread topics — cover several lessons of the level, not just one. Mix difficulty: a few easy, a few harder.
The learner's native language is Russian, English at B2. Prompts in English are fine; do not include Russian glosses in test items."""


def generate_test_for_level(level: str) -> list[dict]:
    """Call Claude to generate a 10-item review test for the given level."""
    lessons = [CURRICULUM[lid] for lid in LESSON_ORDER if lid.startswith(f"{level}/")]
    if not lessons:
        return []
    user_msg = (
        f"Generate a 10-item review test for level {level}. Here are the lessons:\n\n"
        + json.dumps(lessons, ensure_ascii=False, indent=2)
    )
    response = claude.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=TEST_GEN_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        output_format=TestPlan,
    )
    return [item.model_dump() for item in response.parsed_output.items]


def merge_learner_model(tg_id: int, update: LearnerModelUpdate) -> None:
    _, _, _, current = get_learner_state(tg_id)
    if update.assessed_level:
        current["assessed_level"] = update.assessed_level
        with db() as conn:
            conn.execute(
                "UPDATE users SET assessed_spanish_level = ? WHERE telegram_id = ?",
                (update.assessed_level, tg_id),
            )
    if update.current_topic:
        current["current_topic"] = update.current_topic

    def append_capped(key: str, new: list[str], cap: int) -> None:
        existing = current.get(key, [])
        for item in new:
            if item not in existing:
                existing.append(item)
        current[key] = existing[-cap:]

    append_capped("known_vocab", update.new_vocab, VOCAB_CAP)
    append_capped("weak_grammar", update.new_weak_grammar, WEAK_CAP)
    append_capped("recent_errors", update.new_errors, ERROR_CAP)

    with db() as conn:
        conn.execute(
            "UPDATE learner_state SET learner_model = ?, last_updated_at = datetime('now') WHERE user_id = ?",
            (json.dumps(current, ensure_ascii=False), tg_id),
        )
        conn.execute(
            "UPDATE users SET last_seen_at = datetime('now') WHERE telegram_id = ?",
            (tg_id,),
        )


FOCUS_MODE_INSTRUCTIONS = {
    "auto": "",
    "grammar": (
        "## Focus mode: GRAMMAR\n\n"
        "She has chosen to focus on grammar right now. Center this turn on the current lesson's grammar rules. "
        "Drill conjugations, agreement, sentence patterns. Less vocab, more rule-and-practice. "
        "If the lesson has a card image and she hasn't seen it yet this lesson, send it (set send_image=true)."
    ),
    "vocab": (
        "## Focus mode: VOCABULARY\n\n"
        "She has chosen to focus on vocabulary right now. Flashcard-style: present an item, ask her to recall or use it, "
        "grade with `evaluation` and `srs_item`. Prioritize due SRS items if present. "
        "Less grammar talk, more word/phrase recall."
    ),
    "reading": (
        "## Focus mode: READING\n\n"
        "She has chosen to focus on reading right now. Write a short (2–4 sentences) original Spanish paragraph using "
        "vocab from her current lesson and recent material. Stay at her CEFR level. Then ask her to translate it "
        "into English (or Russian, her choice). Grade the translation in your next turn with `evaluation`. "
        "Pick a fun, light topic — café, weather, a dog, anything she'd find amusing."
    ),
}


def build_tutor_system_prompt(
    user,
    current_lesson_id: Optional[str],
    at_junction: bool,
    completed_lessons: list[str],
    learner_model: dict,
    test_state: str = "none",
    test_items: Optional[list[dict]] = None,
    test_responses: Optional[list[dict]] = None,
    due_srs_items: Optional[list[dict]] = None,
    focus_mode: str = "auto",
) -> str:
    test_items = test_items or []
    test_responses = test_responses or []
    due_srs_items = due_srs_items or []
    profile = {
        "name": user["name"],
        "goal": user["goal"],
        "native_lang": user["native_lang"],
        "known_languages": json.loads(user["known_languages"] or "[]"),
        "self_reported_spanish_level": user["self_reported_spanish_level"],
        "assessed_spanish_level": user["assessed_spanish_level"],
    }

    sections = [
        BASE_SYSTEM_PROMPT,
        "## Learner profile (stable)\n" + json.dumps(profile, ensure_ascii=False, indent=2),
        "## Learner model (observed)\n" + json.dumps(learner_model, ensure_ascii=False, indent=2),
    ]

    if focus_mode != "auto" and FOCUS_MODE_INSTRUCTIONS.get(focus_mode):
        sections.append(FOCUS_MODE_INSTRUCTIONS[focus_mode])

    # Junction state — tells the bot what mode it's in.
    current_lesson = CURRICULUM.get(current_lesson_id) if current_lesson_id else None
    nxt_id = next_lesson_id(current_lesson_id) if current_lesson_id else None
    nxt = CURRICULUM.get(nxt_id) if nxt_id else None
    current_completed = current_lesson_id in completed_lessons
    current_level = current_lesson_id.split("/")[0] if current_lesson_id else None

    # Test state overrides junction behavior when active.
    if test_state == "pending":
        sections.append(
            f"## test_state: PENDING\n\n"
            f"She just completed the FINAL lesson of level {current_level}. A 10-item review test is ready to start. "
            "Your reply for this turn should celebrate finishing the level AND clearly offer her three options:\n"
            "1) start the review test now (10 mixed questions)\n"
            "2) practice anything from this level first\n"
            "3) ask any questions\n\n"
            "If she says 'go' / 'start' / 'I'm ready' etc., emit `advance_now: true` — the system will generate test items and switch to test_state=active.\n"
            "Do NOT yourself try to generate test questions. Code handles that on next turn."
        )
    elif test_state == "active":
        next_idx = len(test_responses)
        items_block = json.dumps(test_items, ensure_ascii=False, indent=2)
        responses_block = json.dumps(test_responses, ensure_ascii=False, indent=2) if test_responses else "(none yet)"
        sections.append(
            f"## test_state: ACTIVE — administering review test for level {current_level}\n\n"
            f"Total items: {len(test_items)}. Answered so far: {next_idx}.\n\n"
            "Procedure each turn:\n"
            f"1. If next_idx > 0 (she just answered item {next_idx - 1}): grade her last attempt. EMIT `evaluation`: "
            "{attempted: her verbatim Spanish, correct: bool, what_was_wrong: short reason if wrong, corrected: right answer, "
            f"tag: from the item's `tag` field, test_item_idx: {next_idx - 1 if next_idx > 0 else 0}}}. "
            "Be strict but fair on grading (minor typos OK, wrong meaning is not).\n"
            f"2. In your reply: '✓' or '✗ + corrected' for the previous, then PRESENT item[{next_idx}] cleanly. "
            "For multiple_choice, list options as 1) ... 2) ... 3) ...\n"
            "3. If next_idx is the last (i.e. you just graded the last item): write a one-line summary "
            "('You got X/10 — well done!' or 'X/10 — close. We'll review Y first before retrying.'). "
            "Code will then transition state automatically.\n\n"
            f"If next_idx == 0 (test just started, no answers yet): no evaluation needed. Just present item[0] with "
            "a friendly intro like 'Here we go! Item 1 of 10:'.\n\n"
            f"PASS = {int(PASS_THRESHOLD * 100)}%+ correct. You don't compute pass/fail; just grade each item honestly.\n\n"
            f"## test_items\n{items_block}\n\n"
            f"## test_responses (so far)\n{responses_block}"
        )
    elif test_state == "review_after_fail":
        sections.append(
            f"## test_state: REVIEW_AFTER_FAIL — she failed the {current_level} test\n\n"
            "Her weak topics from the failed test are in learner_model.weak_grammar (recently appended). "
            "Drill her on those specifically — use the level's topic list and lesson summaries to construct quick exercises. "
            "Be encouraging, not discouraging — she's close.\n\n"
            "When she demonstrates recovery (multiple successful uses of the weak topics) OR she explicitly asks "
            "to retake the test ('let me try again', 'I'm ready for the retest'), emit `request_retest: true`. "
            "The system will generate a fresh test.\n\n"
            "If she demands retry too early without engaging review, gently insist: "
            "'Let's nail [topic] first, then we'll retake — you'll do better.'"
        )

    # Standard junction note only when no test state is active.
    if test_state == "none":
        if at_junction:
            if not current_completed and current_lesson:
                junction_note = (
                    f"## at_junction = TRUE\n\n"
                    f"She is waiting for direction. The lesson she would start next is "
                    f"**{current_lesson['title']}** (id: {current_lesson_id}) — its content is below.\n"
                    f"Use the junction rules from your base prompt. Do not teach yet."
                )
            elif current_completed and nxt:
                junction_note = (
                    f"## at_junction = TRUE\n\n"
                    f"She just completed **{current_lesson['title']}**. The next lesson would be "
                    f"**{nxt['title']}** (id: {nxt_id}) — its content is below.\n"
                    f"Use the junction rules. Do not teach yet."
                )
            elif current_completed and not nxt:
                junction_note = (
                    "## at_junction = TRUE\n\n"
                    "She just completed the FINAL lesson. No next lesson available. "
                    "Congratulate her on finishing the level. She can still practice or ask questions."
                )
            else:
                junction_note = "## at_junction = TRUE\n\nNo lesson set. Welcome her and offer to start."
        else:
            junction_note = "## at_junction = FALSE\n\nActive teaching mode. Teach the current lesson below."
        sections.append(junction_note)

    # The lesson she's currently on / about to start.
    if current_lesson:
        label = "Current lesson (active)" if not at_junction else (
            "Lesson she just completed" if current_completed else "Lesson she's about to start"
        )
        sections.append(f"## {label}\n" + json.dumps(current_lesson, ensure_ascii=False, indent=2))

    # Next lesson — relevant when she might advance.
    if nxt:
        sections.append(
            "## Next lesson (use this content when she advances)\n"
            + json.dumps(nxt, ensure_ascii=False, indent=2)
        )

    # SRS — items due for review now. Bot weaves them in opportunistically.
    if due_srs_items and test_state != "active":
        srs_block = [
            "## SRS — items due for review now",
            "These are previously-seen vocab/phrases the spaced-repetition queue says she should review NOW. "
            "Weave 1–2 of them naturally into THIS turn before continuing the lesson — a quick recall prompt "
            "(e.g. \"Quick recap — how do you say 'I'm hungry'?\"). When she answers, emit `evaluation` "
            "with `srs_item: {es, en}` set so the system can promote (right) or reset (wrong) her in the Leitner queue.",
            "",
            json.dumps(due_srs_items, ensure_ascii=False, indent=2),
        ]
        sections.append("\n".join(srs_block))

    # Completed history — collapsed by level mastery.
    finished_levels, other_summaries = organize_completed(completed_lessons, current_lesson_id)

    if finished_levels:
        levels_block = [
            "## Levels mastered",
            "She owns these levels — no lesson-by-lesson detail needed. Her actual command lives in the learner_model (vocab she's used, errors observed, weak grammar). Trust she's seen the topic; drill from your own Spanish knowledge if she revisits."
        ]
        for lvl in finished_levels:
            topics = [CURRICULUM[lid]["title"] for lid in LESSON_ORDER if lid.startswith(f"{lvl}/")]
            levels_block.append(f"- **{lvl}** topics: " + "; ".join(topics))
        sections.append("\n".join(levels_block))

    if other_summaries:
        completed_block = [
            "## Completed in current level (for practice queries)",
            "Terse references for lessons she just covered. Drill from your own Spanish knowledge."
        ]
        for lid in other_summaries:
            lesson = CURRICULUM.get(lid)
            if lesson:
                completed_block.append(lesson_summary(lesson))
        sections.append("\n\n".join(completed_block))

    return "\n\n".join(sections)


def organize_completed(
    completed_lessons: list[str], current_lesson_id: Optional[str]
) -> tuple[list[str], list[str]]:
    """Split completed lessons into fully-mastered levels vs in-progress-level lessons."""
    completed_set = set(completed_lessons)
    levels: dict[str, list[str]] = {}
    for lid in LESSON_ORDER:
        lvl = lid.split("/")[0]
        levels.setdefault(lvl, []).append(lid)

    finished_levels: list[str] = []
    finished_set: set[str] = set()
    for lvl, lids in levels.items():
        if all(lid in completed_set for lid in lids):
            finished_levels.append(lvl)
            finished_set.update(lids)

    other_summaries = [
        lid for lid in completed_lessons
        if lid not in finished_set and lid != current_lesson_id
    ]
    return finished_levels, other_summaries


def lesson_summary(lesson: dict) -> str:
    """Compact summary of a lesson for in-current-level completed context."""
    parts = [f"### {lesson['id']} — {lesson['title']}"]
    objectives = lesson.get("objectives") or []
    if objectives:
        parts.append("- Objectives: " + "; ".join(objectives))
    vocab_es = [v["es"] for v in lesson.get("vocab", []) if "es" in v]
    if vocab_es:
        parts.append("- Vocab (Spanish only): " + ", ".join(vocab_es))
    grammar = lesson.get("grammar_notes") or []
    if grammar:
        parts.append("- Grammar tags: " + " | ".join(grammar))
    return "\n".join(parts)


async def tutor_turn(tg_id: int, user_text: str, save_user_turn: bool = True) -> tuple[str, Optional[Path]]:
    user = get_user(tg_id)
    current_lesson_id, at_junction, completed_lessons, learner_model = get_learner_state(tg_id)
    test_state, test_items, test_responses = get_test_state(tg_id)
    due_srs = get_due_srs_items(tg_id, limit=DUE_ITEMS_PER_TURN)
    focus_mode = get_focus_mode(tg_id)
    system_prompt = build_tutor_system_prompt(
        user, current_lesson_id, at_junction, completed_lessons, learner_model,
        test_state, test_items, test_responses, due_srs, focus_mode,
    )

    history = get_recent_history(tg_id)
    messages = history + [{"role": "user", "content": user_text}]

    turn: Optional[TutorTurn] = None
    for attempt in range(2):
        try:
            response = claude.messages.parse(
                model=MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                output_format=TutorTurn,
            )
            if response.stop_reason == "max_tokens":
                logger.warning("user %s: hit max_tokens cap — reply may be truncated", tg_id)
            turn = response.parsed_output
            break
        except Exception as e:
            logger.warning("tutor_turn parse attempt %d failed: %s", attempt + 1, e)

    if turn is None:
        fallback = "Sorry, my brain hiccupped for a second 🤖 Could you say that again?"
        if save_user_turn:
            save_turn(tg_id, "user", user_text)
        save_turn(tg_id, "assistant", fallback)
        trim_history(tg_id)
        return fallback, None

    merge_learner_model(tg_id, turn.learner_model_update)
    if save_user_turn:
        save_turn(tg_id, "user", user_text)
    save_turn(tg_id, "assistant", turn.reply)
    trim_history(tg_id)

    # --- Evaluation logging (every graded attempt feeds attempts + SRS) ---
    if turn.evaluation:
        ev = turn.evaluation
        log_attempt(tg_id, current_lesson_id, ev.attempted, ev.correct, ev.tag)
        if ev.srs_item:
            upsert_srs_item(
                tg_id,
                ev.srs_item.es,
                ev.srs_item.en,
                ev.tag,
                ev.correct,
            )

    # --- Test signal handling ---
    # Record a graded test response, and trigger pass/fail transitions when complete.
    if turn.evaluation and test_state == "active":
        responses_now = append_test_response(tg_id, turn.evaluation.model_dump())
        if len(responses_now) >= len(test_items):
            correct_count = sum(1 for r in responses_now if r.get("correct"))
            score = correct_count / max(1, len(responses_now))
            level = current_lesson_id.split("/")[0] if current_lesson_id else "?"
            if score >= PASS_THRESHOLD:
                logger.info("user %s PASSED %s test: %d/%d", tg_id, level, correct_count, len(responses_now))
                clear_test(tg_id)
                nxt = next_lesson_id(current_lesson_id) if current_lesson_id else None
                if nxt:
                    set_current_lesson(tg_id, nxt)
                set_at_junction(tg_id, True)
            else:
                logger.info("user %s FAILED %s test: %d/%d", tg_id, level, correct_count, len(responses_now))
                weak_tags = list({r.get("tag") for r in responses_now if not r.get("correct") and r.get("tag")})
                if weak_tags:
                    merge_learner_model(tg_id, LearnerModelUpdate(new_weak_grammar=weak_tags))
                set_test_state(tg_id, "review_after_fail")
                set_at_junction(tg_id, True)

    # Retest request after a failed test.
    if turn.request_retest and test_state == "review_after_fail" and current_lesson_id:
        level = current_lesson_id.split("/")[0]
        try:
            new_items = generate_test_for_level(level)
            save_test_items(tg_id, new_items)
            set_test_state(tg_id, "active")
            set_at_junction(tg_id, False)
            logger.info("user %s starting retest of %s (%d items)", tg_id, level, len(new_items))
        except Exception as e:
            logger.error("user %s retest generation failed: %s", tg_id, e)

    # --- Lesson completion ---
    if turn.lesson_complete and current_lesson_id and current_lesson_id not in completed_lessons:
        mark_lesson_completed(tg_id, current_lesson_id)
        set_at_junction(tg_id, True)
        if is_last_lesson_of_level(current_lesson_id):
            set_test_state(tg_id, "pending")
            logger.info("user %s: %s completed (last in level) -> test pending", tg_id, current_lesson_id)
        else:
            logger.info("user %s: lesson %s completed -> at_junction", tg_id, current_lesson_id)

    # --- Advance ---
    if turn.advance_now and current_lesson_id:
        # Re-read state (lesson_complete may have just changed it)
        _, _, completed_now, _ = get_learner_state(tg_id)
        test_state_now, _, _ = get_test_state(tg_id)

        if test_state_now == "pending":
            # Start the level test instead of advancing.
            level = current_lesson_id.split("/")[0]
            try:
                items = generate_test_for_level(level)
                if items:
                    save_test_items(tg_id, items)
                    set_test_state(tg_id, "active")
                    set_at_junction(tg_id, False)
                    logger.info("user %s: starting %s test (%d items)", tg_id, level, len(items))
            except Exception as e:
                logger.error("user %s test generation failed: %s", tg_id, e)
        elif test_state_now in ("active", "review_after_fail"):
            # Don't allow lesson advancement while a test is in flight.
            pass
        else:
            # Normal advance.
            if current_lesson_id in completed_now:
                nxt = next_lesson_id(current_lesson_id)
                if nxt:
                    set_current_lesson(tg_id, nxt)
                    set_at_junction(tg_id, False)
                    logger.info("user %s: advanced %s -> %s", tg_id, current_lesson_id, nxt)
                else:
                    logger.info("user %s: at final lesson, no advance possible", tg_id)
            else:
                set_at_junction(tg_id, False)
                logger.info("user %s: starting lesson %s", tg_id, current_lesson_id)

    # If the bot asked for the lesson's image, resolve its path.
    image_path: Optional[Path] = None
    if turn.send_image and current_lesson_id:
        lesson = CURRICULUM.get(current_lesson_id, {})
        image_slug = lesson.get("image")
        if image_slug:
            candidate = CURRICULUM_DIR / "images" / f"{image_slug}.png"
            if candidate.exists():
                image_path = candidate

    return turn.reply, image_path


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info("/start from %s (id=%s)", user.username, user.id)
    ensure_user(user.id, user.first_name or "")
    row = get_user(user.id)

    if row["onboarded"]:
        await update.message.reply_text(
            f"Welcome back, {row['name']}! Send me a message whenever you're ready to continue."
        )
        return

    await update.message.reply_text(ONBOARDING_QUESTION)


async def profile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    row = get_user(user.id)
    if not row or not row["onboarded"]:
        await update.message.reply_text("No profile yet — send /start to set one up.")
        return

    current_lesson_id, at_junction, completed_lessons, learner_model = get_learner_state(user.id)
    test_state, test_items, test_responses = get_test_state(user.id)
    srs_stats = get_srs_stats(user.id)
    focus_mode = get_focus_mode(user.id)
    profile = {
        "name": row["name"],
        "goal": row["goal"],
        "native_lang": row["native_lang"],
        "known_languages": json.loads(row["known_languages"] or "[]"),
        "self_reported_spanish_level": row["self_reported_spanish_level"],
        "assessed_spanish_level": row["assessed_spanish_level"],
        "current_lesson": current_lesson_id,
        "at_junction": at_junction,
        "completed_lessons": completed_lessons,
        "test_state": test_state,
        "test_progress": f"{len(test_responses)}/{len(test_items)}" if test_items else None,
        "focus_mode": focus_mode,
        "srs": srs_stats,
        "learner_model": learner_model,
    }
    await update.message.reply_text(json.dumps(profile, ensure_ascii=False, indent=2))


async def lesson_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    row = get_user(user.id)
    if not row or not row["onboarded"]:
        await update.message.reply_text("No lesson yet — send /start first.")
        return
    current_lesson_id, at_junction, completed_lessons, _ = get_learner_state(user.id)
    lesson = CURRICULUM.get(current_lesson_id) if current_lesson_id else None
    if not lesson:
        await update.message.reply_text("No current lesson.")
        return
    header = f"📖 {lesson['title']} ({lesson['id']})"
    if at_junction:
        if current_lesson_id in completed_lessons:
            header += " — ✅ completed (waiting for next step)"
        else:
            header += " — ⏸ awaiting kickoff"
    lines = [header, "", "Objectives:"]
    lines += [f"  • {o}" for o in lesson["objectives"]]
    lines += ["", "Vocab:"]
    for v in lesson["vocab"]:
        hint = f" — {v['hint']}" if v.get("hint") else ""
        lines.append(f"  • {v['es']} = {v['en']} / {v['ru']}{hint}")
    await update.message.reply_text("\n".join(lines))


HELP_TEXT = (
    "Hi! Here's what I can do:\n\n"
    "/start — start fresh or come back\n"
    "/menu — pick what to focus on (grammar / vocab / reading)\n"
    "/lesson — show the current lesson card\n"
    "/progress — your stats (level, streak, lessons done, SRS queue)\n"
    "/review — drill items due in your spaced-repetition queue\n"
    "/profile — full JSON state (for debugging)\n"
    "/help — this list\n\n"
    "Just send me any message and I'll teach you Spanish 💛"
)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


async def progress_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    row = get_user(user.id)
    if not row or not row["onboarded"]:
        await update.message.reply_text("No profile yet — send /start to set one up.")
        return
    stats = get_progress_stats(user.id)

    lines = [
        f"📊 Progress for {stats['name']}",
        "",
        f"🎯 Level: {stats['level']}",
        f"📖 Current lesson: {stats['current_lesson_title']}",
        f"🔥 Streak: {stats['streak_days']} day{'s' if stats['streak_days'] != 1 else ''}",
        "",
        "Lessons completed:",
    ]
    for lvl in ("A0", "A1", "A2", "B1"):
        if lvl in stats["by_level"]:
            d = stats["by_level"][lvl]
            bar = "█" * d["done"] + "·" * max(0, d["total"] - d["done"])
            lines.append(f"  {lvl}: {d['done']:>2}/{d['total']:>2}  {bar}")

    total = stats["attempts_total"]
    correct = stats["attempts_correct"]
    accuracy = f"{(correct / total * 100):.0f}%" if total else "—"
    lines += [
        "",
        f"💪 Attempts: {total} total ({correct} correct, {accuracy})",
        f"🧠 SRS items: {stats['srs']['total_items']}  (due now: {stats['srs']['due_now']})",
    ]
    if stats["srs"]["by_box"]:
        boxes = "  ".join(f"box{b}:{c}" for b, c in sorted(stats["srs"]["by_box"].items()))
        lines.append(f"   Boxes: {boxes}")

    await update.message.reply_text("\n".join(lines))


FOCUS_LABEL = {
    "auto": "✨ Auto",
    "grammar": "📐 Grammar",
    "vocab": "📚 Vocabulary",
    "reading": "📖 Reading",
}


def focus_menu_markup(current: str) -> InlineKeyboardMarkup:
    """Build the inline keyboard. Current mode gets a ✓ next to it."""
    def label(mode: str) -> str:
        return ("✓ " if mode == current else "") + FOCUS_LABEL[mode]

    rows = [
        [InlineKeyboardButton(label("auto"), callback_data="focus:auto")],
        [
            InlineKeyboardButton(label("grammar"), callback_data="focus:grammar"),
            InlineKeyboardButton(label("vocab"), callback_data="focus:vocab"),
        ],
        [InlineKeyboardButton(label("reading"), callback_data="focus:reading")],
    ]
    return InlineKeyboardMarkup(rows)


async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    row = get_user(user.id)
    if not row or not row["onboarded"]:
        await update.message.reply_text("Send /start first.")
        return
    current = get_focus_mode(user.id)
    await update.message.reply_text(
        "What do you want to focus on today?\n\nTap a button — your choice sticks until you change it.",
        reply_markup=focus_menu_markup(current),
    )


async def focus_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    row = get_user(user.id)
    if not row or not row["onboarded"]:
        await query.edit_message_text("Send /start first.")
        return

    data = query.data or ""
    if not data.startswith("focus:"):
        return
    choice = data.split(":", 1)[1]

    if choice == "disabled":
        await query.answer("Coming soon — audio support needs an extra service. We'll add it later!", show_alert=True)
        return

    if choice not in FOCUS_LABEL:
        return

    set_focus_mode(user.id, choice)
    logger.info("user %s set focus_mode=%s", user.id, choice)
    await query.edit_message_text(
        f"Focus set to {FOCUS_LABEL[choice]}.\n\nSend any message and I'll dive in.",
        reply_markup=focus_menu_markup(choice),
    )


async def review_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    row = get_user(user.id)
    if not row or not row["onboarded"]:
        await update.message.reply_text("Send /start first.")
        return
    due = get_due_srs_items(user.id, limit=20)
    if not due:
        await update.message.reply_text("Nothing due for review right now! Your SRS queue is clear ✨")
        return
    await update.message.chat.send_action("typing")
    reply, image_path = await tutor_turn(
        user.id,
        f"[/review session: there are {len(due)} items due now. Start a focused review by drilling these items one at a time. After the session, the learner can return to her current lesson.]",
        save_user_turn=False,
    )
    await send_tutor_reply(update, reply, image_path)


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    text = update.message.text
    logger.info("chat from %s: %r", user.username, text[:80])

    ensure_user(user.id, user.first_name or "")
    row = get_user(user.id)

    if row["onboarded"]:
        update_streak(user.id)

    if not row["onboarded"]:
        await update.message.chat.send_action("typing")
        parsed = claude.messages.parse(
            model=MODEL,
            max_tokens=1024,
            system=ONBOARDING_PARSE_SYSTEM,
            messages=[{"role": "user", "content": text}],
            output_format=OnboardingResult,
        )
        save_onboarding(user.id, parsed.parsed_output)
        logger.info("onboarded user %s", user.id)

        first_reply, image_path = await tutor_turn(
            user.id,
            "[system: this is the learner's first interaction. She just finished onboarding and is at_junction. Welcome her warmly by name in one line, acknowledge her goal in one line, preview Lesson 1: Greetings in one line, and ask whether she wants to start now or has any questions first. Do NOT teach lesson content yet.]",
            save_user_turn=False,
        )
        await send_tutor_reply(update, first_reply, image_path)
        return

    await update.message.chat.send_action("typing")
    reply, image_path = await tutor_turn(user.id, text)
    await send_tutor_reply(update, reply, image_path)


async def send_tutor_reply(update: Update, reply: str, image_path: Optional[Path] = None) -> None:
    """Send a tutor reply (optionally with a photo) using HTML parsing; fall back to plain on parse error."""
    if image_path and image_path.exists():
        try:
            # Telegram caption limit is 1024 chars. Longer reply → send photo first, then text.
            if len(reply) <= 1024:
                await update.message.reply_photo(photo=image_path.open("rb"), caption=reply, parse_mode="HTML")
                return
            await update.message.reply_photo(photo=image_path.open("rb"))
        except BadRequest as e:
            logger.warning("photo send failed (%s); falling back to text only", e)
    try:
        await update.message.reply_text(reply, parse_mode="HTML")
    except BadRequest as e:
        logger.warning("HTML send failed (%s); retrying as plain text", e)
        await update.message.reply_text(reply)


def main() -> None:
    init_db()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("profile", profile_cmd))
    app.add_handler(CommandHandler("lesson", lesson_cmd))
    app.add_handler(CommandHandler("progress", progress_cmd))
    app.add_handler(CommandHandler("review", review_cmd))
    app.add_handler(CommandHandler("menu", menu_cmd))
    app.add_handler(CallbackQueryHandler(focus_callback, pattern=r"^focus:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    logger.info(
        "bot starting (polling) — model=%s db=%s curriculum=%d lessons",
        MODEL,
        DB_PATH,
        len(CURRICULUM),
    )
    app.run_polling()


if __name__ == "__main__":
    main()