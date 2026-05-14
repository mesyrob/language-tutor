"""Schema, migrations, and basic CRUD."""
import json
import sqlite3

import main


def _tables(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def _columns(db_path, table):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    conn.close()
    return {r[1] for r in rows}


def test_init_db_creates_all_tables(tmp_db):
    tables = _tables(tmp_db)
    assert {"users", "learner_state", "conversation_turns", "attempts", "srs_items"} <= tables


def test_learner_state_has_phase5_6_columns(tmp_db):
    cols = _columns(tmp_db, "learner_state")
    assert {"at_junction", "completed_lessons", "test_state", "test_items", "test_responses"} <= cols


def test_srs_items_has_unique_constraint(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)
    conn = sqlite3.connect(tmp_db)
    count = conn.execute(
        "SELECT COUNT(*) FROM srs_items WHERE es = 'hola'"
    ).fetchone()[0]
    conn.close()
    assert count == 1  # updated, not duplicated


def test_ensure_user_creates_row(tmp_db):
    main.ensure_user(99, "Test")
    row = main.get_user(99)
    assert row is not None
    assert row["name"] == "Test"
    assert row["onboarded"] == 0


def test_ensure_user_is_idempotent(tmp_db):
    main.ensure_user(99, "Test")
    main.ensure_user(99, "DifferentName")
    row = main.get_user(99)
    assert row["name"] == "Test"  # INSERT OR IGNORE preserved the original


def test_get_user_missing_returns_none(tmp_db):
    assert main.get_user(99999) is None


def test_save_onboarding_sets_all_fields_and_lesson(tmp_db):
    main.ensure_user(42, "")
    main.save_onboarding(
        42,
        main.OnboardingResult(
            name="Ana",
            goal="travel",
            native_lang="ru",
            known_languages=[main.KnownLanguage(lang="en", level="B2")],
            self_reported_spanish_level="A0",
        ),
    )
    row = main.get_user(42)
    assert row["onboarded"] == 1
    assert row["name"] == "Ana"
    assert row["self_reported_spanish_level"] == "A0"
    current_lesson, at_junction, completed, _ = main.get_learner_state(42)
    assert current_lesson == main.FIRST_LESSON
    assert at_junction is True
    assert completed == []


def test_mark_lesson_completed_appends(tmp_db, onboarded_user):
    main.mark_lesson_completed(onboarded_user, "A0/01-greetings")
    _, _, completed, _ = main.get_learner_state(onboarded_user)
    assert completed == ["A0/01-greetings"]


def test_mark_lesson_completed_is_idempotent(tmp_db, onboarded_user):
    main.mark_lesson_completed(onboarded_user, "A0/01-greetings")
    main.mark_lesson_completed(onboarded_user, "A0/01-greetings")
    _, _, completed, _ = main.get_learner_state(onboarded_user)
    assert completed == ["A0/01-greetings"]


def test_history_save_and_retrieve(tmp_db, onboarded_user):
    main.save_turn(onboarded_user, "user", "hola")
    main.save_turn(onboarded_user, "assistant", "¡Hola!")
    history = main.get_recent_history(onboarded_user)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hola"}
    assert history[1] == {"role": "assistant", "content": "¡Hola!"}


def test_history_trim_keeps_last_n(tmp_db, onboarded_user):
    for i in range(20):
        main.save_turn(onboarded_user, "user", f"msg {i}")
    main.trim_history(onboarded_user)
    history = main.get_recent_history(onboarded_user, limit=999)
    assert len(history) == main.HISTORY_LIMIT
    # Should keep the most RECENT messages
    assert history[-1]["content"] == "msg 19"


def test_test_state_default_is_none(tmp_db, onboarded_user):
    state, items, responses = main.get_test_state(onboarded_user)
    assert state == "none"
    assert items == []
    assert responses == []


def test_set_test_state_transitions(tmp_db, onboarded_user):
    main.set_test_state(onboarded_user, "pending")
    state, _, _ = main.get_test_state(onboarded_user)
    assert state == "pending"


def test_save_test_items_resets_responses(tmp_db, onboarded_user):
    main.save_test_items(onboarded_user, [{"idx": 0, "prompt": "x"}])
    main.append_test_response(onboarded_user, {"item_idx": 0, "correct": True})
    main.save_test_items(onboarded_user, [{"idx": 0, "prompt": "y"}])
    _, items, responses = main.get_test_state(onboarded_user)
    assert len(items) == 1
    assert responses == []  # reset


def test_clear_test_resets_all(tmp_db, onboarded_user):
    main.set_test_state(onboarded_user, "active")
    main.save_test_items(onboarded_user, [{"idx": 0}])
    main.append_test_response(onboarded_user, {"item_idx": 0})
    main.clear_test(onboarded_user)
    state, items, responses = main.get_test_state(onboarded_user)
    assert state == "none"
    assert items == []
    assert responses == []


def test_unicode_round_trips(tmp_db, onboarded_user):
    """Russian/Spanish chars must survive DB write+read."""
    main.upsert_srs_item(onboarded_user, "buenos días", "good morning / доброе утро", "g", True)
    due = main.get_due_srs_items(onboarded_user, limit=10)
    # Manually force due
    conn = sqlite3.connect(tmp_db)
    from datetime import datetime, timedelta, timezone
    past = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    conn.execute("UPDATE srs_items SET due_at = ?", (past,))
    conn.commit()
    conn.close()
    due = main.get_due_srs_items(onboarded_user, limit=10)
    assert due[0]["es"] == "buenos días"
    assert "доброе утро" in due[0]["en"]
