"""SRS Leitner mechanics."""
import sqlite3
from datetime import datetime, timedelta, timezone

import main


def _force_due_now(db_path: str) -> None:
    """Pretend all SRS items are due (move due_at into the past)."""
    past = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE srs_items SET due_at = ?", (past,))
    conn.commit()
    conn.close()


def _box_of(db_path: str, es: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT box_level FROM srs_items WHERE es = ?", (es,)).fetchone()
    conn.close()
    return row["box_level"]


def test_new_item_correct_starts_at_box_2(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "hola", "hello", "greeting", correct=True)
    assert _box_of(tmp_db, "hola") == 2


def test_new_item_wrong_starts_at_box_1(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=False)
    assert _box_of(tmp_db, "hola") == 1


def test_correct_promotes_one_box_at_a_time(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)  # box 2
    assert _box_of(tmp_db, "hola") == 2
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)
    assert _box_of(tmp_db, "hola") == 3
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)
    assert _box_of(tmp_db, "hola") == 4


def test_correct_caps_at_max_box(tmp_db, onboarded_user):
    for _ in range(20):
        main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)
    assert _box_of(tmp_db, "hola") == main.MAX_BOX


def test_wrong_resets_to_box_1(tmp_db, onboarded_user):
    for _ in range(3):
        main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=False)
    assert _box_of(tmp_db, "hola") == 1


def test_times_correct_and_wrong_accumulate(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "x", "x", "g", correct=True)
    main.upsert_srs_item(onboarded_user, "x", "x", "g", correct=True)
    main.upsert_srs_item(onboarded_user, "x", "x", "g", correct=False)
    main.upsert_srs_item(onboarded_user, "x", "x", "g", correct=True)
    conn = sqlite3.connect(tmp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT times_correct, times_wrong FROM srs_items WHERE es = ?", ("x",)).fetchone()
    conn.close()
    assert row["times_correct"] == 3
    assert row["times_wrong"] == 1


def test_due_at_schedules_per_box(tmp_db, onboarded_user):
    """Newly-correct item should be due ~2 days out (box 2 = 2-day interval)."""
    main.upsert_srs_item(onboarded_user, "x", "x", "g", correct=True)
    conn = sqlite3.connect(tmp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT due_at FROM srs_items WHERE es = ?", ("x",)).fetchone()
    conn.close()
    due = datetime.fromisoformat(row["due_at"])
    diff = due - datetime.now(timezone.utc)
    assert 1.5 <= diff.total_seconds() / 86400 <= 2.5


def test_get_due_returns_nothing_when_all_future(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)
    due = main.get_due_srs_items(onboarded_user)
    assert due == []  # scheduled for tomorrow, nothing due now


def test_get_due_returns_overdue_items(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=False)
    _force_due_now(tmp_db)
    due = main.get_due_srs_items(onboarded_user)
    assert len(due) == 1
    assert due[0]["es"] == "hola"


def test_get_due_ordered_by_times_wrong_desc(tmp_db, onboarded_user):
    """The item she's gotten wrong more should come first."""
    main.upsert_srs_item(onboarded_user, "easy", "easy", "g", correct=True)
    main.upsert_srs_item(onboarded_user, "hard", "hard", "g", correct=False)
    main.upsert_srs_item(onboarded_user, "hard", "hard", "g", correct=False)
    _force_due_now(tmp_db)
    due = main.get_due_srs_items(onboarded_user, limit=10)
    assert due[0]["es"] == "hard"


def test_get_due_respects_limit(tmp_db, onboarded_user):
    for i in range(5):
        main.upsert_srs_item(onboarded_user, f"word{i}", f"word{i}", "g", correct=False)
    _force_due_now(tmp_db)
    due = main.get_due_srs_items(onboarded_user, limit=3)
    assert len(due) == 3


def test_empty_es_ignored(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "   ", "x", "g", correct=True)
    conn = sqlite3.connect(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM srs_items").fetchone()[0]
    conn.close()
    assert count == 0


def test_srs_stats(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "a", "a", "g", correct=True)   # box 2
    main.upsert_srs_item(onboarded_user, "b", "b", "g", correct=False)  # box 1
    main.upsert_srs_item(onboarded_user, "c", "c", "g", correct=False)  # box 1
    _force_due_now(tmp_db)
    stats = main.get_srs_stats(onboarded_user)
    assert stats["total_items"] == 3
    assert stats["due_now"] == 3
    assert stats["by_box"][1] == 2
    assert stats["by_box"][2] == 1


def test_isolation_between_users(tmp_db):
    main.ensure_user(1, "u1")
    main.ensure_user(2, "u2")
    main.upsert_srs_item(1, "hola", "hello", "g", correct=True)
    main.upsert_srs_item(2, "adios", "bye", "g", correct=True)
    u1_stats = main.get_srs_stats(1)
    u2_stats = main.get_srs_stats(2)
    assert u1_stats["total_items"] == 1
    assert u2_stats["total_items"] == 1
