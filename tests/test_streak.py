"""Streak tracking edge cases."""
import sqlite3
from datetime import datetime, timedelta, timezone

import main


def _set_last_active(db_path: str, tg_id: int, date_iso: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE users SET last_active_date = ? WHERE telegram_id = ?", (date_iso, tg_id)
    )
    conn.commit()
    conn.close()


def _get_streak(db_path: str, tg_id: int) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT streak_days FROM users WHERE telegram_id = ?", (tg_id,)
    ).fetchone()
    conn.close()
    return row["streak_days"]


def test_first_ever_activity_starts_streak_at_1(tmp_db, onboarded_user):
    streak = main.update_streak(onboarded_user)
    assert streak == 1
    assert _get_streak(tmp_db, onboarded_user) == 1


def test_same_day_activity_does_not_re_bump(tmp_db, onboarded_user):
    main.update_streak(onboarded_user)
    main.update_streak(onboarded_user)
    main.update_streak(onboarded_user)
    assert _get_streak(tmp_db, onboarded_user) == 1


def test_consecutive_day_increments_streak(tmp_db, onboarded_user):
    main.update_streak(onboarded_user)  # day 1
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    _set_last_active(tmp_db, onboarded_user, yesterday)
    main.update_streak(onboarded_user)  # day 2
    assert _get_streak(tmp_db, onboarded_user) == 2


def test_gap_resets_streak(tmp_db, onboarded_user):
    main.update_streak(onboarded_user)
    # Pretend last active was 5 days ago
    five_days_ago = (datetime.now(timezone.utc).date() - timedelta(days=5)).isoformat()
    _set_last_active(tmp_db, onboarded_user, five_days_ago)
    # Also crank the count up so we can see it reset
    conn = sqlite3.connect(tmp_db)
    conn.execute("UPDATE users SET streak_days = 10 WHERE telegram_id = ?", (onboarded_user,))
    conn.commit()
    conn.close()

    main.update_streak(onboarded_user)
    assert _get_streak(tmp_db, onboarded_user) == 1


def test_progress_stats_basic_shape(tmp_db, onboarded_user):
    main.upsert_srs_item(onboarded_user, "hola", "hello", "g", correct=True)
    main.mark_lesson_completed(onboarded_user, "A0/01-greetings")
    stats = main.get_progress_stats(onboarded_user)
    assert stats["name"] == "Lena"
    assert stats["by_level"]["A0"] == {"total": 8, "done": 1}
    assert stats["by_level"]["B1"]["total"] == 30
    assert stats["srs"]["total_items"] == 1
    assert stats["current_lesson_title"]
