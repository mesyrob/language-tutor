"""Shared pytest fixtures: fresh DB per test, mocked Claude client."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make project root importable so tests can `import main`.
sys.path.insert(0, str(Path(__file__).parent.parent))

import main  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh empty DB at a temp path, patched into main.DB_PATH."""
    db_file = tmp_path / "test_tutor.db"
    monkeypatch.setattr(main, "DB_PATH", str(db_file))
    main.init_db()
    return str(db_file)


@pytest.fixture
def mock_claude(monkeypatch):
    """Replace main.claude with a MagicMock so no real API calls happen."""
    mock = MagicMock()
    monkeypatch.setattr(main, "claude", mock)
    return mock


@pytest.fixture
def onboarded_user(tmp_db):
    """An onboarded user with a sensible profile, current_lesson = first lesson, at_junction=True."""
    tg_id = 12345
    main.ensure_user(tg_id, "TestUser")
    main.save_onboarding(
        tg_id,
        main.OnboardingResult(
            name="Lena",
            goal="talk to family",
            native_lang="ru",
            known_languages=[main.KnownLanguage(lang="en", level="B2")],
            self_reported_spanish_level="A0",
        ),
    )
    return tg_id


