"""End-to-end tutor_turn flow with Claude API mocked.

Each test sets up a specific learner state, mocks claude.messages.parse to
return a TutorTurn with specific signals, runs tutor_turn(), then asserts the
resulting DB state.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import main
from tests.helpers import make_test_plan, make_tutor_response


# ---------- Lesson advancement ----------


@pytest.mark.asyncio
async def test_lesson_complete_marks_done_and_sets_junction(tmp_db, onboarded_user, mock_claude):
    main.set_at_junction(onboarded_user, False)
    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="✅ Lesson complete!",
        lesson_complete=True,
    )

    await main.tutor_turn(onboarded_user, "I think I have it")

    _, at_junction, completed, _ = main.get_learner_state(onboarded_user)
    assert "A0/01-greetings" in completed
    assert at_junction is True


@pytest.mark.asyncio
async def test_advance_now_from_junction_starts_lesson(tmp_db, onboarded_user, mock_claude):
    # Default state: at_junction=True, current=A0/01, not completed
    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="📖 Lesson 1: Greetings",
        advance_now=True,
    )

    await main.tutor_turn(onboarded_user, "let's go")

    current_lesson, at_junction, completed, _ = main.get_learner_state(onboarded_user)
    assert current_lesson == "A0/01-greetings"  # lesson didn't advance — she's starting it
    assert at_junction is False
    assert completed == []


@pytest.mark.asyncio
async def test_advance_now_after_lesson_complete_advances(tmp_db, onboarded_user, mock_claude):
    """Both signals in one turn: complete current, immediately start next."""
    main.set_at_junction(onboarded_user, False)
    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="✅ Done! 📖 Lesson 2: Introductions",
        lesson_complete=True,
        advance_now=True,
    )

    await main.tutor_turn(onboarded_user, "next")

    current_lesson, at_junction, completed, _ = main.get_learner_state(onboarded_user)
    assert current_lesson == "A0/02-introductions"
    assert at_junction is False
    assert "A0/01-greetings" in completed


# ---------- End-of-level test flow ----------


@pytest.mark.asyncio
async def test_lesson_complete_on_last_in_level_sets_test_pending(tmp_db, onboarded_user, mock_claude):
    main.set_current_lesson(onboarded_user, "A0/08-wants-essentials")
    main.set_at_junction(onboarded_user, False)
    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="🎉 You've finished A0!",
        lesson_complete=True,
    )

    await main.tutor_turn(onboarded_user, "got it")

    state, _, _ = main.get_test_state(onboarded_user)
    assert state == "pending"


@pytest.mark.asyncio
async def test_advance_now_during_test_pending_starts_test(tmp_db, onboarded_user, mock_claude):
    """advance_now while in test_pending should generate items and switch to active."""
    main.set_current_lesson(onboarded_user, "A0/08-wants-essentials")
    main.set_test_state(onboarded_user, "pending")
    main.mark_lesson_completed(onboarded_user, "A0/08-wants-essentials")

    fake_items = [
        main.TestItem(idx=i, type="translate_to_es", prompt=f"Q{i}", correct_answer="A", options=[], tag="t")
        for i in range(main.TEST_ITEM_COUNT)
    ]
    # tutor_turn calls parse() once for TutorTurn; then generate_test_for_level calls parse() again for TestPlan.
    mock_claude.messages.parse.side_effect = [
        make_tutor_response(reply="Starting test!", advance_now=True),
        make_test_plan(fake_items),
    ]

    await main.tutor_turn(onboarded_user, "go")

    state, items, responses = main.get_test_state(onboarded_user)
    assert state == "active"
    assert len(items) == main.TEST_ITEM_COUNT
    assert responses == []


@pytest.mark.asyncio
async def test_evaluation_during_test_appends_response(tmp_db, onboarded_user, mock_claude):
    """Mid-test: grading one item appends to test_responses without clearing the test."""
    main.set_current_lesson(onboarded_user, "A0/08-wants-essentials")
    main.set_test_state(onboarded_user, "active")
    # 3 items so the test isn't complete after grading 1.
    main.save_test_items(onboarded_user, [
        {"idx": i, "type": "translate_to_es", "prompt": "Q", "correct_answer": "A", "options": [], "tag": "t"}
        for i in range(3)
    ])

    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="✓ next item",
        evaluation=main.Evaluation(attempted="hola", correct=True, tag="greeting", test_item_idx=0),
    )

    await main.tutor_turn(onboarded_user, "hola")

    state, items, responses = main.get_test_state(onboarded_user)
    assert state == "active"
    assert len(items) == 3
    assert len(responses) == 1
    assert responses[0]["correct"] is True


@pytest.mark.asyncio
async def test_test_passes_advances_to_next_level(tmp_db, onboarded_user, mock_claude):
    main.set_current_lesson(onboarded_user, "A0/08-wants-essentials")
    main.set_test_state(onboarded_user, "active")
    # 10 items, 9 already correct in DB; the bot grades the 10th correct → 10/10 pass.
    items = [{"idx": i, "type": "translate_to_es", "prompt": "x", "correct_answer": "y", "options": [], "tag": "t"} for i in range(10)]
    main.save_test_items(onboarded_user, items)
    for i in range(9):
        main.append_test_response(onboarded_user, {"item_idx": i, "correct": True, "tag": "t"})

    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="🎉 10/10!",
        evaluation=main.Evaluation(attempted="y", correct=True, tag="t", test_item_idx=9),
    )

    await main.tutor_turn(onboarded_user, "last answer")

    state, items_after, _ = main.get_test_state(onboarded_user)
    current_lesson, at_junction, _, _ = main.get_learner_state(onboarded_user)
    assert state == "none"
    assert items_after == []
    assert current_lesson == "A1/01-numbers-21-100"
    assert at_junction is True


@pytest.mark.asyncio
async def test_test_fails_goes_to_review_after_fail(tmp_db, onboarded_user, mock_claude):
    main.set_current_lesson(onboarded_user, "A0/08-wants-essentials")
    main.set_test_state(onboarded_user, "active")
    items = [{"idx": i, "type": "translate_to_es", "prompt": "x", "correct_answer": "y", "options": [], "tag": "tag" + str(i)} for i in range(10)]
    main.save_test_items(onboarded_user, items)
    # 9 already wrong
    for i in range(9):
        main.append_test_response(onboarded_user, {"item_idx": i, "correct": False, "tag": f"tag{i}"})

    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="3/10 — let's review",
        evaluation=main.Evaluation(attempted="x", correct=False, tag="tag9", test_item_idx=9),
    )

    await main.tutor_turn(onboarded_user, "wrong answer")

    state, _, _ = main.get_test_state(onboarded_user)
    current_lesson, _, _, learner_model = main.get_learner_state(onboarded_user)
    assert state == "review_after_fail"
    assert current_lesson == "A0/08-wants-essentials"  # NOT advanced to A1
    # Failed tags should have been appended to weak_grammar (capped at WEAK_CAP).
    weak = learner_model.get("weak_grammar", [])
    assert len(weak) > 0
    assert all(t.startswith("tag") for t in weak)


@pytest.mark.asyncio
async def test_request_retest_generates_fresh_test(tmp_db, onboarded_user, mock_claude):
    main.set_current_lesson(onboarded_user, "A0/08-wants-essentials")
    main.set_test_state(onboarded_user, "review_after_fail")

    fake_items = [main.TestItem(idx=i, type="translate_to_es", prompt=f"Q{i}", correct_answer="A", options=[], tag="t") for i in range(10)]
    mock_claude.messages.parse.side_effect = [
        make_tutor_response(reply="Restarting!", request_retest=True),
        make_test_plan(fake_items),
    ]

    await main.tutor_turn(onboarded_user, "let me try again")

    state, items, responses = main.get_test_state(onboarded_user)
    assert state == "active"
    assert len(items) == 10
    assert responses == []


@pytest.mark.asyncio
async def test_advance_now_during_active_test_is_ignored(tmp_db, onboarded_user, mock_claude):
    """She can't escape a test by saying 'next lesson'."""
    main.set_current_lesson(onboarded_user, "A0/08-wants-essentials")
    main.set_test_state(onboarded_user, "active")
    main.save_test_items(onboarded_user, [{"idx": 0}])

    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="Stay focused!",
        advance_now=True,
    )

    await main.tutor_turn(onboarded_user, "next please")

    current_lesson, _, _, _ = main.get_learner_state(onboarded_user)
    state, _, _ = main.get_test_state(onboarded_user)
    assert current_lesson == "A0/08-wants-essentials"  # didn't advance
    assert state == "active"  # still in test


# ---------- SRS through tutor_turn ----------


@pytest.mark.asyncio
async def test_evaluation_with_srs_item_writes_to_queue(tmp_db, onboarded_user, mock_claude):
    main.set_at_junction(onboarded_user, False)
    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="Nice!",
        evaluation=main.Evaluation(
            attempted="tengo hambre",
            correct=True,
            tag="tener-idioms",
            srs_item=main.SrsItem(es="tengo hambre", en="I'm hungry"),
        ),
    )

    await main.tutor_turn(onboarded_user, "tengo hambre")

    stats = main.get_srs_stats(onboarded_user)
    assert stats["total_items"] == 1


@pytest.mark.asyncio
async def test_evaluation_logs_attempt_regardless_of_srs(tmp_db, onboarded_user, mock_claude):
    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="ok",
        evaluation=main.Evaluation(attempted="hola", correct=True, tag="g"),
    )

    await main.tutor_turn(onboarded_user, "hola")

    conn = sqlite3.connect(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    conn.close()
    assert count == 1


# ---------- Robustness ----------


@pytest.mark.asyncio
async def test_parse_failure_returns_fallback_and_keeps_state(tmp_db, onboarded_user, mock_claude):
    """If Claude returns garbage twice, bot returns a fallback message, no state change."""
    mock_claude.messages.parse.side_effect = Exception("malformed JSON")

    before = main.get_learner_state(onboarded_user)
    reply = await main.tutor_turn(onboarded_user, "test")
    after = main.get_learner_state(onboarded_user)

    assert "hiccup" in reply.lower() or "brain" in reply.lower()
    assert before == after


@pytest.mark.asyncio
async def test_parse_succeeds_on_second_attempt(tmp_db, onboarded_user, mock_claude):
    mock_claude.messages.parse.side_effect = [
        Exception("bad JSON"),
        make_tutor_response(reply="ok this time"),
    ]

    reply = await main.tutor_turn(onboarded_user, "hello")
    assert reply == "ok this time"


@pytest.mark.asyncio
async def test_history_preserved_across_turns(tmp_db, onboarded_user, mock_claude):
    """User and bot turns should accumulate in conversation_turns."""
    mock_claude.messages.parse.return_value = make_tutor_response(reply="reply1")
    await main.tutor_turn(onboarded_user, "msg1")
    mock_claude.messages.parse.return_value = make_tutor_response(reply="reply2")
    await main.tutor_turn(onboarded_user, "msg2")

    history = main.get_recent_history(onboarded_user, limit=10)
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "msg1"}
    assert history[1] == {"role": "assistant", "content": "reply1"}
    assert history[2] == {"role": "user", "content": "msg2"}
    assert history[3] == {"role": "assistant", "content": "reply2"}


@pytest.mark.asyncio
async def test_lesson_complete_on_final_b1_lesson_no_advance(tmp_db, onboarded_user, mock_claude):
    """Reaching the very last lesson (B1/30) sets test_pending — no advance to a non-existent next level."""
    main.set_current_lesson(onboarded_user, "B1/30-cultural-literary")
    main.set_at_junction(onboarded_user, False)
    mock_claude.messages.parse.return_value = make_tutor_response(
        reply="You're done!",
        lesson_complete=True,
    )

    await main.tutor_turn(onboarded_user, "got it")

    state, _, _ = main.get_test_state(onboarded_user)
    current_lesson, _, _, _ = main.get_learner_state(onboarded_user)
    assert state == "pending"  # test still triggers at level end
    assert current_lesson == "B1/30-cultural-literary"
