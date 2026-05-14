"""Test helpers for building fake Claude responses."""
from unittest.mock import MagicMock

import main


def make_tutor_response(reply="ok", **fields):
    """Build a fake claude.messages.parse() return shaped like the real one."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.parsed_output = main.TutorTurn(
        reply=reply,
        learner_model_update=main.LearnerModelUpdate(),
        **fields,
    )
    return response


def make_test_plan(items):
    """Build a fake response for generate_test_for_level."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.parsed_output = main.TestPlan(items=items)
    return response
