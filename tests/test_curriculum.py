"""Curriculum loading + lesson traversal."""
import main


def test_loads_all_four_levels():
    levels = {lid.split("/")[0] for lid in main.CURRICULUM.keys()}
    assert {"A0", "A1", "A2", "B1"} <= levels


def test_expected_lesson_counts():
    counts = {}
    for lid in main.CURRICULUM.keys():
        lvl = lid.split("/")[0]
        counts[lvl] = counts.get(lvl, 0) + 1
    assert counts["A0"] == 8
    assert counts["A1"] == 20
    assert counts["A2"] == 25
    assert counts["B1"] == 30


def test_lesson_order_is_sorted():
    assert main.LESSON_ORDER == sorted(main.LESSON_ORDER)


def test_first_lesson_is_a0_01():
    assert main.FIRST_LESSON == "A0/01-greetings"


def test_next_lesson_within_level():
    assert main.next_lesson_id("A0/01-greetings") == "A0/02-introductions"
    assert main.next_lesson_id("A1/05-present-ar-verbs") == "A1/06-present-er-ir-verbs"


def test_next_lesson_crosses_level_boundary():
    assert main.next_lesson_id("A0/08-wants-essentials") == "A1/01-numbers-21-100"
    assert main.next_lesson_id("A1/20-questions-negation") == "A2/01-preterito-indefinido-ar"
    assert main.next_lesson_id("A2/25-hypothetical-real") == "B1/01-subjunctive-formation"


def test_next_lesson_at_end_returns_none():
    assert main.next_lesson_id(main.LESSON_ORDER[-1]) is None


def test_next_lesson_unknown_returns_none():
    assert main.next_lesson_id("XX/99-fake") is None
    assert main.next_lesson_id("") is None


def test_is_last_lesson_of_level():
    assert main.is_last_lesson_of_level("A0/08-wants-essentials")
    assert main.is_last_lesson_of_level("A1/20-questions-negation")
    assert main.is_last_lesson_of_level("A2/25-hypothetical-real")
    assert main.is_last_lesson_of_level("B1/30-cultural-literary")
    assert not main.is_last_lesson_of_level("A0/01-greetings")
    assert not main.is_last_lesson_of_level("A1/05-present-ar-verbs")
    assert not main.is_last_lesson_of_level(None)
    assert not main.is_last_lesson_of_level("XX/99-fake")


def test_every_lesson_has_required_fields():
    """A lesson missing key fields would crash the bot — check them all."""
    required = {"id", "level", "order", "title", "objectives", "vocab"}
    for lid, lesson in main.CURRICULUM.items():
        missing = required - lesson.keys()
        assert not missing, f"{lid} missing fields: {missing}"


def test_every_vocab_item_has_es_en_ru():
    for lid, lesson in main.CURRICULUM.items():
        for v in lesson["vocab"]:
            assert "es" in v, f"{lid} vocab item missing es: {v}"
            assert "en" in v, f"{lid} vocab item missing en: {v}"
            assert "ru" in v, f"{lid} vocab item missing ru: {v}"


def test_lesson_id_matches_filename_pattern():
    """id format 'LEVEL/NN-slug' so sorting works lexicographically."""
    import re
    for lid in main.CURRICULUM:
        assert re.match(r"^[A-Z]\d/\d{2}-[a-z0-9-]+$", lid), f"bad id: {lid}"
