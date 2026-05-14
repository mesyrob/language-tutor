"""Preview the bot's system prompt at any point in the curriculum.

Usage:
    uv run inspect_prompt.py                       # mid-B1 by default
    uv run inspect_prompt.py B1/15-passive-ser     # specific lesson
    uv run inspect_prompt.py A1/05-present-ar-verbs
    uv run inspect_prompt.py --junction A2/10-conditional   # at_junction = True

Shows the exact text the bot would receive as system prompt at that point,
plus an approximate token count.
"""

import sys

from main import LESSON_ORDER, build_tutor_system_prompt


def main() -> None:
    args = sys.argv[1:]
    at_junction = False
    if "--junction" in args:
        at_junction = True
        args.remove("--junction")

    target = args[0] if args else "B1/15-passive-ser"

    if target not in LESSON_ORDER:
        print(f"Unknown lesson id: {target}", file=sys.stderr)
        print(f"Available IDs:", file=sys.stderr)
        for lid in LESSON_ORDER:
            print(f"  {lid}", file=sys.stderr)
        sys.exit(1)

    # Pretend everything before target is completed.
    completed = []
    for lid in LESSON_ORDER:
        if lid == target:
            break
        completed.append(lid)

    fake_user = {
        "name": "Lena",
        "goal": "to talk to her partner's family",
        "native_lang": "ru",
        "known_languages": '[{"lang": "en", "level": "B2"}]',
        "self_reported_spanish_level": "A0",
        "assessed_spanish_level": "B1",
    }

    fake_learner_model = {
        "assessed_level": "B1",
        "known_vocab": [
            "hola", "buenos días", "tener hambre", "me gusta", "soy", "estar cansada",
            "fui a Madrid", "había comido", "se vende", "ojalá venga",
        ],
        "weak_grammar": [
            "ser-vs-estar",
            "por-vs-para",
            "subjunctive-with-emotion",
            "preterito-vs-imperfecto",
        ],
        "recent_errors": [
            "said 'la día' (día is masculine)",
            "used indicative after 'es importante que'",
            "forgot stem change in 'querer'",
            "mixed up 'fui' (ser) vs 'fui' (ir) — context error",
        ],
        "current_topic": "passive voice",
    }

    prompt = build_tutor_system_prompt(
        fake_user,
        target,
        at_junction,
        completed,
        fake_learner_model,
    )

    print(prompt)
    print(file=sys.stderr)
    print(f"--- target: {target}  at_junction: {at_junction}", file=sys.stderr)
    print(f"--- completed_lessons: {len(completed)} lessons before target", file=sys.stderr)
    print(f"--- prompt size: {len(prompt)} chars  (~{len(prompt) // 4} tokens)", file=sys.stderr)


if __name__ == "__main__":
    main()
