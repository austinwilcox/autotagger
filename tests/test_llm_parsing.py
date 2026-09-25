from autotagger.llm import _extract_json


def test_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_markdown_fenced():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_thinking_block_is_stripped():
    raw = '<think>Let me consider the durations...</think>\n{"selected_index": 2}'
    assert _extract_json(raw) == {"selected_index": 2}


def test_prose_before_and_after():
    raw = 'Here is my answer:\n{"selected_index": null, "confidence": 0.1}\nHope that helps!'
    assert _extract_json(raw)["selected_index"] is None


def test_braces_inside_strings_do_not_confuse_the_scanner():
    raw = 'x {"notes": "the title is {weird}", "ok": true} y'
    assert _extract_json(raw)["notes"] == "the title is {weird}"


def test_garbage_returns_none():
    assert _extract_json("I could not determine the track.") is None
