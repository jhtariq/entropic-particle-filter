"""Offline tests for the Mellow serving shim's pure layer — no GPU/model/server."""

import base64
import os

import numpy as np
import pytest

from benchmarking.mmau_pro.serve_mellow import (
    ParsedRequest,
    RequestError,
    apply_stop_strings,
    build_response_payload,
    fit_waveform,
    flatten_messages,
    parse_request,
    resolve_audio_paths,
)
from its_hub.core.utils import summarize_step_logprobs


def _audio_url_part(path):
    return {"type": "audio_url", "audio_url": {"url": f"file://{path}"}}


def _pf_step_body():
    """The exact request shape a PF/EPF step puts on the wire."""
    return {
        "model": "mellow",
        "messages": [
            {"role": "system", "content": "You are an expert audio analyst."},
            {
                "role": "user",
                "content": [
                    _audio_url_part("/data/clip.wav"),
                    {"type": "text", "text": "Question: what do you hear?\n\nOptions:\nA. dog\nB. cat"},
                ],
            },
            {"role": "assistant", "content": "Step 1: I hear barking."},
        ],
        "extra_body": {
            "add_generation_prompt": False,
            "continue_final_message": True,
            "include_stop_str_in_output": False,
        },
        "add_generation_prompt": False,
        "continue_final_message": True,
        "include_stop_str_in_output": False,
        "stop": "\n\n",
        "max_tokens": 300,
        "temperature": 0.8,
        "logprobs": True,
        "top_logprobs": 20,
    }


# ---------------------------- parse_request ---------------------------- #


def test_parse_pf_step_request():
    req = parse_request(_pf_step_body())
    assert req.continuation == "Step 1: I hear barking."
    assert req.prompt_text.startswith("You are an expert audio analyst.")
    assert "Question: what do you hear?" in req.prompt_text
    assert len(req.audio_refs) == 1 and req.audio_refs[0].kind == "file"
    assert req.stop == ["\n\n"]
    assert req.max_tokens == 300
    assert req.temperature == 0.8
    assert req.logprobs is True
    assert req.top_logprobs == 20
    assert req.include_stop_str is False


def test_parse_greedy_screen_request():
    body = {
        "model": "mellow",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 700,
    }
    req = parse_request(body)
    assert req.continuation == ""
    assert req.temperature == 0.0
    assert req.logprobs is False and req.top_logprobs is None
    assert req.stop == []


def test_parse_flags_from_extra_body_only():
    body = _pf_step_body()
    del body["include_stop_str_in_output"]
    body["extra_body"]["include_stop_str_in_output"] = True
    assert parse_request(body).include_stop_str is True


def test_parse_default_max_tokens_and_bad_messages():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert parse_request(body, default_max_tokens=99).max_tokens == 99
    with pytest.raises(RequestError):
        parse_request({"messages": []})
    with pytest.raises(RequestError):
        parse_request({})


def test_parse_trailing_assistant_must_be_string():
    body = _pf_step_body()
    body["messages"][-1]["content"] = [{"type": "text", "text": "x"}]
    with pytest.raises(RequestError):
        parse_request(body)


# ---------------------------- flatten_messages ---------------------------- #


def test_flatten_joins_text_in_order_and_collects_audio():
    messages = [
        {"role": "system", "content": "SYS"},
        {
            "role": "user",
            "content": [
                _audio_url_part("/a/1.wav"),
                _audio_url_part("/a/2.wav"),
                {"type": "text", "text": "USR"},
            ],
        },
    ]
    text, refs = flatten_messages(messages)
    assert text == "SYS\n\nUSR"
    assert [r.path for r in refs] == ["/a/1.wav", "/a/2.wav"]


def test_flatten_zero_audio_ok():
    # ab_causality strips audio parts entirely in its "without" arm
    text, refs = flatten_messages([{"role": "user", "content": [{"type": "text", "text": "q"}]}])
    assert text == "q" and refs == []


def test_three_audios_rejected():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    _audio_url_part("/a/1.wav"),
                    _audio_url_part("/a/2.wav"),
                    _audio_url_part("/a/3.wav"),
                    {"type": "text", "text": "q"},
                ],
            }
        ]
    }
    with pytest.raises(RequestError, match="at most 2 audio"):
        parse_request(body)


def test_input_audio_b64_and_bad_parts():
    data = b"RIFFfakewav"
    part = {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": "wav"}}
    _, refs = flatten_messages([{"role": "user", "content": [part]}])
    assert refs[0].kind == "bytes" and refs[0].data == data and refs[0].fmt == "wav"
    with pytest.raises(RequestError):
        flatten_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])
    with pytest.raises(RequestError):
        flatten_messages([{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "not-b64!!", "format": "wav"}}]}])
    with pytest.raises(RequestError):
        flatten_messages([{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "https://x/y.wav"}}]}])


# ---------------------------- allowed media roots ---------------------------- #


def test_resolve_audio_paths_allowed_roots(tmp_path):
    inside = tmp_path / "root" / "clip.wav"
    inside.parent.mkdir()
    inside.write_bytes(b"x")
    outside = tmp_path / "other.wav"
    outside.write_bytes(b"x")

    body = {"messages": [{"role": "user", "content": [
        _audio_url_part(str(inside)), {"type": "text", "text": "q"}]}]}
    req = parse_request(body)
    resolve_audio_paths(req.audio_refs, [str(tmp_path / "root")])
    assert req.audio_refs[0].path == os.path.realpath(str(inside))

    body["messages"][0]["content"][0] = _audio_url_part(str(outside))
    req = parse_request(body)
    with pytest.raises(RequestError, match="outside the allowed"):
        resolve_audio_paths(req.audio_refs, [str(tmp_path / "root")])

    body["messages"][0]["content"][0] = _audio_url_part(str(tmp_path / "root" / "missing.wav"))
    req = parse_request(body)
    with pytest.raises(RequestError, match="not found"):
        resolve_audio_paths(req.audio_refs, [str(tmp_path / "root")])


# ---------------------------- stop strings ---------------------------- #


def test_stop_string_basic_exclude_and_include():
    # tokens: ["Hello"," wor","ld","\n\n","next"] -> cumulative decoded lengths
    text = "Hello world\n\nnext"
    cum = [5, 9, 11, 13, 17]
    kept, n, hit = apply_stop_strings(text, cum, ["\n\n"], include_stop=False)
    assert (kept, n, hit) == ("Hello world", 3, True)
    kept, n, hit = apply_stop_strings(text, cum, ["\n\n"], include_stop=True)
    assert (kept, n, hit) == ("Hello world\n\n", 4, True)


def test_stop_string_spanning_token_boundary():
    # "\n\n" split across two tokens: ["a\n", "\nb"]
    text = "a\n\nb"
    cum = [2, 4]
    kept, n, hit = apply_stop_strings(text, cum, ["\n\n"], include_stop=False)
    assert (kept, hit) == ("a", True)
    assert n == 1  # first token covers the kept text


def test_stop_string_earliest_of_multiple_and_none():
    text = "abc STOP def END"
    cum = [len(text)]
    kept, _, hit = apply_stop_strings(text, cum, ["END", "STOP"], include_stop=False)
    assert kept == "abc " and hit is True
    kept, n, hit = apply_stop_strings("no stops here", [13], ["\n\n"], include_stop=False)
    assert (kept, n, hit) == ("no stops here", 1, False)


# ---------------------------- waveform fitting ---------------------------- #


def test_fit_waveform_pad_by_repetition_and_first_crop():
    short = np.arange(4, dtype=np.float32)
    fitted = fit_waveform(short, 10)
    assert fitted.shape == (10,)
    assert np.array_equal(fitted, np.tile(short, 3)[:10])  # reference repeat math

    long = np.arange(100, dtype=np.float32)
    fitted = fit_waveform(long, 10)
    assert np.array_equal(fitted, long[:10])  # deterministic FIRST crop
    assert np.array_equal(fit_waveform(long, 10), fitted)  # deterministic across calls

    with pytest.raises(RequestError):
        fit_waveform(np.zeros(0, dtype=np.float32), 10)


# ---------------------------- response payload ---------------------------- #


def _result(entries):
    return {
        "content": "a dog barking",
        "logprob_entries": entries,
        "prompt_tokens": 519,
        "completion_tokens": 3,
        "finish_reason": "stop",
    }


def test_response_payload_without_logprobs():
    payload = build_response_payload("mellow", 1, {**_result(None)})
    choice = payload["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "a dog barking"}
    assert choice["logprobs"] is None  # client guards on .get("logprobs") is not None
    assert payload["usage"]["prompt_tokens"] == 519
    assert payload["usage"]["total_tokens"] == 522
    assert payload["model"] == "mellow"


def test_response_payload_feeds_summarize_step_logprobs():
    entries = [
        {"token": "a", "logprob": -0.5,
         "top_logprobs": [{"token": "a", "logprob": -0.5}, {"token": "b", "logprob": -1.5}]},
        {"token": "dog", "logprob": -0.25,
         "top_logprobs": [{"token": "dog", "logprob": -0.25}, {"token": "cat", "logprob": -2.0}]},
    ]
    payload = build_response_payload("mellow", 1, _result(entries))
    summary = summarize_step_logprobs(payload["choices"][0]["logprobs"])
    assert summary["num_tokens"] == 2
    assert summary["mean_logprob"] == pytest.approx(-0.375)
    assert summary["entropy"] is not None and summary["entropy"] > 0


def test_response_payload_mean_logprob_only_no_top():
    # PF/mean_logprob cells request logprobs without top_logprobs
    entries = [{"token": "a", "logprob": -0.5}, {"token": "b", "logprob": -1.0}]
    payload = build_response_payload("mellow", 1, _result(entries))
    summary = summarize_step_logprobs(payload["choices"][0]["logprobs"])
    assert summary["mean_logprob"] == pytest.approx(-0.75)
    assert summary["entropy"] is None


def test_parsed_request_dataclass_defaults():
    req = ParsedRequest(
        prompt_text="p", continuation="", audio_refs=[], stop=[], max_tokens=10,
        temperature=1.0, top_p=None, logprobs=False, top_logprobs=None,
        include_stop_str=False,
    )
    assert req.extras == {}
