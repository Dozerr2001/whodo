"""LLM failures and dev mode, with no network and no LLM calls. Run with: .venv/bin/python tests/test_llm_errors.py"""

import importlib.util
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_TOKEN", "not-a-real-token")

from whodo import config, extract, pipeline, summarize  # noqa: E402
from whodo.extract import LLM_UNAVAILABLE, ExtractionError, LLMUnavailable  # noqa: E402
from whodo.render import summary_block, summary_text  # noqa: E402

MESSAGE = "The AI service is temporarily unavailable. Try the example instead."


class FakeHTTPError(Exception):
    """Looks like a 402 from the provider."""

    response = type("R", (), {"status_code": 402})()


class FakeClient:
    """chat_completion fails for the first `fail` calls, then answers."""

    def __init__(self, fail):
        self.fail, self.calls = fail, 0

    def chat_completion(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail:
            raise FakeHTTPError("402 Payment Required secret-token-value")
        message = type("M", (), {"content": "{}"})()
        return type("Resp", (), {"choices": [type("C", (), {"message": message})()]})()


def test_message_text():
    assert LLM_UNAVAILABLE == MESSAGE and str(LLMUnavailable()) == MESSAGE
    assert issubclass(LLMUnavailable, ExtractionError)  # the command-line script already handles ExtractionError


def test_both_calls_failing_raises_unavailable_without_leaking_details():
    try:
        extract.ask(FakeClient(fail=2), "m", [], extract.Extraction)
    except LLMUnavailable as e:
        assert str(e) == MESSAGE and "secret-token-value" not in str(e)
    else:
        raise AssertionError("expected LLMUnavailable")


def test_groq_failure_falls_back_to_hf_and_both_failing_is_unavailable():
    groq, hf = FakeClient(fail=2), FakeClient(fail=0)
    assert extract.ask_any([("Groq", groq), ("Hugging Face", hf)], "m", [], extract.Extraction) == "{}"
    assert groq.calls == 2 and hf.calls == 1
    groq, hf = FakeClient(fail=2), FakeClient(fail=2)
    try:
        extract.ask_any([("Groq", groq), ("Hugging Face", hf)], "m", [], extract.Extraction)
    except LLMUnavailable as e:
        assert str(e) == MESSAGE
    else:
        raise AssertionError("expected LLMUnavailable")


def test_groq_is_first_only_when_key_is_set():
    saved = os.environ.pop("GROQ_API_KEY", None)
    try:
        assert [n for n, _ in extract.make_clients("t")] == ["Hugging Face"]
        os.environ["GROQ_API_KEY"] = "fake"
        assert [n for n, _ in extract.make_clients("t")] == ["Groq", "Hugging Face"]
    finally:
        os.environ.pop("GROQ_API_KEY", None)
        if saved:
            os.environ["GROQ_API_KEY"] = saved


class RateLimited(Exception):
    def __init__(self, retry_after=None):
        headers = {"retry-after": retry_after} if retry_after else {}
        self.response = type("R", (), {"status_code": 429, "headers": headers})()


class Always429:
    def __init__(self, retry_after=None, succeed_after=None):
        self.retry_after, self.succeed_after, self.calls = retry_after, succeed_after, 0

    def chat_completion(self, **kwargs):
        self.calls += 1
        if self.succeed_after is not None and self.calls > self.succeed_after:
            return FakeClient(0).chat_completion()
        raise RateLimited(self.retry_after)


def _sleeps_during(fn):
    sleeps, original = [], extract.time.sleep
    extract.time.sleep = sleeps.append
    try:
        fn()
    finally:
        extract.time.sleep = original
    return sleeps


def _expect_unavailable(client):
    try:
        extract.ask(client, "m", [], extract.Extraction)
    except LLMUnavailable:
        return
    raise AssertionError("expected LLMUnavailable")


def test_429_waits_for_retry_after_then_succeeds():
    client = Always429("7", succeed_after=2)
    sleeps = _sleeps_during(lambda: extract.ask(client, "m", [], extract.Extraction))
    assert sleeps == [7.0, 7.0] and client.calls == 3  # two waits, success on the third try


def test_429_without_hint_waits_a_few_seconds_and_gives_up_after_two_retries():
    client = Always429()
    sleeps = _sleeps_during(lambda: _expect_unavailable(client))
    assert sleeps == [extract.RATE_LIMIT_DEFAULT_WAIT] * 4 and client.calls == 6  # 3 tries each for schema, then prompt only


def test_429_with_a_very_long_wait_is_not_retried():
    sleeps = _sleeps_during(lambda: _expect_unavailable(Always429("600")))
    assert sleeps == []


class GroqLimit(Exception):
    """A 429 or 413 shaped like Groq's: limit headers plus a JSON error body."""

    def __init__(self, status=429, reset=None, retry_after=None, message="Limit 8000, Requested 9500"):
        headers = {k: v for k, v in {"x-ratelimit-reset-tokens": reset, "retry-after": retry_after}.items() if v}
        body = {"error": {"message": message}}
        self.response = type("R", (), {"status_code": status, "headers": headers, "json": lambda self: body})()


def test_429_waits_for_the_token_window_to_reset():
    assert extract._retry_wait(GroqLimit(reset="44.257s")) == 44.257 + extract.RESET_MARGIN
    assert extract._retry_wait(GroqLimit(reset="1m5s")) == 65 + extract.RESET_MARGIN  # minutes are understood
    assert extract._retry_wait(GroqLimit(reset="3s", retry_after="20")) == 20  # the longer hint wins
    assert extract._retry_wait(GroqLimit(reset="70s")) == 71  # still under the cap
    assert extract._retry_wait(GroqLimit(reset="74.5s")) is None  # 75.5s with the margin: too long, fall back
    assert extract._retry_wait(GroqLimit(reset="10m")) is None  # a daily limit is not worth waiting for
    assert extract._retry_wait(GroqLimit()) == extract.RATE_LIMIT_DEFAULT_WAIT  # no hints at all
    assert extract._retry_wait(GroqLimit(status=413)) is None  # only a 429 is retried


def test_429_retry_sleeps_the_reset_time():
    class Limited(Always429):
        def chat_completion(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                return FakeClient(0).chat_completion()
            raise GroqLimit(reset="44.257s")

    sleeps = _sleeps_during(lambda: extract.ask(Limited(), "m", [], extract.Extraction))
    assert sleeps == [45.257]


def test_log_line_has_the_limit_message_for_413_and_429_only():
    line = extract._describe(GroqLimit(status=413, message="Request too large for org_01abc23 on TPM: Limit 8000, Requested 9500"))
    assert line.startswith("GroqLimit 413: ") and "Limit 8000, Requested 9500" in line
    assert "org_01abc23" not in line  # the organization id is cut out
    assert "429: Limit 8000" in extract._describe(GroqLimit(status=429))
    assert extract._describe(FakeHTTPError("secret-token-value")) == "FakeHTTPError 402"  # other statuses: type and status only
    assert extract._describe(ValueError("secret-token-value")) == "ValueError"
    assert len(extract._describe(GroqLimit(message="x" * 1000))) < 340  # a long message is cut


def test_reasoning_effort_is_off_by_default_and_only_for_gpt_oss():
    original = config.GROQ_REASONING_EFFORT
    try:
        assert original == "default" and extract.groq_settings("openai/gpt-oss-120b") is None
        config.GROQ_REASONING_EFFORT = "low"
        assert extract.groq_settings("openai/gpt-oss-120b") == {"reasoning_effort": "low"}
        assert extract.groq_settings("Qwen/Qwen3-235B-A22B-Instruct-2507") is None  # other models do not take it
    finally:
        config.GROQ_REASONING_EFFORT = original


def _messages(chars):
    return [{"role": "system", "content": "x" * 1000}, {"role": "user", "content": "y" * (chars - 1000)}]


def test_a_request_over_groqs_limit_goes_straight_to_hf():
    big = _messages(22061)  # the real 19-minute meeting: Groq refused it with 413 "Requested 9724"
    assert 9700 < extract.estimated_request_tokens(big) < 9760
    groq, hf = FakeClient(fail=0), FakeClient(fail=0)
    sleeps = _sleeps_during(lambda: extract.ask_any([("Groq", groq), ("Hugging Face", hf)], "m", big, extract.Extraction))
    assert groq.calls == 0 and hf.calls == 1 and sleeps == []  # Groq never called, nothing waited

    small = _messages(14501)  # the first 10 minutes of it: completed on Groq (about 7,770 estimated)
    assert extract.estimated_request_tokens(small) <= config.GROQ_TOKEN_LIMIT
    groq, hf = FakeClient(fail=0), FakeClient(fail=0)
    extract.ask_any([("Groq", groq), ("Hugging Face", hf)], "m", small, extract.Extraction)
    assert groq.calls == 1 and hf.calls == 0


def test_groq_limit_can_be_raised_and_a_lone_groq_is_still_tried():
    original = config.GROQ_TOKEN_LIMIT
    try:
        big = _messages(22061)
        config.GROQ_TOKEN_LIMIT = 20000  # e.g. a paid plan
        groq, hf = FakeClient(fail=0), FakeClient(fail=0)
        extract.ask_any([("Groq", groq), ("Hugging Face", hf)], "m", big, extract.Extraction)
        assert groq.calls == 1 and hf.calls == 0
        config.GROQ_TOKEN_LIMIT = 8000
        only = FakeClient(fail=0)  # with no fallback there is nothing to skip to
        extract.ask_any([("Groq", only)], "m", big, extract.Extraction)
        assert only.calls == 1
    finally:
        config.GROQ_TOKEN_LIMIT = original


def test_fallback_without_schema_still_works():
    client = FakeClient(fail=1)
    assert extract.ask(client, "m", [], extract.Extraction) == "{}" and client.calls == 2


def test_failed_summary_is_reported_not_hidden(monkeypatch=None):
    canned = lambda *a, **k: {"speaker_map": {}, "decisions": [], "action_items": [], "unassigned_items": [], "open_questions": []}
    original = (summarize.extract_to_dict, summarize.summarize)
    summarize.extract_to_dict = canned
    try:
        for error, expected in [(LLMUnavailable(), summarize.SUMMARY_UNAVAILABLE), (ValueError("x"), summarize.SUMMARY_FAILED)]:
            def failing(*a, _e=error, **k):
                raise _e
            summarize.summarize = failing
            result = summarize.extract_with_summary([], "m", "t")
            assert result["summary"] is None and result["summary_error"] == expected
            assert MESSAGE in summarize.SUMMARY_UNAVAILABLE
        summarize.summarize = lambda *a, **k: {"overview": "ok", "topics": []}
        ok = summarize.extract_with_summary([], "m", "t")
        assert ok["summary"]["overview"] == "ok" and ok["summary_error"] is None

        def extraction_down(*a, **k):
            raise LLMUnavailable()
        summarize.extract_to_dict = extraction_down  # if the extraction itself fails, the run fails
        try:
            summarize.extract_with_summary([], "m", "t")
        except LLMUnavailable:
            pass
        else:
            raise AssertionError("expected LLMUnavailable")
    finally:
        summarize.extract_to_dict, summarize.summarize = original


def test_summary_failure_shows_a_notice_and_copies_nothing():
    ex = {"summary": None, "summary_error": "No summary <this time>."}
    html = summary_block(ex)
    assert "mm-notice" in html and "No summary &lt;this time&gt;." in html
    assert summary_text(ex) == ""
    assert summary_block({"summary": None}) == ""  # an older saved result with no summary and no error


def test_dev_mode_never_calls_the_llm():
    original = config.DEV_LLM
    try:
        config.DEV_LLM = "saved"
        saved = pipeline.dev_extraction([], "m", "t", None)
        assert saved["summary"]["overview"] and len(saved["action_items"]) > 0
        config.DEV_LLM = "summary-outage"
        half = pipeline.dev_extraction([], "m", "t", None)
        assert half["summary"] is None and half["summary_error"] == summarize.SUMMARY_UNAVAILABLE and half["action_items"]
        config.DEV_LLM = "outage"
        try:
            pipeline.dev_extraction([], "m", "t", None)
        except LLMUnavailable:
            pass
        else:
            raise AssertionError("expected LLMUnavailable")
    finally:
        config.DEV_LLM = original


def _load_app():
    spec = importlib.util.spec_from_file_location("mm_app", ROOT / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_shows_the_message_and_keeps_buttons_working():
    app = _load_app()
    audio = str(ROOT / "test_data" / "meeting_01.mp3")

    def outage_run(*a, **k):
        yield pipeline.Progress(3, "extracting", "Extracting", 0.0)
        raise LLMUnavailable()

    app.run = outage_run
    updates = list(app.stream_run(audio, "", None, date(2026, 10, 26), "", app.RunControl()))
    status, _banner, results = updates[-1][:3]
    assert status == f"⚠️ {MESSAGE}" and results["visible"] is False  # message where the progress was; no results shown
    assert "Step 3" not in status

    result = pipeline.Result([], {**pipeline.json.loads(pipeline.DEV_SAVED_RESULT.read_text()), "summary": None,
                                  "summary_error": summarize.SUMMARY_UNAVAILABLE},
                          timings={"total": 5.0})
    app.run = lambda *a, **k: iter([result])
    final = list(app.stream_run(audio, "", None, date(2026, 10, 26), "", app.RunControl()))[-1]
    assert "summary could not be written" in final[0]  # the status line says so
    assert summarize.SUMMARY_UNAVAILABLE in final[4]  # and the notice is where the summary would be
    assert final[3] == ""  # nothing for the Copy summary button to copy


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
