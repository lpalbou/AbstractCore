"""Live proof that pixels reach the MLX forward pass (backlog 0840).

Gated behind two env vars: the weight gate the rest of the suite already uses,
plus a vision-specific one, because this lane additionally needs `mlx-vlm` and a
sighted checkpoint that a plain MLX developer may not have.

    ABSTRACTCORE_RUN_MLX_TESTS=1 ABSTRACTCORE_RUN_MLX_VISION_TESTS=1 pytest ...
"""

import os
import random
import string

import pytest

MODEL = os.getenv("ABSTRACTCORE_MLX_VISION_TEST_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit")

pytestmark = pytest.mark.slow


def _gate():
    if os.getenv("ABSTRACTCORE_RUN_MLX_TESTS", "0") != "1":
        pytest.skip("Set ABSTRACTCORE_RUN_MLX_TESTS=1 to run MLX weight tests")
    if os.getenv("ABSTRACTCORE_RUN_MLX_VISION_TESTS", "0") != "1":
        pytest.skip("Set ABSTRACTCORE_RUN_MLX_VISION_TESTS=1 to run MLX vision tests")


_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _big_font(size):
    from PIL import ImageFont

    for candidate in _FONT_CANDIDATES:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return None


# Six colours a small VLM names reliably. The discriminator is deliberately NOT
# text: OCR quality varies sharply with model size, and a fixture that a 4B
# cannot read makes the test fail for a reason unrelated to the transport.
_COLORS = {
    "red": (220, 20, 20),
    "green": (20, 160, 60),
    "blue": (30, 80, 220),
    "yellow": (245, 210, 20),
    "purple": (140, 40, 180),
    "orange": (240, 130, 20),
}


def _payload_image(tmp_path, color_name):
    """A canvas whose only content is a colour chosen at call time.

    Nothing in the prompt names it, so no prior can satisfy the assertion: a
    blind model has a 1-in-6 chance per arm, and the structural assertion below
    does not depend on the content check at all.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([(56, 56), (456, 456)], fill=_COLORS[color_name])
    path = tmp_path / f"{color_name}.png"
    img.save(path)
    return str(path)


def test_answer_tracks_the_pixels(tmp_path):
    """The single highest-confidence assertion in this suite.

    Two calls whose TEXT is byte-identical, differing only in pixels. A
    transport that cannot carry pixels is mathematically incapable of producing
    two different greedy answers, so this falsifies every silent-text-fallback
    path at once -- including a cached prefix answering from the previous image,
    since two same-size images tokenize identically.
    """
    _gate()
    from abstractcore import create_llm

    picks = random.sample(sorted(_COLORS), 2)
    img_a = _payload_image(tmp_path, picks[0])
    img_b = _payload_image(tmp_path, picks[1])
    question = "What colour is the circle in this image? Answer with one word."

    llm = create_llm("mlx", model=MODEL)
    a = llm.generate(question, media=[img_a], max_tokens=24, temperature=0.0)
    b = llm.generate(question, media=[img_b], max_tokens=24, temperature=0.0)

    # Determinism control: without media the same prompt must be reproducible,
    # otherwise a difference above proves nothing about pixels.
    c1 = llm.generate(question, max_tokens=24, temperature=0.0)
    c2 = llm.generate(question, max_tokens=24, temperature=0.0)
    assert c1.content == c2.content, "greedy harness is not deterministic"

    assert a.content != b.content, "identical text prompts gave identical answers"

    # Content check: each answer must name the colour it was shown and not the
    # one it was not. Upgrades "the image changed the answer" to "the image was
    # read correctly", at a 1-in-6 prior per arm.
    for answer, shown, other in ((a.content, picks[0], picks[1]), (b.content, picks[1], picks[0])):
        low = answer.lower()
        assert shown in low, f"{answer!r} does not name {shown!r}"
        assert other not in low, f"{answer!r} names the colour it was not shown"


def test_delivery_record_reports_measured_expansion(tmp_path):
    """`tokens` must be measured off the expanded ids, never intended. A 1024x1024
    image expands to 1024 placeholder tokens on this family -- a number no
    accidental code path produces."""
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=MODEL)
    r = llm.generate(
        "Describe this image in one short sentence.",
        media=["examples/media/image2.png"],
        max_tokens=48,
    )
    delivered = (r.metadata or {}).get("media_delivered")
    assert delivered, "a sighted checkpoint must report positive delivery"
    assert delivered[0]["tokens"] == 1024
    assert delivered[0]["transport"] == "mlx_vision_addon"
    # ADR 0001: the known 1-D RoPE substitution is annotated, not absorbed.
    assert "rope_1d_substituted" in delivered[0]["fidelity"]
    assert "media_dropped" not in (r.metadata or {})
    # Not-an-error is what this assertion is for. It cannot be `== "stop"`: a
    # thinking checkpoint spends a 48-token budget reasoning and is genuinely cut
    # at the limit, which now reports `length` instead of claiming the model chose
    # to stop. See `test_truncated_generation_is_reported_as_length_not_stop`.
    assert r.finish_reason in ("stop", "length")


def test_text_only_request_carries_no_media_keys():
    """A request with no media must be byte-identical to before this lane existed."""
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=MODEL)
    r = llm.generate("Say the word ready.", max_tokens=12)
    meta = r.metadata or {}
    assert "media_delivered" not in meta
    assert "media_dropped" not in meta


def test_prompt_cache_is_bypassed_on_a_media_turn(tmp_path):
    """A media turn must not enter key-mode delta feed: an image turn's ids
    contain N identical placeholder tokens, so the same question with a different
    same-size image matches perfectly and would serve the previous image's KV."""
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=MODEL)
    key = "vision-bypass-probe"
    llm.generate("Remember the number 41.", max_tokens=12, prompt_cache_key=key)
    r = llm.generate(
        "Describe this image in one short sentence.",
        media=["examples/media/image2.png"],
        max_tokens=32,
        prompt_cache_key=key,
    )
    # The media turn produced an answer and did not record itself into the key's
    # fed-token ledger, so the text lane's cache is left consistent.
    assert (r.metadata or {}).get("media_delivered")
    assert "prompt_cache" not in (r.metadata or {})


def test_structured_output_with_media_refuses_loudly():
    """Structured output returns a validated model, so there is no metadata
    channel on which the provider could admit it dropped the image. It must
    refuse rather than silently answer from text."""
    _gate()
    from pydantic import BaseModel

    from abstractcore import create_llm

    class Scene(BaseModel):
        summary: str

    llm = create_llm("mlx", model=MODEL)
    with pytest.raises(Exception) as excinfo:
        llm.generate(
            "Describe this image.",
            media=["examples/media/image2.png"],
            response_model=Scene,
        )
    assert "structured_output_unsupported" in str(excinfo.value)


def test_streaming_sees_the_image_too(tmp_path):
    """Streaming must feed the encoder's embeddings, not just the text prompt.

    Regression test for a real defect: the delivery report was written by the
    encoder, but the streaming branch returned the TEXT prompt and never passed
    `input_embeddings`. The model answered blind while the response carried a
    positive `media_delivered` record -- the exact "claims sight, answers from
    text" failure the whole contract exists to prevent, in the one request shape
    the funnel docstring names.
    """
    _gate()
    from abstractcore import create_llm

    picks = random.sample(sorted(_COLORS), 2)
    img = _payload_image(tmp_path, picks[0])
    question = "What colour is the circle in this image? Answer with one word."

    llm = create_llm("mlx", model=MODEL)
    streamed = "".join(
        (c.content or "")
        for c in llm.generate(question, media=[img], max_tokens=24, temperature=0.0, stream=True)
    )
    assert picks[0] in streamed.lower(), f"streamed answer {streamed!r} is blind"
    assert picks[1] not in streamed.lower()


def test_streaming_carries_the_delivery_record(tmp_path):
    """And the record must survive to the consumer, on the same request."""
    _gate()
    from abstractcore import create_llm
    from abstractcore.media.delivery import media_delivery_verdict

    img = _payload_image(tmp_path, "red")
    llm = create_llm("mlx", model=MODEL)
    chunks = list(
        llm.generate(
            "What colour is the circle?", media=[img], max_tokens=16, temperature=0.0, stream=True
        )
    )
    assert chunks
    verdicts = [media_delivery_verdict(c, provider="mlx").state for c in chunks]
    assert "delivered" in verdicts, f"no chunk reported delivery: {set(verdicts)}"


def test_thinking_control_reaches_the_vision_lane(tmp_path):
    """The vision lane renders its own prompt for the encoder, and the encoder's
    ids are what the decoder consumes -- so if that render ignores thinking
    control, the control is silently lost for image requests only.

    Regression test: this was real. A model that reasons freely spends a small
    budget thinking (empty content) and can talk itself out of a correct reading.
    """
    _gate()
    from abstractcore import create_llm

    img = _payload_image(tmp_path, "red")
    llm = create_llm("mlx", model=MODEL)
    addon_prompt = llm._build_prompt("x", None, None, None, enable_thinking=False)
    assert addon_prompt.endswith(
        "<think>\n\n</think>\n\n"
    ), "precondition: enable_thinking=False must emit the disabled-thinking prefill"

    r = llm.generate(
        "What colour is the circle? Answer with one word.",
        media=[img],
        max_tokens=32,
        temperature=0.0,
        thinking="off",
    )
    # With thinking off the model must answer directly, not reason.
    assert not (r.reasoning or "").strip(), "thinking control did not reach the vision lane"
    assert "red" in (r.content or "").lower()


DEEPSTACK_MODEL = os.getenv(
    "ABSTRACTCORE_MLX_DEEPSTACK_TEST_MODEL",
    "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
)


def _word_image(tmp_path, word):
    from PIL import Image, ImageDraw

    font = _big_font(120)
    if font is None:
        pytest.skip("no scalable font available")
    img = Image.new("RGB", (1000, 320), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), word, font=font)
    d.text(
        ((1000 - (box[2] - box[0])) // 2, (320 - (box[3] - box[1])) // 2 - box[1]),
        word,
        fill=(0, 0, 0),
        font=font,
    )
    p = tmp_path / f"{word}.png"
    img.save(p)
    return str(p)


def test_deepstack_features_are_carried(tmp_path):
    """Qwen3-VL's tower returns multi-scale features that its own runtime adds
    into the hidden states after the first few decoder layers. Dropping them is
    not free: the same image reads "MARMARMALMALADEADE" without them and
    "MARMALADE" with them. This asserts we carry them.
    """
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=DEEPSTACK_MODEL)
    q = "What word is printed in the image? Answer with the word only."
    a = llm.generate(
        q,
        media=[_word_image(tmp_path, "MARMALADE")],
        max_tokens=64,
        temperature=0.0,
        thinking="off",
    )
    assert "MARMALADE" in (a.content or "").upper()
    # The duplication signature of the dropped-features failure must be absent.
    assert "MARMARM" not in (a.content or "").upper()


def test_deepstack_does_not_leak_between_requests(tmp_path):
    """The wrappers hold ONE request's features. If they outlived the request a
    later prompt would inherit a previous image's detail, and a text-only request
    would carry visual residue."""
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=DEEPSTACK_MODEL)
    q = "What word is printed in the image? Answer with the word only."
    first = llm.generate(
        q,
        media=[_word_image(tmp_path, "MARMALADE")],
        max_tokens=64,
        temperature=0.0,
        thinking="off",
    )
    second = llm.generate(
        q, media=[_word_image(tmp_path, "ZEPPELIN")], max_tokens=64, temperature=0.0, thinking="off"
    )
    assert "MARMALADE" in (first.content or "").upper()
    assert "ZEPPELIN" in (second.content or "").upper()
    assert "MARMALADE" not in (second.content or "").upper()

    # And the model still answers plain text afterwards.
    third = llm.generate("Name a colour. One word.", max_tokens=12, temperature=0.0, thinking="off")
    assert (third.content or "").strip()


def test_prompt_tokens_include_the_image_expansion():
    """`usage.input_tokens` must count the pixels the model actually processed.

    The text estimator cannot see the expanded placeholder tokens, so an image
    worth ~11.8k tokens was reported as a 56-token prompt -- which silently
    wrecks every context meter and cost figure built on this number.
    """
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=MODEL)
    r = llm.generate(
        "Answer with one word: is this a photograph?",
        media=["examples/media/image2.png"],
        max_tokens=32,
    )
    delivered = (r.metadata or {}).get("media_delivered")
    assert delivered, "a sighted checkpoint must report positive delivery"
    image_tokens = int(delivered[0]["tokens"])
    assert image_tokens > 1
    # The prompt must account for at least the image; a text-only estimate cannot.
    assert r.usage["input_tokens"] >= image_tokens, (
        f"prompt reported {r.usage['input_tokens']} tokens for an image that "
        f"expanded to {image_tokens}"
    )
    assert r.usage["prompt_tokens"] == r.usage["input_tokens"]
    assert r.usage["total_tokens"] == r.usage["input_tokens"] + r.usage["output_tokens"]


def test_truncated_generation_is_reported_as_length_not_stop():
    """A response cut at `max_tokens` must not look like one the model ended.

    A thinking model on a small budget spends the whole allowance reasoning and
    returns empty content. Reported as `stop` with `output_tokens: 0`, that reads
    as a model that had nothing to say on a free call -- so callers neither retry
    nor notice the runaway reasoning they paid for.
    """
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=MODEL)
    # A budget no answer can fit, so the outcome does not depend on how verbose
    # the checkpoint happens to be or on whatever thinking default is in effect --
    # an earlier version keyed on a 60-token budget and passed or failed with test
    # ORDER, which is not a property of the code under test.
    r = llm.generate("Write a 500-word essay on the history of cartography.", max_tokens=16)
    assert r.finish_reason == "length", (
        f"a 16-token budget on a 500-word essay reported {r.finish_reason!r}"
    )
    # Tokens the model EMITTED, not tokens that survived thinking-tag stripping.
    assert r.usage["output_tokens"] > 0, "generated tokens reported as zero"
    assert r.usage["completion_tokens"] == r.usage["output_tokens"]

    # The invariant behind the assertion above, stated directly: hitting the
    # budget and reporting a chosen stop cannot both be true.
    if r.usage["output_tokens"] >= 16:
        assert r.finish_reason == "length"
