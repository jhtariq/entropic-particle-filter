"""Audio-aware LLM-judge reward models for MMAR (Qwen2.5-Omni as the judge).

The judge hears the SAME audio clip as the policy model and sees the question,
the lettered options, and the candidate text; it returns a 0-100 score as
strict JSON (a 0-10 rubric produced argmax ties on ~80% of items in smoke
testing - the coarse scale could not separate same-answer candidates). Two thin adapters expose the shared core to its_hub's algorithms:

  - `AudioJudgeORM`  -> `BestOfN(orm=...)`: scores complete candidate responses.
  - `AudioJudgePRM`  -> `BeamSearch(sg, prm=...)`: scores PARTIAL reasoning
    chains at every step (with the audio), per the run protocol.

The judge never parses question/choices/audio back out of conversations: the
runner constructs one judge core per item, bound to that item's `MCQRecord`.
JSON parsing reuses `LLMJudge`'s battle-tested extractors; every judge call is
recorded in `trace` for offline judge-quality analysis.
"""

import re

from benchmarking.mmau_pro.audio import audio_content_parts
from benchmarking.mmau_pro.prompt import format_choices
from its_hub.api import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractProcessRewardModel,
    ChatMessage,
    ChatMessages,
)
from its_hub.core.orchestrator import LMOrchestrator
from its_hub.core.reward_models.llm_judge import LLMJudge

FINAL_RUBRIC_SYSTEM = (
    "You are grading a candidate answer to a multiple-choice question about an "
    "audio clip. Listen to the audio, read the question and the options, then "
    "score how likely the candidate's final choice is the CORRECT option, on a "
    "0-100 scale (0 = certainly wrong, 50 = cannot tell, 100 = certainly "
    "correct). Judge only against the audio evidence; well-written reasoning "
    "that contradicts the audio scores low. Use the full 0-100 range to express "
    "fine differences in confidence; do not default to round numbers like 80 - "
    "pick the exact number that reflects your confidence (e.g. 73 or 86). "
    'Return ONLY JSON: {"score": <number 0-100>, "reasoning": "<one or two sentences>"}.'
)

STEP_RUBRIC_SYSTEM = (
    "You are grading a PARTIAL, in-progress reasoning chain for a multiple-choice "
    "question about an audio clip. The reasoning may be unfinished and may not "
    "state a final answer yet - do not penalize it for being incomplete. Listen "
    "to the audio, then score 0-100 how promising the reasoning is so far: is "
    "every claim consistent with what is actually audible, and is it on track "
    "toward the correct option? (0 = contradicts the audio or is heading to a "
    "wrong answer, 100 = fully grounded in the audio and on track). Use the full "
    "0-100 range to express fine differences; do not default to round numbers "
    "like 80 - pick the exact number that reflects your judgment (e.g. 73 or 86). "
    'Return ONLY JSON: {"score": <number 0-100>, "reasoning": "<one or two sentences>"}.'
)


def truncate_middle(text: str, max_chars: int) -> str:
    """Keep the head and tail of over-long candidate text (never the audio)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 5
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


class MMARAudioJudge:
    """Per-item judge core bound to one `MCQRecord`.

    The judge LM and the orchestrator (global judge-side concurrency cap) are
    shared across all instances; only the record binding is per-item.
    """

    def __init__(
        self,
        judge_lm: AbstractLanguageModel,
        rec,
        orchestrator: LMOrchestrator,
        audio_mode: str = "local-path",
        fallback_score: float = 50.0,
        response_format: dict | None = LLMJudge.SCORE_RESPONSE_FORMAT,
        max_candidate_chars: int = 4000,
        judge_max_tokens: int = 256,
        keep_reasoning_chars: int = 300,
    ):
        self.judge_lm = judge_lm
        self.rec = rec
        self.orchestrator = orchestrator
        self.audio_mode = audio_mode
        self.fallback_score = fallback_score
        self.response_format = response_format
        self.max_candidate_chars = max_candidate_chars
        self.judge_max_tokens = judge_max_tokens
        self.keep_reasoning_chars = keep_reasoning_chars
        self.trace: list[dict] = []
        self._parser = LLMJudge(judge_lm, fallback_score=fallback_score)

    def _judge_messages(self, candidate_text: str, partial: bool) -> list[ChatMessage]:
        parts = audio_content_parts(self.rec.audio_paths, mode=self.audio_mode)
        block = "Partial reasoning so far" if partial else "Candidate response"
        text = (
            f"Question: {self.rec.question}\n\n"
            f"Options:\n{format_choices(self.rec.choices)}\n\n"
            f"{block}:\n<<<\n"
            f"{truncate_middle(candidate_text, self.max_candidate_chars)}\n>>>"
        )
        return [
            ChatMessage(
                role="system",
                content=STEP_RUBRIC_SYSTEM if partial else FINAL_RUBRIC_SYSTEM,
            ),
            ChatMessage(role="user", content=[*parts, {"type": "text", "text": text}]),
        ]

    def _parse(self, content: str) -> tuple[float, bool, str]:
        """Parse (raw_score, parse_ok, reasoning) from a judge response."""
        parsed = self._parser._extract_json(content or "")
        if isinstance(parsed, dict) and "score" in parsed:
            try:
                return (
                    float(parsed["score"]),
                    True,
                    str(parsed.get("reasoning", "")),
                )
            except (TypeError, ValueError):
                pass
        match = re.search(r'"score"\s*:\s*(-?[\d.]+)', content or "")
        if match:
            try:
                return float(match.group(1)), True, ""
            except ValueError:
                pass
        return self.fallback_score, False, ""

    async def ascore_texts(self, texts: list[str], partial: bool) -> list[float]:
        """Score candidate texts against the bound record; returns [0,1] scores."""
        prompts = [self._judge_messages(t, partial) for t in texts]
        kwargs = {"temperature": 0.0, "max_tokens": self.judge_max_tokens}
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        responses = await self.orchestrator.agenerate(self.judge_lm, prompts, **kwargs)

        scores = []
        for text, response in zip(texts, responses):
            content = (response.get("content") or "") if isinstance(response, dict) else ""
            raw, parse_ok, reasoning = self._parse(content)
            score = max(0.0, min(100.0, raw)) / 100.0
            scores.append(score)
            self.trace.append(
                {
                    "kind": "step" if partial else "final",
                    "score_raw": raw,
                    "score": score,
                    "parse_ok": parse_ok,
                    "reasoning": reasoning[: self.keep_reasoning_chars],
                    "candidate_chars": len(text),
                }
            )
        return scores


class AudioJudgeORM(AbstractOutcomeRewardModel):
    """BestOfN adapter: scores complete candidates with the FINAL rubric.

    BestOfN hands over full conversations (original messages + the candidate as
    the last assistant message) and its policy orchestrator; the orchestrator is
    deliberately ignored so judge-side load is capped by the shared judge
    orchestrator instead.
    """

    def __init__(self, core: MMARAudioJudge):
        self.core = core

    def score(self, messages, **kwargs):
        raise NotImplementedError("AudioJudgeORM is async-only; use ascore().")

    async def ascore(self, messages, orchestrator=None, **kwargs):
        is_batch = bool(messages) and isinstance(messages[0], list)
        conversations = messages if is_batch else [messages]
        texts = [conv[-1].extract_text_content() or "" for conv in conversations]
        scores = await self.core.ascore_texts(texts, partial=False)
        return scores if is_batch else scores[0]


class AudioJudgePRM(AbstractProcessRewardModel):
    """BeamSearch adapter: scores partial chains with the STEP rubric.

    BeamSearch calls ``ascore(score_target, [partial_response_str, ...])``; the
    partial strings are judged against the bound record's audio + question.
    """

    def __init__(self, core: MMARAudioJudge):
        self.core = core

    def score(self, prompt_or_messages, steps):
        raise NotImplementedError("AudioJudgePRM is async-only; use ascore().")

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        return await self.core.ascore_texts(list(steps), partial=True)
