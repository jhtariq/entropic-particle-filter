"""Async client for the Qwen3-Omni Yes/No judge used by the judge-weighted PF
arms in probe_epf.py.

Judge config is EXACTLY prm_judge/qwen_omni_orm.py (validated offline
2026-09-02: within-item AUC 0.603 on MMAR/7B b8 candidates, 74 calls/s on one
H200): Yes/No verification prompt over (audio, question+options, candidate),
temperature 1.0, max_tokens 1, top-20 logprobs, P(Yes) = yes_mass /
(yes_mass + no_mass), 0.5 when neither token appears in the top-20.

A failed call (after client retries) returns None — the PF arm treats that as
a neutral weight rather than failing the whole item.
"""

import asyncio
import math
import os

from openai import AsyncOpenAI

from benchmarking.mmau_pro.prompt import format_choices

JUDGE_PROMPT_TEMPLATE = """## ROLE
You are an expert evaluator judging answers to audio reasoning questions.

## TASK
Listen to the audio and read the question below. Determine whether the
candidate answer (including its reasoning) is CORRECT and well-supported
by the audio.

## Question
{question}

## Candidate Answer
{candidate}

## INSTRUCTION
Respond with exactly one word: "Yes" if the candidate answer is correct,
or "No" if it is incorrect. Do not provide any explanation.
"""

JUDGE_PROMPT_TEMPLATE_STEP = """## ROLE
You are an expert evaluator judging step-by-step reasoning on audio reasoning questions.

## TASK
Listen to the audio and read the question below. You are shown the candidate's
reasoning so far (context only) and its LATEST STEP. The reasoning is still in
progress — it may not state a final answer yet, and it must NOT be penalized
for being incomplete. Judge ONLY the latest step: is it correct, consistent
with what is actually audible in the audio, and does it advance toward the
correct option?

## Question
{question}

## Reasoning so far (context)
{context}

## Latest step (evaluate this)
{latest_step}

## INSTRUCTION
Respond with exactly one word: "Yes" if the latest step is correct and
advances toward the right answer, or "No" if it contains a mistake,
contradicts the audio, or leads toward a wrong answer. Do not provide any
explanation.
"""

TOP_LOGPROBS = 20


def yes_probability(top_logprobs):
    yes_mass = no_mass = 0.0
    for entry in top_logprobs:
        token = entry.token.strip().strip('."“”').lower()
        probability = math.exp(entry.logprob)
        if token == "yes":
            yes_mass += probability
        elif token == "no":
            no_mass += probability
    total = yes_mass + no_mass
    if total == 0:
        return 0.5
    return yes_mass / total


class JudgeClient:
    def __init__(self, endpoint, model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                 concurrency=64):
        self.model = model
        self.client = AsyncOpenAI(base_url=endpoint, api_key="EMPTY",
                                  timeout=300, max_retries=3)
        self.sem = asyncio.Semaphore(concurrency)
        self.calls = 0
        self.failures = 0

    async def _score_one(self, prompt_text, audio_parts, candidate):
        # candidate: str -> outcome template (complete trajectory);
        # (context, latest_step) tuple -> step-level PRM template (signed off
        # 2026-09-03: full prefix shown as context, verdict on the latest step only)
        if isinstance(candidate, tuple):
            context, latest = candidate
            prompt = JUDGE_PROMPT_TEMPLATE_STEP.format(
                question=prompt_text,
                context=context or "(none — this is the first step)",
                latest_step=latest)
        else:
            prompt = JUDGE_PROMPT_TEMPLATE.format(question=prompt_text, candidate=candidate)
        # text part first, audio after — mirrors qwen_omni_orm._build_messages
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, *audio_parts]}]
        async with self.sem:
            self.calls += 1
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model, temperature=1.0, max_tokens=1,
                    logprobs=True, top_logprobs=TOP_LOGPROBS, messages=messages)
            except Exception:
                self.failures += 1
                return None
        return yes_probability(completion.choices[0].logprobs.content[0].top_logprobs)

    def scorer_for(self, rec):
        """Bind (audio, question+options) once per item; returns
        async scorer: list[str] -> list[float | None]."""
        base = f"Question: {rec.question}\n\nOptions:\n{format_choices(rec.choices)}"
        # realpath: vLLM resolves symlinks BEFORE the allowed-media-path check
        audio_parts = [
            {"type": "audio_url", "audio_url": {"url": "file://" + os.path.realpath(p)}}
            for p in rec.audio_paths
        ]

        async def score(candidates):
            return list(await asyncio.gather(
                *(self._score_one(base, audio_parts, c) for c in candidates)))

        return score

    async def close(self):
        await self.client.close()
