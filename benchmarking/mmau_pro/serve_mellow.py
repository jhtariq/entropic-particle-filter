"""OpenAI-compatible serving shim for the Mellow audio LM (Run 17).

Mellow (soham97/mellow; HTSAT encoder -> 389-token prefix -> SmolLM2-135M, 167M params)
is not supported by vLLM, so this shim exposes the minimal OpenAI surface the PF/EPF
harness needs, one process per GPU:

- ``GET /v1/models``       -> health check (lib.sh ``wait_healthy`` greps ``"id"``)
- ``POST /v1/chat/completions`` with
    * ``messages`` whose user turn carries audio content parts — both
      ``{"type": "audio_url", "audio_url": {"url": "file:///..."}}`` and
      ``{"type": "input_audio", "input_audio": {"data": <b64>, "format": "wav"}}``;
    * per-token ``logprobs`` and ``top_logprobs`` (needed for the entropy signal);
    * vLLM-style ``continue_final_message``/``add_generation_prompt`` (accepted at the
      top level and inside ``extra_body``): a trailing assistant message is tokenized
      without EOS and generation continues from it;
    * ``stop`` strings (with ``include_stop_str_in_output``), ``max_tokens``,
      ``temperature``, optional ``top_p``.

The chat messages are mapped onto Mellow's native format: all system/user text parts are
joined into one plain-text prompt (Mellow has no chat template) and the audio parts fill
its two audio slots. Generation re-implements ``MellowWrapper._generate_batch`` with a
KV cache and true temperature sampling (the reference loop is effectively greedy and
exposes no logprobs). Two intentional deviations from the reference, both required by
the harness: clips longer than the 10 s window get a deterministic FIRST-10s crop
(reference: random segment — would feed different audio to each PF step), and sampling
is real multinomial sampling at ``temperature`` (reference: top-p filter + argmax).

Everything above the engine (request parsing / message flattening / stop handling /
response building) is pure and torch-free so it is unit-testable offline.

Launch (one per GPU):

    CUDA_VISIBLE_DEVICES=0 python -m benchmarking.mmau_pro.serve_mellow \
        --port 8100 --model-name mellow --variant v0 \
        --code-dir /path/to/mellow_code \
        --allowed-media-root /home/exx/inference-time-scaling
"""

import base64
import binascii
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

MELLOW_REPO = "soham97/mellow"
MELLOW_REVISION = "83672db0dae28764e283210d5bb732621e903d8a"
MELLOW_CODE_COMMIT = "349f9b2be84bec713ac71e54fcbcac9bf4d116e5"
SMOLLM2_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MAX_AUDIO_SLOTS = 2


class RequestError(ValueError):
    """Invalid request -> HTTP 400 (non-retryable client-side)."""


# --------------------------------------------------------------------------------------
# pure layer: request parsing
# --------------------------------------------------------------------------------------


@dataclass
class AudioRef:
    kind: str  # "file" | "bytes"
    path: str | None = None
    data: bytes | None = None
    fmt: str | None = None


@dataclass
class ParsedRequest:
    prompt_text: str
    continuation: str
    audio_refs: list[AudioRef]
    stop: list[str]
    max_tokens: int | None
    temperature: float
    top_p: float | None
    logprobs: bool
    top_logprobs: int | None
    include_stop_str: bool
    raw_model: str = ""
    extras: dict = field(default_factory=dict)


def _flag(body: dict, extra: dict, name: str, default):
    if name in body:
        return body[name]
    if name in extra:
        return extra[name]
    return default


def flatten_messages(messages: list[dict]) -> tuple[str, list[AudioRef]]:
    """Join all text parts (message order) into Mellow's single plain-text prompt and
    collect the audio parts in order. Mellow has no chat template, so system and user
    text are simply concatenated with blank lines."""
    texts: list[str] = []
    audio_refs: list[AudioRef] = []
    for msg in messages:
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            if content:
                texts.append(content)
            continue
        if not isinstance(content, list):
            raise RequestError(f"message content must be a string or list, got {type(content).__name__}")
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                txt = part.get("text", "")
                if txt:
                    texts.append(txt)
            elif ptype == "audio_url":
                url = (part.get("audio_url") or {}).get("url", "")
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme != "file":
                    raise RequestError(f"audio_url must be a file:// URL, got {url!r}")
                path = urllib.request.url2pathname(parsed.path)
                audio_refs.append(AudioRef(kind="file", path=path))
            elif ptype == "input_audio":
                blob = part.get("input_audio") or {}
                try:
                    data = base64.b64decode(blob.get("data", ""), validate=True)
                except (binascii.Error, ValueError) as e:
                    raise RequestError(f"input_audio data is not valid base64: {e}") from e
                if not data:
                    raise RequestError("input_audio data is empty")
                audio_refs.append(AudioRef(kind="bytes", data=data, fmt=blob.get("format") or "wav"))
            else:
                raise RequestError(f"unsupported content part type {ptype!r}")
    return "\n\n".join(texts), audio_refs


def parse_request(body: dict, default_max_tokens: int = 512) -> ParsedRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    extra = body.get("extra_body") or {}

    continuation = ""
    head = messages
    if messages[-1].get("role") == "assistant":
        # the harness sets continue_final_message exactly when the last message is a
        # trailing assistant turn; honor the shape itself so gate 2 also passes when
        # the flag is stripped (e.g. an "openai"-typed endpoint)
        cont = messages[-1].get("content")
        if not isinstance(cont, str):
            raise RequestError("trailing assistant content must be a plain string to continue")
        continuation = cont
        head = messages[:-1]

    prompt_text, audio_refs = flatten_messages(head)
    if len(audio_refs) > MAX_AUDIO_SLOTS:
        raise RequestError(
            f"Mellow supports at most {MAX_AUDIO_SLOTS} audio clips per prompt, got {len(audio_refs)}"
        )

    stop_raw = body.get("stop")
    if stop_raw is None:
        stop = []
    elif isinstance(stop_raw, str):
        stop = [stop_raw]
    elif isinstance(stop_raw, list):
        stop = [s for s in stop_raw if isinstance(s, str) and s]
    else:
        raise RequestError("stop must be a string or list of strings")

    max_tokens = body.get("max_tokens")
    max_tokens = int(max_tokens) if max_tokens else default_max_tokens
    top_p = body.get("top_p")
    top_logprobs = body.get("top_logprobs")

    return ParsedRequest(
        prompt_text=prompt_text,
        continuation=continuation,
        audio_refs=audio_refs,
        stop=stop,
        max_tokens=max_tokens,
        temperature=float(body.get("temperature", 1.0)),
        top_p=float(top_p) if top_p is not None else None,
        logprobs=bool(body.get("logprobs", False)),
        top_logprobs=int(top_logprobs) if top_logprobs is not None else None,
        include_stop_str=bool(_flag(body, extra, "include_stop_str_in_output", False)),
        raw_model=str(body.get("model", "")),
    )


def resolve_audio_paths(audio_refs: list[AudioRef], allowed_roots: list[str]) -> None:
    """Reject file refs outside the allowed roots (vLLM's --allowed-local-media-path)."""
    roots = [os.path.realpath(r) for r in allowed_roots]
    for ref in audio_refs:
        if ref.kind != "file":
            continue
        real = os.path.realpath(ref.path)
        if not os.path.isfile(real):
            raise RequestError(f"audio file not found: {ref.path}")
        if not any(real == r or real.startswith(r.rstrip(os.sep) + os.sep) for r in roots):
            raise RequestError(f"audio path {ref.path} is outside the allowed media roots")
        ref.path = real


# --------------------------------------------------------------------------------------
# pure layer: stop strings, waveform fitting, response building
# --------------------------------------------------------------------------------------


def apply_stop_strings(
    text: str, cum_lengths: list[int], stops: list[str], include_stop: bool
) -> tuple[str, int, bool]:
    """Trim ``text`` at the earliest stop-string occurrence.

    ``cum_lengths[k]`` is ``len(decode(tokens[:k+1]))`` so the kept token count stays
    consistent with the kept text (stop strings may span token boundaries).
    Returns (kept_text, kept_token_count, hit).
    """
    best: tuple[int, str] | None = None
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, s)
    if best is None:
        return text, len(cum_lengths), False
    idx, s = best
    end = idx + (len(s) if include_stop else 0)
    kept = text[:end]
    n_tokens = len(cum_lengths)
    for k, ln in enumerate(cum_lengths):
        if ln >= end:
            n_tokens = k + 1
            break
    return kept, n_tokens, True


def fit_waveform(wav, need: int):
    """Fit a 1-D waveform to exactly ``need`` samples, mirroring the reference
    ``load_audio_into_tensor`` math: pad-by-repetition when short, crop when long —
    except the crop is the deterministic FIRST window (reference: random segment)."""
    import numpy as np

    n = wav.shape[0]
    if n == 0:
        raise RequestError("audio clip decoded to zero samples")
    if need >= n:
        reps = int(np.ceil(need / n))
        wav = np.tile(wav, reps)[:need]
    else:
        wav = wav[:need]
    return wav


def build_response_payload(model_name: str, created: int, result: dict) -> dict:
    """Assemble the chat-completion JSON from an engine result dict."""
    logprobs = None
    if result.get("logprob_entries") is not None:
        logprobs = {"content": result["logprob_entries"]}
    prompt_tokens = int(result.get("prompt_tokens", 0))
    completion_tokens = int(result.get("completion_tokens", 0))
    return {
        "id": f"chatcmpl-{created}-{abs(hash(result['content'])) % 10**8}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["content"]},
                "logprobs": logprobs,
                "finish_reason": result.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# --------------------------------------------------------------------------------------
# engine (torch / mellow imports stay inside methods)
# --------------------------------------------------------------------------------------


class MellowEngine:
    """Loads the pinned Mellow checkpoint and serves per-request generation.

    Generation is a KV-cache rewrite of ``MellowWrapper._generate_batch``; greedy
    (temperature 0) output is validated token-identical against the reference loop in
    ``run17/validate_shim.py``.
    """

    def __init__(
        self,
        code_dir: str,
        variant: str = "v0",
        revision: str = MELLOW_REVISION,
        smollm2_revision: str | None = SMOLLM2_REVISION,
        device: str = "cuda:0",
        max_text_tokens: int = 0,
        single_audio_fill: str = "silence",
        use_kv_cache: bool = True,
        waveform_cache_size: int = 64,
        prefix_cache_size: int = 0,
    ):
        self.code_dir = code_dir
        self.variant = variant
        self.revision = revision
        self.smollm2_revision = smollm2_revision
        self.device = device
        self.max_text_tokens = max_text_tokens  # 0 = faithful: yaml length (129) + '!' pads
        self.single_audio_fill = single_audio_fill  # silence | duplicate
        self.use_kv_cache = use_kv_cache
        self._wav_cache: dict = {}
        self._wav_cache_order: list = []
        self._wav_cache_size = waveform_cache_size
        # Audio-prefix cache (off unless --prefix-cache > 0). PF/EPF sends `budget`
        # particles per item, all sharing the SAME audio and prompt, so the HTSAT
        # encode and the 389-token prefill are recomputed identically B times per
        # step: measured 13.6 ms encode + 15.0 ms prefill of a ~73 ms request, i.e.
        # ~39% pure duplication (~1.6x). Both are deterministic under inference_mode,
        # so replaying them is bit-identical -- see run17/validate_shim.py parity.
        self._pfx_cache: dict = {}
        self._pfx_cache_order: list = []
        self._pfx_cache_size = prefix_cache_size
        self.model = None

    # -- loading -----------------------------------------------------------------

    def load(self) -> None:
        import sys

        import torch
        import yaml
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoTokenizer

        if self.code_dir not in sys.path:
            sys.path.insert(0, self.code_dir)
        from mellow.model.model import get_model_class

        cfg_path = os.path.join(self.code_dir, "mellow", "config", "v0.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        text_decoder = cfg["model"]["decoder"]["text_decoder"]
        if self.smollm2_revision:
            # pin the SmolLM2 base (architecture + tokenizer; weights are overwritten
            # by the Mellow ckpt). The snapshot path still contains "smollm2" so the
            # decoder's name-based branches keep working.
            text_decoder = snapshot_download(text_decoder, revision=self.smollm2_revision)

        model_cls = get_model_class(model_type=cfg["model"]["model_type"])
        model = model_cls(
            audioenc_name=cfg["model"]["encoder"]["audioenc_name"],
            d_in=cfg["model"]["encoder"]["out_emb"],
            text_decoder=text_decoder,
            prefix_length=cfg["model"]["decoder"]["prefix_length"],
            d_out=cfg["model"]["encoder"]["d_proj"],
        )
        ckpt_path = hf_hub_download(MELLOW_REPO, f"{self.variant}.ckpt", revision=self.revision)
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            model.load_state_dict(state)
        model.eval().to(self.device)
        self.model = model

        tokenizer = AutoTokenizer.from_pretrained(text_decoder)
        tokenizer.add_special_tokens({"pad_token": "!"})
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.encode("<|endoftext|>")[0]

        self.sample_rate = cfg["data"]["sampling_rate"]
        self.segment_samples = cfg["data"]["segment_seconds"] * self.sample_rate
        self.text_len = cfg["data"]["text_tokenization_len"]

    # -- audio -------------------------------------------------------------------

    def _decode_audio(self, ref: AudioRef):
        """Load + resample to the model rate; return a 1-D float32 numpy array."""
        import io

        import numpy as np
        import torch
        import torchaudio

        source = ref.path if ref.kind == "file" else io.BytesIO(ref.data)
        wav = None
        sr = None
        try:
            wav_t, sr = torchaudio.load(source)
            wav = wav_t.numpy()
        except Exception:
            try:
                import soundfile as sf

                if ref.kind == "bytes":
                    source = io.BytesIO(ref.data)
                data, sr = sf.read(source, dtype="float32", always_2d=True)
                wav = data.T  # (channels, samples)
            except Exception:
                import librosa

                if ref.kind == "bytes":
                    source = io.BytesIO(ref.data)
                data, sr = librosa.load(source, sr=None, mono=False)
                wav = data if data.ndim == 2 else data[None, :]
        if wav is None or wav.size == 0:
            raise RequestError("could not decode audio clip")
        wav_t = torch.from_numpy(np.ascontiguousarray(wav)).float()
        if sr != self.sample_rate:
            wav_t = torchaudio.functional.resample(wav_t, sr, self.sample_rate)
        # reference flattens all channels end-to-end (reshape(-1)), keep identical
        return wav_t.reshape(-1).numpy()

    def _waveform(self, ref: AudioRef):
        """Fitted (exactly segment-length) waveform with a small LRU cache — PF re-sends
        the same clips at every step."""
        import hashlib

        if ref.kind == "file":
            st = os.stat(ref.path)
            key = (ref.path, st.st_mtime_ns, st.st_size)
        else:
            key = ("b64", hashlib.sha1(ref.data).hexdigest())
        if key in self._wav_cache:
            return self._wav_cache[key]
        wav = fit_waveform(self._decode_audio(ref), self.segment_samples)
        self._wav_cache[key] = wav
        self._wav_cache_order.append(key)
        if len(self._wav_cache_order) > self._wav_cache_size:
            old = self._wav_cache_order.pop(0)
            self._wav_cache.pop(old, None)
        return wav

    def _ref_key(self, ref: AudioRef):
        """Stable identity for one audio clip (same scheme as the waveform cache)."""
        import hashlib

        if ref.kind == "file":
            st = os.stat(ref.path)
            return (ref.path, st.st_mtime_ns, st.st_size)
        return ("b64", hashlib.sha1(ref.data).hexdigest())

    def _clone_past(self, past):
        """Deep-copy a KV cache so the cached prefill is never mutated by decoding.

        Copies tensors rather than sharing them: generation appends to the cache in
        place, so a shared object would corrupt the next particle's prefix.
        """
        import copy

        return copy.deepcopy(past)

    def _prefix_entry(self, req, model, lm, input_ids):
        """(prefix_embeds, prefill_past, prefill_logits) for the audio+prompt prefix.

        Cached per (audio clips, prompt) when --prefix-cache > 0; the returned KV is
        always a private clone, so callers may decode into it freely.
        """
        import torch

        key = (tuple(self._ref_key(r) for r in req.audio_refs),
               self.single_audio_fill, req.prompt_text, self.max_text_tokens)
        hit = self._pfx_cache.get(key) if self._pfx_cache_size else None
        if hit is None:
            a1, a2 = self._audio_slots(req.audio_refs)
            d = {
                "audio1": torch.from_numpy(a1).unsqueeze(0).to(self.device),
                "audio2": torch.from_numpy(a2).unsqueeze(0).to(self.device),
                "input": {"input_ids": input_ids.to(self.device)},
            }
            prefix, _, _ = model.generate_prefix_inference(d)
            out = lm(inputs_embeds=prefix, use_cache=self.use_kv_cache)
            past = out.past_key_values if self.use_kv_cache else None
            logits = out.logits[0, -1, :]
            if self._pfx_cache_size:
                self._pfx_cache[key] = (prefix, past, logits)
                self._pfx_cache_order.append(key)
                if len(self._pfx_cache_order) > self._pfx_cache_size:
                    self._pfx_cache.pop(self._pfx_cache_order.pop(0), None)
                hit = self._pfx_cache[key]
            else:
                return prefix, past, logits
        prefix, past, logits = hit
        return prefix, (self._clone_past(past) if past is not None else None), logits

    def _audio_slots(self, audio_refs: list[AudioRef]):
        import numpy as np

        wavs = [self._waveform(r) for r in audio_refs]
        silence = np.zeros(self.segment_samples, dtype="float32")
        if len(wavs) == 0:
            a1, a2 = silence, silence
        elif len(wavs) == 1:
            a1 = wavs[0]
            a2 = wavs[0] if self.single_audio_fill == "duplicate" else silence
        else:
            a1, a2 = wavs
        return a1, a2

    # -- text --------------------------------------------------------------------

    def _tokenize_prompt(self, text: str):
        if self.max_text_tokens and self.max_text_tokens > 0:
            # extended mode: no '!' padding, longer window
            return self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_text_tokens,
                return_tensors="pt",
            )["input_ids"]
        # faithful mode: exactly the reference preprocess_text (pads are attended)
        return self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.text_len,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]

    # -- generation --------------------------------------------------------------

    def _sample_step(self, logits, temperature: float, top_p: float | None, want_lp: bool, top_k: int):
        """One sampling step. Returns (token_id, logprob, top_entries|None)."""
        import torch
        from torch.nn import functional as nnf

        logits = logits.float()
        if temperature <= 1e-5:
            dist = nnf.log_softmax(logits, dim=-1)
            tok = int(torch.argmax(logits, dim=-1).item())
        else:
            scaled = logits / temperature
            dist = nnf.log_softmax(scaled, dim=-1)  # logprobs pre-top-p (vLLM convention)
            if top_p is not None and 0.0 < top_p < 1.0:
                probs = dist.exp()
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = (cum - sorted_probs) > top_p  # keep the first token crossing top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum()
                pick = int(torch.multinomial(sorted_probs, 1).item())
                tok = int(sorted_idx[pick].item())
            else:
                tok = int(torch.multinomial(dist.exp(), 1).item())

        if not want_lp:
            return tok, None, None
        chosen_lp = float(dist[tok].item())
        top_entries = None
        if top_k:
            k = min(top_k, dist.shape[-1])
            vals, idxs = torch.topk(dist, k)
            top_entries = [
                {"token": self.tokenizer.decode([int(i)]), "logprob": float(v)}
                for v, i in zip(vals.tolist(), idxs.tolist(), strict=True)
            ]
            if tok not in idxs.tolist():
                top_entries.append({"token": self.tokenizer.decode([tok]), "logprob": chosen_lp})
        return tok, chosen_lp, top_entries

    def generate(self, req: ParsedRequest) -> dict:
        import torch

        model, tokenizer = self.model, self.tokenizer
        lm = model.caption_decoder.lm

        input_ids = self._tokenize_prompt(req.prompt_text)

        with torch.inference_mode():
            # audio prefix + its prefill (cached across an item's particles when
            # --prefix-cache > 0; identical work otherwise)
            prefix, past, logits = self._prefix_entry(req, model, lm, input_ids)

            cont_ids: list[int] = []
            if req.continuation:
                cont_ids = tokenizer(req.continuation, add_special_tokens=False)["input_ids"]

            prompt_tokens = int(prefix.shape[1]) + len(cont_ids)
            want_lp = req.logprobs
            top_k = req.top_logprobs or 0

            tokens: list[int] = []
            entries: list[dict] = []
            cum_lengths: list[int] = []
            text = ""
            finish_reason = "length"

            generated = None
            if cont_ids:
                cont_t = torch.tensor([cont_ids], dtype=torch.long, device=self.device)
                cont_embed = lm.model.embed_tokens(cont_t)
                if self.use_kv_cache:
                    # extend the (cached) prefix KV with the continuation
                    out = lm(inputs_embeds=cont_embed, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[0, -1, :]
                else:
                    generated = torch.cat((prefix, cont_embed), dim=1)
                    out = lm(inputs_embeds=generated)
                    logits = out.logits[0, -1, :]
            elif not self.use_kv_cache:
                generated = prefix
                logits = lm(inputs_embeds=generated).logits[0, -1, :]

            for _ in range(req.max_tokens):
                tok, lp, top = self._sample_step(logits, req.temperature, req.top_p, want_lp, top_k)
                if tok == self.eos_id:
                    finish_reason = "stop"
                    break
                tokens.append(tok)
                if want_lp:
                    entry = {"token": tokenizer.decode([tok]), "logprob": lp}
                    if top is not None:
                        entry["top_logprobs"] = top
                    entries.append(entry)
                text = tokenizer.decode(tokens)
                cum_lengths.append(len(text))
                if req.stop:
                    kept, n_tokens, hit = apply_stop_strings(
                        text, cum_lengths, req.stop, req.include_stop_str
                    )
                    if hit:
                        text = kept
                        tokens = tokens[:n_tokens]
                        entries = entries[:n_tokens]
                        finish_reason = "stop"
                        break
                tok_t = torch.tensor([[tok]], dtype=torch.long, device=self.device)
                tok_embed = lm.model.embed_tokens(tok_t)
                if self.use_kv_cache:
                    out = lm(inputs_embeds=tok_embed, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                else:
                    generated = torch.cat((generated, tok_embed), dim=1)
                    out = lm(inputs_embeds=generated)
                logits = out.logits[0, -1, :]

        return {
            "content": text,
            "logprob_entries": entries if want_lp else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "finish_reason": finish_reason,
        }


# --------------------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------------------


def create_app(engine, model_name: str, allowed_roots: list[str], default_max_tokens: int):
    import asyncio
    import logging
    import time
    import traceback

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()
    lock = asyncio.Lock()
    logger = logging.getLogger("serve_mellow")

    def _error(status: int, message: str):
        return JSONResponse(status_code=status, content={"error": {"message": message}})

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mellow",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _error(400, "request body is not valid JSON")
        try:
            parsed = parse_request(body, default_max_tokens=default_max_tokens)
            resolve_audio_paths(parsed.audio_refs, allowed_roots)
        except RequestError as e:
            return _error(400, str(e))
        try:
            # serialize GPU work; the event loop stays free for /v1/models health checks
            async with lock:
                result = await asyncio.to_thread(engine.generate, parsed)
        except RequestError as e:
            return _error(400, str(e))
        except Exception as e:
            logger.error("generation failed: %s\n%s", e, traceback.format_exc())
            return _error(500, f"generation failed: {e}")
        return build_response_payload(model_name, int(time.time()), result)

    return app


def main():
    import click

    @click.command()
    @click.option("--host", default="0.0.0.0")
    @click.option("--port", default=8100, type=int)
    @click.option("--model-name", default="mellow")
    @click.option("--variant", type=click.Choice(["v0", "v0_s"]), default="v0")
    @click.option("--revision", default=MELLOW_REVISION)
    @click.option("--smollm2-revision", default=SMOLLM2_REVISION)
    @click.option("--code-dir", required=True, help="pinned clone of github.com/soham97/mellow")
    @click.option("--device", default="cuda:0")
    @click.option("--max-text-tokens", default=0, type=int,
                  help="0 = faithful (yaml 129 + '!' pads); N>0 = extended (truncate at N, no pads)")
    @click.option("--single-audio-fill", type=click.Choice(["silence", "duplicate"]), default="silence")
    @click.option("--allowed-media-root", multiple=True,
                  default=("/home/exx/inference-time-scaling",))
    @click.option("--default-max-tokens", default=512, type=int)
    @click.option("--no-kv-cache", is_flag=True, default=False,
                  help="full re-forward per token (reference behavior; slow fallback)")
    @click.option("--waveform-cache", default=64, type=int,
                  help="LRU size for fitted waveforms (~1.3 MB each; raise when the item "
                       "working set is small, e.g. subset test runs)")
    @click.option("--prefix-cache", default=0, type=int,
                  help="LRU size for cached audio-prefix encodes + their prefill KV, keyed by "
                       "(audio, prompt). PF/EPF sends `budget` particles per item that share "
                       "both, so this removes ~39%% of per-request work (~1.6x). Deterministic "
                       "and bit-identical; 0 = OFF (the run17/MMAR reference behaviour)")
    @click.option("--pid-file", default=None,
                  help="write this process's PID here at startup (setsid-safe for kill-by-PID)")
    def cli(host, port, model_name, variant, revision, smollm2_revision, code_dir, device,
            max_text_tokens, single_audio_fill, allowed_media_root, default_max_tokens,
            no_kv_cache, waveform_cache, prefix_cache, pid_file):
        import uvicorn

        if pid_file:
            with open(pid_file, "w") as f:
                f.write(f"{os.getpid()}\n")

        engine = MellowEngine(
            code_dir=code_dir,
            variant=variant,
            revision=revision,
            smollm2_revision=smollm2_revision or None,
            device=device,
            max_text_tokens=max_text_tokens,
            single_audio_fill=single_audio_fill,
            use_kv_cache=not no_kv_cache,
            waveform_cache_size=waveform_cache,
            prefix_cache_size=prefix_cache,
        )
        print(f"[serve_mellow] loading {MELLOW_REPO}@{revision[:8]} variant={variant} on {device} ...")
        engine.load()
        print(f"[serve_mellow] ready on :{port} (model-name={model_name})")
        app = create_app(engine, model_name, list(allowed_media_root), default_max_tokens)
        uvicorn.run(app, host=host, port=port, log_level="warning")

    cli()


if __name__ == "__main__":
    main()
