"""vLLM serve wrapper for Kimi-Audio (Run 19) — REQUIRED, never plain `vllm serve`.

vLLM 0.22.1 registers MoonshotKimiaForCausalLM natively, but its chat path is
broken three ways for this model (verified in source + in-memory repro):

  1. No chat template is auto-wired (the chat-template registry has no kimi
     entry and the checkpoint ships none) -> every /v1/chat/completions request
     would 400. Serving must pass --chat-template (config.sh KIMI_CHAT_TEMPLATE,
     the corrected template_kimi_audio_epf.jinja next to this file).
  2. KimiAudioTokenizer.apply_chat_template passes the conversation UNWRAPPED to
     transformers' render_jinja_template, which iterates per-conversation -> each
     *message dict* is rendered as a conversation and every prompt comes out
     EMPTY (the bundled template reads `conversations[0]`, undefined at render
     time).
  3. continue_final_message=True dies with KeyError(-1) on the same unwrapped
     call (`chat[-1]` on a message dict) -> phase-0 gate 2 and every EPF
     continuation step would 400.
  4. Concurrent audio requests KILL THE ENGINE (EngineDeadError; observed live
     at screen concurrency 6, twice). When >=2 audio requests land in one
     engine step, (a) KimiAudioForConditionalGeneration._process_audio_input
     gets a LIST of per-item feature tensors where it assumes a stacked tensor
     (AttributeError: 'list' has no .dim), and (b) even normalized, the
     Whisper-encoder wrapper torch.stack()s per-item outputs, which crashes
     whenever the clips differ in length (the feature extractor pads each clip
     to its own length): "stack expects each tensor to be equal size". Only the
     batch-of-ONE path is actually correct in stock 0.22.1.
  5. Model-GENERATED text 400s the next request. Kimi-Audio likes to end
     answers with the literal text "[EOS]"; tiktoken's encode (default
     disallowed_special="all") refuses to encode any special-token text it
     finds in a prompt, so the EPF step loop — which feeds the generated text
     back as the assistant prefill — dies with "disallowed special token
     '[EOS]'" (29/32 P2 smoke rows before the fix).
  6. top_logprobs KILLS THE SERVER. The LM head covers the AUDIO-token vocab
     (ids >= kimia_token_offset 152064); under temp-0.8 sampling an audio id
     eventually cracks the top-20, vLLM decodes every top-k candidate id
     individually, tiktoken raises KeyError('Invalid token for decoding: …'),
     and the AsyncLLM output handler dies -> every later request 500s and the
     server shuts down. Stochastic: survived gates/screens/smoke, then killed
     BOTH replicas minutes into the b1 stage.

Fixes: (1-3) wrap the conversation (`[conversation]`) before
render_jinja_template and unwrap the single rendered prompt; (4) patch
embed_multimodal to run the known-good batch-of-one encoder+projector path once
per item and return the per-item embedding list directly (placeholder counts
are per-item too, so lengths line up by construction; the tiny encoder loses
only cross-item batching); (5) patch encode with disallowed_special=() so
special-token-looking TEXT encodes as plain text (the 6 kimi chat markers stay
allowed as real specials); (6) patch decode with a per-token fallback that
renders undecodable (audio-range) ids as "" — EPF consumes the top-logprob
NUMBERS, the candidate token strings are cosmetic. Chat rendering happens in
the API-server frontend
process and the EngineCore is forked from it (VLLM_WORKER_MULTIPROC_METHOD
defaults to "fork"), so patching these classes here — before the CLI starts —
covers both processes. Everything else is delegated verbatim to the stock
`vllm serve` CLI:

    python run19/serve_kimi.py <model> --revision <sha> --chat-template \
        run19/template_kimi_audio_epf.jinja ... (all flags as in lib.sh)
"""

import os
import sys

import torch
from transformers.utils import chat_template_utils as hf_chat_utils

from vllm.model_executor.models.kimi_audio import KimiAudioForConditionalGeneration
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer


def _patched_apply_chat_template(
    self,
    messages=None,
    tools=None,
    chat_template=None,
    tokenize=False,
    **kwargs,
):
    conversation = messages if messages is not None else kwargs.pop("conversation", None)
    if conversation is None:
        raise ValueError("Either 'messages' or 'conversation' must be provided.")
    template = self.get_chat_template(chat_template, tools=tools)
    if template is None:
        raise ValueError("No chat template available. Provide `chat_template` explicitly.")
    # render_jinja_template takes a LIST of conversations; the stock 0.22.1 method
    # passes the conversation bare, which is the whole bug this wrapper exists for
    rendered, _ = hf_chat_utils.render_jinja_template(
        [conversation], chat_template=template, tools=tools, **kwargs
    )
    prompt = rendered[0] if rendered else ""
    if tokenize:
        return self.encode(prompt, add_special_tokens=False)
    return prompt


KimiAudioTokenizer.apply_chat_template = _patched_apply_chat_template


def _embed_one_audio(self, feat2d):
    # the known-good batch-of-one path: (128, T_mel) -> (T_mel/2/4, llm_dim).
    # Mirrors the stock _process_audio_input maths exactly, for a single item.
    audio_features = self.audio_tower([feat2d])  # (1, T, D_enc)
    _, T, D = audio_features.shape
    if T % 4 != 0:
        audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, 4 - (T % 4)))
        T = audio_features.shape[1]
    audio_features = audio_features.reshape(1, T // 4, D * 4)
    return self.multi_modal_projector(audio_features)[0]  # (T/4, llm_dim)


def _patched_embed_multimodal(self, **kwargs):
    audio_input = self._parse_and_validate_audio_input(**kwargs)
    if audio_input is None:
        return []
    feats = audio_input["whisper_input_features"]
    # concurrent audio requests batched into one engine step arrive as a LIST of
    # per-item tensors with per-clip lengths; the stock path crashes on the list
    # AND on stacking unequal encoder outputs — process per item instead
    if isinstance(feats, (list, tuple)):
        flat = [f.squeeze(0) if f.dim() == 3 else f for f in feats]
    elif feats.dim() == 3:
        flat = list(feats.unbind(dim=0))
    else:
        flat = [feats]
    return [_embed_one_audio(self, f) for f in flat]


KimiAudioForConditionalGeneration.embed_multimodal = _patched_embed_multimodal


_KIMI_ALLOWED_SPECIAL = {
    "<|im_media_begin|>",
    "<|im_media_end|>",
    "<|im_kimia_text_blank|>",
    "<|im_msg_end|>",
    "<|im_kimia_user_msg_start|>",
    "<|im_kimia_assistant_msg_start|>",
}


def _patched_encode(
    self, text, truncation=None, max_length=None, add_special_tokens=True, **kwargs
):
    del add_special_tokens
    # disallowed_special=(): model-generated text (e.g. a literal "[EOS]") fed
    # back as an assistant prefill must encode as PLAIN TEXT, never raise; the
    # 6 chat markers remain real special tokens
    tokens = self._tokenizer.encode(
        text, allowed_special=_KIMI_ALLOWED_SPECIAL, disallowed_special=()
    )
    if truncation:
        tokens = self._maybe_truncate(tokens, max_length)
    return tokens


KimiAudioTokenizer.encode = _patched_encode


_orig_decode = KimiAudioTokenizer.decode


def _patched_decode(self, ids, skip_special_tokens=False):
    # audio-vocab ids (>= kimia_token_offset) have no BPE bytes; stock decode
    # raises KeyError and takes the whole AsyncLLM output handler down with it.
    # Fall back per token and render such ids as "" — never raise.
    if isinstance(ids, int):
        ids = [ids]
    try:
        return _orig_decode(self, ids, skip_special_tokens)
    except KeyError:
        parts = []
        for i in ids:
            try:
                parts.append(_orig_decode(self, [i], skip_special_tokens))
            except KeyError:
                parts.append("")
        return "".join(parts)


KimiAudioTokenizer.decode = _patched_decode


if __name__ == "__main__":
    # --pid-file: written by THIS process (run17 lesson — `nohup setsid ... &`
    # may fork, leaving the launcher's $! stale; the recorded PID must be the
    # server's own so kill_servers kills the real thing)
    if "--pid-file" in sys.argv:
        i = sys.argv.index("--pid-file")
        pid_file = sys.argv[i + 1]
        del sys.argv[i : i + 2]
        with open(pid_file, "w") as f:
            f.write(f"{os.getpid()}\n")

    from vllm.entrypoints.cli.main import main

    sys.argv = [sys.argv[0], "serve", *sys.argv[1:]]
    main()
