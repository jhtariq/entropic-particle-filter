# Policy: Kimi-Audio-7B-Instruct. MUST be served via the run19 wrapper
# (serve_policy.sh dispatches to benchmarking/mmar/scripts/serve_kimi.sh);
# plain `vllm serve` is broken 6 ways for Kimi on vLLM 0.22.1.
# le30s subset + max_model_len 8192 (= config.json max_position_embeddings),
# single audio per prompt (fine: MMAR is single-audio and judge prompts carry
# one clip). Matches MMAR Run 6 except GPU/port (policy slot :8100 here).
MODEL_ID="moonshotai/Kimi-Audio-7B-Instruct"
MODEL_REV="9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
SERVED_NAME="kimi-audio"
SERVE_KIND=kimi
SUBSET=le30s
MAX_MODEL_LEN=8192
GPUS="0"
