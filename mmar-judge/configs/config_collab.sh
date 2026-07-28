# Collaborator budget grid (larger GPUs / more time). WARNING on beam cost:
# each beam level judges ~budget partial chains WITH AUDIO, so beam b=128 with
# ~6 levels is ~770 judge calls PER ITEM (~765K for the full 996-item set).
# Run beam <= 32 unless judge capacity is scaled (multiple judge replicas), or
# shard with --limit/--select.
BUDGETS_BON="1 4 8 16 32 64 128"
BUDGETS_BEAM="8 16 32 64 128"
BEAM_WIDTH=4
SMOKE_N=100
MAX_INFLIGHT=8
POLICY_INFLIGHT=32
JUDGE_INFLIGHT=16
