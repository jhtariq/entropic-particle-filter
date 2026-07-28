# Local (2x RTX PRO 6000) budget grid. Beam b=4 is deliberately absent: with
# beam_width=4 it degenerates to a single non-branching chain (num_beams =
# budget // beam_width = 1); BoN b=1 is the shared greedy baseline instead.
BUDGETS_BON="1 4 8"
BUDGETS_BEAM="8"
BEAM_WIDTH=4
SMOKE_N=100
MAX_INFLIGHT=8        # concurrent items
POLICY_INFLIGHT=32    # concurrent policy requests per endpoint
JUDGE_INFLIGHT=16     # concurrent judge requests (global)
