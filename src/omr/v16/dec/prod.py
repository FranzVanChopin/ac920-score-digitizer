"""The ONE production configuration.  Every script imports it from here.

Nine scripts had each grown their own `CFG = dict(...)` and several had gone
stale -- `cost.py` and `diag_where.py` still carried `lam_iso=12` (measured to
destroy voice matching), `decompose.py` still carried `voice_w=1.5` (likewise),
and the rest carried pre-trust-head tuning weights.  Every diagnostic was
therefore run under a different evaluator, and none of them were comparable.

Rule: do not write evaluator weights anywhere else.  Import PROD.

TRAINING ACCELERATION (offline only)
------------------------------------
The deployed decoder stays pure numpy on the A53 -- that boundary is unchanged.
But offline training may use the GPU, because it changes nothing about the
artefact that ships:

    # system Python 3.14 has torch 2.9.1+cu128 and sees the RTX 5080
    python src/torch_train.py --epochs 30 --hidden 192 --aug 8 --state-noise 0.05

That trains the identical model ~11x faster (172 s vs ~2000 s) and writes into
the SAME cache slot `run_experiment.get_model` uses, so every numpy script picks
the GPU-trained weights up transparently.  `torch_train.py --verify` checks the
torch forward against the numpy one (max abs difference ~3e-5) before any run is
trusted.
"""
from __future__ import annotations

# Search geometry.
BEAM = 16            # cost is controlled by beam width, never by truncation:
                     # cutting expansions mid-flight is a cliff, not a taper
K_ATTR = 5          # options per candidate inside the BEAM
DP_K_ATTR = 40      # options per event for the exact DP (see run_experiment)
MAX_EXPANSIONS = 4000   # safety net only; beam=16 peaks near 550, so never hit
# Column-consistent joint value DP (src/column_joint.py), applied only when the final
# answer breaks the exact x-column invariant.  Measured on the full 5 folds, same models:
# perfect 95.81 -> 96.11 (+0.31, t=2.80, 4/5 folds up, 1 tied), any-field 1.80 -> 1.26,
# voice unchanged, 14 measures fixed and 0 broken out of 93 triggered.
COLUMN_JOINT = True
# States kept per step by the column-joint DP.  On all 93 triggered measures the answers
# at 1,000 are IDENTICAL to 4,000 (and to 30,000); 500 changes 3 of them.  1,000 cuts the
# stage's peak heap 68.6 -> 19.5 MiB and its time ~40%, so the headline does not move.
COLUMN_JOINT_MAX_STATES = 1000
# Search-time staff guard (search.solve staff_guard): a voice under construction only
# accepts candidates on its own staff.  Exact for the ground truth (0 of 6,729 voices
# cross staves), but measured on the full 5 folds at 96.38 it is WORSE than the terminal
# staff_w term: guard alone +0.10 (t=1.05, fold 3 -0.21, any-field +0.07); guard + staff_w
# +0.13 (t=1.44, same fold 3 loss); staff_w alone +0.12 (t=3.32, no fold down, any-field
# -0.05).  Pruning in the search costs some measures, as order_guard did.  Kept OFF.
STAFF_GUARD = False
# Intra-piece repetition consensus (src/repeat_consensus.py): twins of one piece whose
# answers disagree share the answer with the lowest summed cost, unless a twin's own
# evidence prefers its answer by >= REPEAT_VETO nats.  Full 5-fold CV at 96.50:
# veto 4 -> 97.04 (+0.54, no fold down, 25 fixed / 0 broken); veto 8 -> 97.02 (one fold
# -0.17); no veto -> 97.05 (two folds down).  The veto was chosen on the same CV.
# Simulator caveat: repeats get independent hint noise here; real misreads may correlate.
REPEAT_CONSENSUS = True
REPEAT_VETO = 4.0
# Engraving-spacing term (column_joint.spacing_residuals / column_joint_refine spacing=).
# ALPHA is FITTED ON REAL ENGRAVING -- PDMX MusicXML default-x, 300 single-part pieces
# outside the benchmark, 159,687 column gaps (pdmx_spacing_fit.py): alpha 0.53 (the
# simulator's synthetic layout uses 0.60).  C and TRIGGER are set from that residual
# distribution, not tuned: C ~ its core width (q05/q95 = -0.09/+0.13), TRIGGER = 0.3 sits
# between q95 and q99 (+0.42).  SPACING_W = 0 leaves production unchanged.
# Measured (report Round 39A): helps on SIMULATOR x (frozen CV w=2 +0.14, PDMX sim-x w=2
# +0.28, ~0 broken) but not on REAL layout (PDMX real-x w=1 +0.06 [-0.08, +0.20],
# w=2 -0.18), where it fires 8x as often and rewrites correct answers.  KEPT OFF.
SPACING_W = 0.0
SPACING_ALPHA = 0.53
SPACING_C = 0.1
SPACING_TRIGGER = 0.3

# Evaluator weights.  Selected by src/tune_eval.py on ALL dev polyphonic
# measures (85 grid points, one test report at the end).  The earlier 200-measure
# dev subset gave a verdict that did not transfer, so the whole dev set is used.
#
# Verified on the frozen test set (650 measures) -- every metric improved:
#              all-perfect  all-any  poly-perfect  poly-any  poly-voice
#   previous      88.31      5.97      78.87       6.71      83.93
#   this          88.46      5.53      79.17       6.20      84.82
# `hint_w` prefers not to rewrite the upstream hint; it only pays off together
# with the trust head, which knows when the hint is worth rewriting.
# `stem_w` charges each voice that shares a staff with another voice yet mixes up- and
# down-stems (search._stem_term); 0 leaves the beam's choice unchanged.  Full 5-fold CV
# at the 96.11 working point: 0.5 -> 96.33, 1.0 -> 96.38 (t=3.28), 2.0 -> 96.38 with a
# worse any-field, so 1.0 (the three were compared on the same CV: mild selection bias).
# `staff_w` charges each voice whose events sit on more than one staff (search._staff_term);
# ground-truth voices never do.  Full 5-fold CV at the 96.38 working point: 0.5, 1, 2 and
# 5 all give 96.50 (+0.12, t=3.32), so the weight is not tuned -- 1.0 is just a value.
EVAL = dict(tau_w=1.0, voice_w=0.0, dev_w=1.5, iso_w=3.0, hint_w=0.5, stem_w=1.0,
            staff_w=1.0)

# lam_iso exposes isochrony to the beam PRIORITY.  Measured harmful (voice match
# 84% -> 64%); keep at 0 and leave isochrony as a terminal loss term only.
LAM_ISO = 0.0

PROD = dict(beam=BEAM, k_attr=K_ATTR, lam_iso=LAM_ISO, warp_top=3,
            max_expansions=MAX_EXPANSIONS, **EVAL)

