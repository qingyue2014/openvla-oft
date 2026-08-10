#!/usr/bin/env bash
# Formal paper-configuration runner for the PhysCogSafe pilot matrix.
#
# Implements the statistical reporting protocol in RESULT_TABLE_DESIGN.md:
# initial states are generated ONCE with a fixed scene seed (paired Er/Ec
# episodes stay identical across repeats), then every selected family is
# evaluated once per policy seed in SEEDS. Runs are tagged
# `<run_id>-seed<N>`, which record_experiment_results.py maps to the same
# scenario and generate_result_tables.py pools automatically (Wilson CIs,
# run-level mean ± CI, Δ Safe SR with McNemar/z tests in Table 5).
#
# Usage (from the repository root, on a GPU node):
#   bash experiments/robot/libero/tasks/run_paper_matrix.sh prepare      # one-time scene generation + gates
#   bash experiments/robot/libero/tasks/run_paper_matrix.sh smoke        # quick 1-seed pass with SMOKE_TRIALS
#   bash experiments/robot/libero/tasks/run_paper_matrix.sh full         # SEEDS x NUM_TRIALS formal runs
#   bash experiments/robot/libero/tasks/run_paper_matrix.sh attribution  # SAR/UIR/OCR/NOR reports
#   bash experiments/robot/libero/tasks/run_paper_matrix.sh tables       # records + result tables
#
# Overridable environment:
#   SEEDS="42"            policy/env seeds; default first-pass protocol is one
#                         seed x NUM_TRIALS=50 per condition. Add seeds after CI
#                         review for selected scenarios. To run five repeats:
#                         SEEDS="42 43 44 45 46". To match LIBERO-Gen's 50 repeats:
#                         SEEDS="$(seq -s' ' 42 91)" (~10x compute, CI ~±2pp).
#   NUM_TRIALS=50         episodes per condition per seed (L2-C2 uses L2C2_TRIALS=20)
#   SCENE_SEED=42         fixed initial-state generation seed; do not vary per repeat
#   FAMILIES="l1a1 l1a2 l1b1 l1b2 l2b2 l2c2"   subset selection
#   CHECKPOINT=...        forwarded to the per-family runners
#
# Notes:
#   - `full` is resumable: L1-A families skip a (run_id, seed) whose completed
#     log already exists; rerunning other families overwrites with a newer
#     timestamp and the table generator keeps the latest run per seed.
#   - L1-A2 evaluation is gated (pairing + occlusion gates from L1-A2_SPEC.md);
#     `prepare` runs the gate checks once so `full` does not regenerate scenes.

set -euo pipefail

MODE="${1:-full}"

SEEDS="${SEEDS:-42}"
NUM_TRIALS="${NUM_TRIALS:-50}"
L2C2_TRIALS="${L2C2_TRIALS:-20}"
SMOKE_TRIALS="${SMOKE_TRIALS:-5}"
SCENE_SEED="${SCENE_SEED:-42}"
FAMILIES="${FAMILIES:-l1a1 l1a2 l1b1 l1b2 l2b2 l2c2 l3a1}"
L2B2_VARIANTS="${L2B2_VARIANTS:-basket basket_off basket_far}"
POOL_SINCE="${POOL_SINCE:-}"

TASKS_DIR="experiments/robot/libero/tasks"
LOG_DIR="${LOG_DIR:-experiments/logs}"

log() { echo; echo "════════════════════════════════════════════════"; echo "  [paper-matrix] $*"; echo "════════════════════════════════════════════════"; }

# ── Per-family runners ─────────────────────────────────────────────────────────
# Each function runs one family for one policy seed. Scene generation always
# uses SCENE_SEED; only EVAL_SEED varies across repeats.

run_family() {
    local family="$1" seed="$2" trials="$3" suffix="$4"
    case "${family}" in
        l1a1)
            SEED="${SCENE_SEED}" EVAL_SEED="${seed}" RUN_ID_SUFFIX="${suffix}" \
                NUM_TRIALS="${trials}" RUN_PREVIEW=False RECORD_RESULTS=False \
                bash "${TASKS_DIR}/run_l1a_evals.sh" l1a1_eval
            ;;
        l1a2)
            # L1-A2 shares task-1's native Eb competence gate with L1-A1. Run
            # it at the same formal seed/trial count instead of reusing pilot10.
            SEED="${SCENE_SEED}" EVAL_SEED="${seed}" RUN_ID_SUFFIX="${suffix}" \
                NUM_TRIALS="${trials}" RUN_PREVIEW=False RECORD_RESULTS=False \
                bash "${TASKS_DIR}/run_l1a_evals.sh" l1a_native_eb
            SEED="${SCENE_SEED}" EVAL_SEED="${seed}" RUN_ID_SUFFIX="${suffix}" \
                NUM_TRIALS="${trials}" RUN_PREVIEW=False RECORD_RESULTS=False \
                bash "${TASKS_DIR}/run_l1a_evals.sh" l1a2
            ;;
        l1b1)
            SCENE_SEED="${SCENE_SEED}" EVAL_SEED="${seed}" RUN_ID_SUFFIX="${suffix}" \
                NUM_TRIALS="${trials}" \
                bash "${TASKS_DIR}/run_l1b_swept.sh" l1b1_native_gripper eval
            ;;
        l1b2)
            SCENE_SEED="${SCENE_SEED}" EVAL_SEED="${seed}" RUN_ID_SUFFIX="${suffix}" \
                NUM_TRIALS="${trials}" \
                bash "${TASKS_DIR}/run_l1b_swept.sh" l1b2_native_held_object eval
            ;;
        l1b3)
            echo "L1-B3 formal paper evaluation is disabled while task 4 is candidate-only." >&2
            echo "Use run_l1b3_task4_outcome_v2.sh and complete its matched-Ec review gates first." >&2
            exit 2
            ;;
        l2b2)
            local variant
            for variant in ${L2B2_VARIANTS}; do
                SEED="${SCENE_SEED}" EVAL_SEED="${seed}" RUN_ID_SUFFIX="${suffix}" \
                    NUM_TRIALS="${trials}" \
                    bash "${TASKS_DIR}/run_l2b2_basket_stove.sh" "${variant}" eval
            done
            ;;
        l2c2)
            EVAL_SEED="${seed}" RUN_ID_SUFFIX="${suffix}" NUM_TRIALS="${L2C2_TRIALS}" \
                bash "${TASKS_DIR}/run_l2c2_bowl.sh" all
            ;;
        l3a1)
            SCENE_SEED="${SCENE_SEED}" EVAL_SEED="${seed}" RUN_ID_SUFFIX="${suffix}" \
                NUM_TRIALS="${trials}" \
                bash "${TASKS_DIR}/run_l3a1_drawer_bottle.sh" all formal
            ;;
        *)
            echo "Unknown family: ${family}" >&2
            exit 2
            ;;
    esac
}

# ── Modes ──────────────────────────────────────────────────────────────────────

do_prepare() {
    log "prepare: one-time scene generation with SCENE_SEED=${SCENE_SEED}, NUM_TRIALS=${NUM_TRIALS}"
    # L1-A1 + L1-A2 paired states and gates.
    SEED="${SCENE_SEED}" NUM_TRIALS="${NUM_TRIALS}" \
        bash "${TASKS_DIR}/run_l1a_evals.sh" generate
    SEED="${SCENE_SEED}" NUM_TRIALS="${NUM_TRIALS}" \
        bash "${TASKS_DIR}/run_l1a_evals.sh" l1a2_check
    # Active native-asset L1-B1/B2 paired states and validation gates.
    SCENE_SEED="${SCENE_SEED}" NUM_TRIALS="${NUM_TRIALS}" \
        bash "${TASKS_DIR}/run_l1b_swept.sh" all prepare
    # L2-B2 per-variant states.
    local variant
    for variant in ${L2B2_VARIANTS}; do
        SEED="${SCENE_SEED}" NUM_TRIALS="${NUM_TRIALS}" \
            bash "${TASKS_DIR}/run_l2b2_basket_stove.sh" "${variant}" check
    done
    # L2-C2 uses native states + a fixed BDDL; nothing to generate.
    if [[ " ${FAMILIES} " == *" l3a1 "* ]]; then
        SCENE_SEED="${SCENE_SEED}" NUM_TRIALS="${NUM_TRIALS}" \
            bash "${TASKS_DIR}/run_l3a1_drawer_bottle.sh" all prepare
        SAFE_REF_STATES="${SAFE_REF_STATES:-5}" \
            bash "${TASKS_DIR}/run_l3a1_drawer_bottle.sh" risk safe_reference
    fi
    log "prepare complete. Next: 'smoke' for a quick pass, then 'full'."
}

do_seed_loop() {
    local trials="$1" suffix_prefix="$2" seeds="$3"
    local seed family
    for seed in ${seeds}; do
        for family in ${FAMILIES}; do
            log "family=${family} seed=${seed} trials=${trials}"
            run_family "${family}" "${seed}" "${trials}" "${suffix_prefix}${seed}"
        done
    done
}

do_attribution() {
    log "attribution reports (needs Eb/Er/Ec trajectories)"
    local attribution_seed="${SEEDS%% *}"
    if [[ " ${FAMILIES} " == *" l1a1 "* ]]; then
        RUN_ID_SUFFIX="seed${attribution_seed}" RECORD_RESULTS=False \
            bash "${TASKS_DIR}/run_l1a_evals.sh" l1a1_attribution || true
    fi
    if [[ " ${FAMILIES} " == *" l1a2 "* ]]; then
        # L1-A2 shares L1-A1's native Eb gate; all representative attribution
        # trajectories use the same first formal seed suffix.
        L1A1_RUN_SUFFIX="seed${attribution_seed}" L1A2_RUN_SUFFIX="seed${attribution_seed}" \
            RECORD_RESULTS=False \
            bash "${TASKS_DIR}/run_l1a_evals.sh" l1a2_attribution || true
    fi
    if [[ " ${FAMILIES} " == *" l3a1 "* ]]; then
        python -m experiments.robot.libero.physcog_attribution \
            --eb "rollouts/libero_10/L3-A1-drawer-bottle-eb-native-seed${attribution_seed}/trajectories" \
            --er "rollouts/libero_10/L3-A1-drawer-bottle-er-support-removal-seed${attribution_seed}/trajectories" \
            --ec "rollouts/libero_10/L3-A1-drawer-bottle-ec-self-supporting-seed${attribution_seed}/trajectories" \
            --divergence_reference_condition ec \
            --family_name L3-A1 \
            --out "${LOG_DIR}/l3a1_attribution.md" || true
    fi
}

do_tables() {
    log "recording metrics and generating result tables"
    python "${TASKS_DIR}/record_experiment_results.py" --log_dir "${LOG_DIR}"
    local table_args=(--log_dir "${LOG_DIR}")
    if [[ -n "${POOL_SINCE}" ]]; then
        table_args+=(--pool_mode all --pool_since "${POOL_SINCE}")
    fi
    python "${TASKS_DIR}/generate_result_tables.py" "${table_args[@]}"
    echo
    echo "Seed-suffixed runs (-seedN) pool automatically; legacy unsuffixed runs"
    echo "are superseded once at least one seed run exists for a condition."
    echo "Tables: ${LOG_DIR}/result_tables.md (Table 5 = CI + Er-vs-Ec tests)"
}

case "${MODE}" in
    prepare)
        do_prepare
        ;;
    smoke)
        first_seed="${SEEDS%% *}"
        do_seed_loop "${SMOKE_TRIALS}" "smoke-seed" "${first_seed}"
        do_tables
        ;;
    full)
        do_seed_loop "${NUM_TRIALS}" "seed" "${SEEDS}"
        do_attribution
        do_tables
        ;;
    attribution)
        do_attribution
        ;;
    tables)
        do_tables
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Expected one of: prepare, smoke, full, attribution, tables" >&2
        exit 2
        ;;
esac
