# Planner Diversity Contract

## Objective

When a cached Ghost-policy distillation is available, each four-candidate HL cycle must explore complementary strategy families instead of producing four variants of defensive risk avoidance.

## Contract

The planner assigns the four existing branch indices these responsibilities:

1. Build a coordinate-free Ghost predictor from the shared modal distillation and use it for a Rollman best response.
2. Improve offensive progress, scoring, or level completion while conditioning on predicted Ghost movement.
3. Use replay counterfactuals to identify and deny the opponent's scoring source or reproduce an effective route pattern in Rollman's own action space.
4. Explore a genuinely different mechanism supported by the replay summaries and research state.

At least two branches must optimize progress, score, or opponent denial rather than primarily vetoing, retreating, waiting, or avoiding contact. Ghost actions are never copied directly because Rollman and Ghost have different action semantics and objectives. The primitive Rollman decision space used for KL measurement remains unchanged.

## Data Flow

The existing content-addressed distillation artifact remains the only opponent-model input. The planner reads it in the same bounded first call and emits the existing `branch_briefs.json` schema. Candidate packets continue to embed the same distillation, so no additional API call or static context transfer is introduced.

## Failure Handling

Without a shared distillation artifact, the planner retains mechanism diversity and replay-grounded causal requirements without claiming opponent-model evidence. Existing candidate activation, fixed-seed screening, finalist gates, rollback, full-pool reporting, and certification remain unchanged.

## Verification

Prompt contract tests verify that the branch-role assignment appears only when a shared distillation is supplied, preserves the asymmetric-role warning, requires offensive diversity, and does not request an additional distillation run. The focused context tests and full test suite must pass before resuming API experiments.
