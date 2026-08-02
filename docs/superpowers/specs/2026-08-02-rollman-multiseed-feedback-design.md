# Rollman Multi-Seed Staged Feedback Design

## Objective

The Rollman HL proposal cycle shall give every candidate enough independent game feedback to diagnose a mechanism across trajectories while retaining an unseen selection gate. The protocol remains a linear `k=4` search with immutable versions, scoped repair, rollback, bounded research state, and hidden certification.

## Evaluation Protocol

For the four fixed learning seeds `[101, 102, 103, 104]`:

1. Every initial candidate plays the common quick-screen seeds `[101, 102]`.
2. Each selected repair descendant plays the same quick-screen seeds `[101, 102]`.
3. Initial and repaired descendants are compared only on identical quick-screen evidence.
4. The strongest two branch representatives play the disjoint finalist seeds `[103, 104]`.
5. The quick-screen and finalist results are combined for successor selection.
6. The parent is evaluated on all four fixed learning seeds and advances only when the candidate wins the existing lexicographic comparison.
7. Certification seeds `[201, 202, 203, 204, 205]` remain hidden from coding acts.

The configuration fields are authoritative:

```yaml
iteration:
  quick_screen_seeds: 2
  finalist_seeds: 2
```

The evaluator rejects a staged split whose counts exceed the number of fixed learning seeds. Both counts must be positive and their seed slices must be disjoint.

## Feedback Flow

The planner and four candidate acts receive the parent evaluation summaries for all fixed learning seeds. A repair act receives:

- the branch brief;
- parent results on all fixed learning seeds;
- candidate results on both common quick-screen seeds;
- bounded replay summaries;
- explicit replay and trace paths for at most two point probes;
- persisted research state and Experience Skill.

Complete replays and traces are not embedded in prompts. Static rules, decision-space definitions, and replay guidance remain content-addressed files referenced through the compact game digest and context manifest. This keeps additional provider input limited to compact factual summaries while local match execution supplies the extra trajectories.

## Stagnation Intervention

At curriculum stagnation count three, the existing coordinate-independent Ghost behavior distillation is generated from legal target traces and shared with the planner and candidates. Branches must use the distilled behavior as a predictive best-response signal for Rollman decisions; they may not inspect human source code or copy Ghost actions.

The four branch mechanisms remain distinct. Parameter grids, coordinate memorization, opponent-identity branches, and seed-specific code are invalid.

## Failure Handling

- An incomplete quick-screen case makes the candidate ineligible.
- A failed or timed-out repair leaves its immutable initial candidate eligible.
- A candidate that improves one quick seed but regresses the other is ranked by the existing game-grounded lexicographic diagnostics across both matches.
- A finalist that fails on unseen seeds cannot replace a stronger parent.
- A provider timeout cannot promote an unchanged or invalid artifact.

## Reproducibility and Telemetry

Events and reports record each phase, seed, opponent, score, replay, trace, version, and provider act. The aggregate curve keeps one integer point per complete proposal cycle. Provider token accounting remains separate from local match count, allowing ablations of `k`, quick-screen seed count, finalist seed count, repair count, and distillation without code changes.

## Verification

Tests shall prove that:

- configured quick-screen and finalist counts reach the evaluator;
- every sibling and repair uses the same two quick seeds;
- finalists use only the two disjoint remaining seeds;
- combined evaluation contains all four seeds exactly once;
- invalid staged splits fail before a billable provider act;
- the default single-seed staged behavior remains available through configuration;
- the complete HL and project test suites pass.
