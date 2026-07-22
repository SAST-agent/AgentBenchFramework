# CI Dashboard Implementation Plan

1. Add a regression fixture covering summary budget, act evaluations, and raw
   policy KL events; verify the old builder fails the new dashboard contract.
2. Extend the CI loader with forward-compatible event reading and derived
   research series while retaining all raw summary fields.
3. Replace the report shell and index page with the research dashboard, inline
   SVG icon sprite, responsive cards, and chart helpers.
4. Update game, agent, and comparison pages to use the same shell and preserve
   their existing links and legacy metrics.
5. Extend aggregation to copy new summary fields into the registry without
   dropping unknown future fields, then run unit tests and a static build.
