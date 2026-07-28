# PR Review Responses Streaming Implementation Plan

1. Extend request-contract tests to require streaming only in Responses mode.
2. Add end-to-end local SSE tests for successful deltas, completed-response
   fallback, explicit failure, and premature EOF; confirm they fail first.
3. Add an incremental fail-closed SSE parser and route Responses requests
   through it while leaving Chat Completions JSON handling unchanged.
4. Document the transport and terminal-event contract.
5. Run focused tests, the complete reviewer test module, compilation, and diff
   checks.
6. Push the change through a PR, merge after verification, then rerun a
   historical PR check against the real proxy.
