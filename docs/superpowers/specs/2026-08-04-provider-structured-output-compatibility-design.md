# Provider Structured Output Compatibility

## Objective

Support reproducible HL control acts across Responses API providers that differ in native JSON-schema compatibility, while preserving strict local validation of every planner and reducer artifact.

## Configuration Contract

`run.provider.structured_output_mode` is an explicit enum:

- `native_schema`: pass the JSON schema to Codex CLI through `--output-schema` and validate the persisted artifact locally.
- `validated_file`: omit `--output-schema`, require the coding agent to write the designated JSON artifact, and validate that artifact locally with the same schema and semantic checks.

The default is `native_schema`. Provider-specific experiment configurations select `validated_file` only when their Responses API implementation does not accept native structured-output requests.

The selected mode is included in the frozen run configuration, provider fingerprint, preflight facts, and act metadata so a run can be reproduced and audited without implicit capability detection.

## Runtime Data Flow

1. The controller builds the bounded planner or reducer prompt and its local JSON schema.
2. The provider constructs the Codex CLI command according to `structured_output_mode`.
3. In `native_schema` mode, the command includes the output-schema and last-message paths.
4. In `validated_file` mode, the command omits both native structured-output arguments; the prompt's required workspace artifact remains authoritative.
5. The controller loads the workspace artifact with the existing strict parser, validates exact fields, counts, indexes, code symbols, and semantic invariants, then persists the accepted artifact into the proposal record.
6. Missing or invalid artifacts fail the act and cannot produce candidates or scientific iteration points.

## Error Handling

- The mode never changes automatically during an act.
- Transport errors remain provider attempts and do not consume a valid coding-agent act when no tokens or tools were used.
- A completed model request with a missing or invalid JSON artifact is a coding-agent failure and is recorded as such.
- Resume reconstructs the selected mode from the frozen run configuration and cannot silently switch protocols.
- `resume --allow-provider-compatibility-change` permits only a `structured_output_mode` transition. Every other frozen-config difference remains an error, and the accepted transition is appended as a `provider_compatibility_selected` event.

## AntWar2 Configuration

The Tsinghua `sub2api` AntWar2 configuration uses `validated_file`. The API agent still receives the strict artifact contract and the controller still applies the same `branch_briefs.json` validator. Only the unsupported native schema request parameter is omitted.

## Verification

Automated tests prove:

- config parsing, serialization, and default behavior for both modes;
- command construction includes schema arguments in `native_schema` and omits them in `validated_file`;
- preflight and provider metadata expose the selected mode;
- validated-file planner artifacts pass only when the strict four-branch contract is satisfied;
- missing or malformed artifacts fail before candidate rollout;
- AntWar2 selects `validated_file` while compatible configurations retain `native_schema`.

An integration probe using the configured provider must produce a valid planner artifact without native schema arguments before resuming paid k=4 candidate rollout.
