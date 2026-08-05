---
name: doto-agent-authoring
description: Use when creating, modifying, compiling, or reviewing a native 23rd DOTO C++ candidate that must implement the original playerAI.cpp SDK contract safely.
---

# DOTO Agent Authoring

Maintain exactly one **complete playerAI.cpp** defining `void playerAI()`.
Return or save the complete file, never a patch fragment, header replacement,
SDK edit, executable, or multi-file candidate.

1. Read `doto-game-rules` before designing behavior.
2. Read [references/sdk-api.md](references/sdk-api.md) before using SDK fields or
   methods; its one minimal example is compile-backed.
3. Read [references/safe-policy-patterns.md](references/safe-policy-patterns.md)
   before adding targeting, roles, geometry, or persistent state.
4. Build through `python -m agentbench_frame.doto iteration build` or the atomic
   `build` command. Treat warnings, stderr, missing `main.out`, and stale builds
   as evidence, not text to ignore.
5. Select revisions only from public training evidence. Never inspect, infer,
   or tune against sealed test policies or a sealed test result.

The fixed build owns `sdk/main.cpp`, `sdk/playerAI.h`, `sdk/logic.*`, geometry,
jsoncpp, logging, official `Maps`, and the vendored `official_server`. Do not
modify these dependencies to make a candidate compile.

## Candidate contract

- Include `playerAI.h`; define exactly one callable `void playerAI()`.
- Obtain state and set operations through `Logic::Instance()`.
- Emit finite absolute coordinates and use SDK cancellation methods for no-op.
- Check death, cooldown, remaining uses, range, wall, and carrier constraints.
- Keep every frame fast and deterministic. Bound loops and debug output.
- Preserve useful state deliberately across frames; reset it on process start.
- Build before evaluation; never evaluate a failed or stale executable.

## Common mistakes

- Do not confuse local action slot with global human ID.
- Do not aim with a direction vector where the SDK expects a destination.
- Do not call `flash(i)` without setting `move(i, destination)`.
- Do not assume the request executed; verify the next replay state.
- Do not use hidden identities, source, paths, or outcomes during authoring.
