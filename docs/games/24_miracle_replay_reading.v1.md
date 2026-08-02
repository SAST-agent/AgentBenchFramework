# 24_miracle Replay Reading Core v1

## Meaning of “watch”

Replay Reading v1 turns one independently approved canonical synthetic
training replay into an ordered structured timeline. “Watch” means reading
DecisionFrames step by step as data or stable text. It is not a video player,
does not run the game, and does not execute a strategy.

This core supports synthetic/fake-only artifacts. It does not establish that a
real replay, evaluation, benchmark, or authoritative experiment has completed.

## Trust chain

The trust chain is:

```text
production allowlist digest
  -> canonical manifest bytes
  -> replay SHA-256
  -> complete replay bytes (frames + terminal)
  -> match plan, case, role, seeds, policy, and champion identities
```

Production approvals are empty. Ordinary run data cannot add an approval.
Tests may temporarily inject a fake digest in test scope only.

Preflight reads bounded manifest bytes, computes their SHA-256, and rejects an
unapproved digest before UTF-8 decoding, JSON parsing, or canonicalization.
Only independently approved bytes reach the manifest parser.

`preflight_replay_reading()` accepts only `role=train`. A validation or test
artifact remains forbidden even if its file is renamed. Both manifest and
replay must be canonical UTF-8 JSON without BOM, extra fields, NaN, Infinity,
or alternate whitespace serialization.

The issued `ReplayReadingContext` contains no mutable ReplayPacket or open file
handle. A closure-owned weak registry binds each preflight-created object to a
strict immutable field snapshot; copied or modified contexts are not issued.
`open_replay_reading()` rereads
the files and rechecks approval, bytes, digest, role, replay SHA, and all nested
identities. A normally constructed or `object.__new__` context is rejected;
packet/role replacement and nested manifest mutation cannot become trust.

Paths are confined to the approved root. On Linux, the approved root and every
relative replay component are opened from retained directory descriptors with
`openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_XDEV)`; this rejects
cross-device mounts and same-device bind mounts. Other POSIX platforms fail
closed because `O_NOFOLLOW` and device numbers alone cannot prove the same
mount boundary. On Windows, only the drive or UNC-share anchor is opened by
pathname; every descendant is opened atomically relative to its retained parent
handle through `NtCreateFile(RootDirectory=...)`, with reparse-point checks and
exact final-handle paths. Before every relative read, both the retained root and
the current lexical root must still match the approved file identity and path.
Every approved-root ancestor must be a real directory, and the final manifest
or replay component must be a regular file; symlink/reparse components,
missing files, and path replacement fail closed. Manifest-contained replay paths reject
`..`, absolute paths, Windows drives, and ADS syntax. The approved root and
supplied manifest path must be absolute; relative inputs are rejected rather
than interpreted against process CWD. Mixed filesystem separators are checked across the complete raw path
before `splitdrive` or normalization; Windows device namespace paths are rejected. An exact `Path` exposes
only the separator form retained by `pathlib`; discarded lexical spelling cannot be recovered.
Manifest reads are capped at `MAX_REPLAY_MANIFEST_BYTES = 262144`; synthetic replay
reads are capped at `MAX_SYNTHETIC_REPLAY_BYTES = 16777216`. The same bounded
fd/handle supplies the bytes and the before/after identity and metadata checks.

## DecisionFrame

Every frame contains:

- a strict continuous `decision_step` in `1..N`;
- canonical `state_before` and `state_after` identities plus observations;
- strict-integer before/after observation camps equal to the case's
  `evaluated_agent_camp`;
- the complete trusted ActionSupport regenerated from `state_before`;
- one chosen canonical command belonging to that support;
- acting policy and champion identity references;
- reward, outcome, terminated/truncated flags;
- case and three-seed identity references;
- `rationale_status: not_recorded`.

Frames must form a continuous state chain. Only the final frame may terminate,
and the separate terminal record must exactly agree with its reward, outcome,
and terminal flags. Replay Reading never guesses or generates an action reason.

## Timeline

`render_replay_timeline()` accepts only an issuer-created
`ReplayReadingContext`. It calls `open_replay_reading()` on every render, so
approval revocation or any post-preflight manifest/replay byte change fails
before output. A publicly constructed `ReplayPacket`, including one claiming
`role=validation` or containing no frames, is never a renderer input.

After revalidation, the renderer emits a deterministic header with case, role, and
all seeds, followed by ordered frame lines showing step, state identity,
complete legal-action count, chosen action, policy/champion identity, reward,
outcome, and `rationale_status=not_recorded`. A final terminal line records the
terminal outcome and flags. Agents can instead inspect the ordered ReplayPacket
and DecisionFrame structures directly.

Every dynamic text value is encoded as an RFC 3986 percent-encoded UTF-8 atom.
The chosen command is first serialized as canonical JSON and then encoded by
the same rule. Newlines, carriage returns, pipes, backslashes, equals signs,
and other delimiters therefore cannot create extra lines or fields; decoding
is deterministic and reversible. The renderer still records
`rationale_status=not_recorded` and never invents rationale.

## Isolation and deferred work

This module creates no workspace, log, store, runner, session, Provider, or
Judge process. Experience Skill generation and human Replay Reading rules are
separate lifecycle concerns and are not implemented here. Real replay adapters,
tracking, Results, reports, evaluation lifecycle, and benchmark publication are
explicitly deferred.
