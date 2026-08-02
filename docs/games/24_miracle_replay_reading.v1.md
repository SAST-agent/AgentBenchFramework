# 24_miracle Replay Reading Core v1

## Meaning of “watch”

Replay Reading v1 turns one independently approved canonical synthetic
training replay into an ordered structured timeline. “Watch” means reading
DecisionFrames step by step as data or stable text. It is not a video player,
does not run the game, and does not execute a strategy.

The original core supports synthetic/fake-only artifacts. The separate
`real_replay_adapter_v1` mechanism described below can audit and normalize a
real Judge replay triad. Production access remains digest-gated: the H08
role=train Manifest is the sole approved real-Judge digest, its final
HumanReplaySkill is the sole approved human replay Skill digest, and synthetic
replay approvals remain empty.

## Trust chain

The trust chain is:

```text
production allowlist digest
  -> canonical manifest bytes
  -> replay SHA-256
  -> complete replay bytes (frames + terminal)
  -> match plan, case, role, seeds, policy, and champion identities
```

The synthetic replay approval set is empty. Ordinary run data cannot add an
approval. Tests may temporarily inject a fake digest in test scope only. The
separate Real Judge approval boundary is described below.

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
Every ancestor and the final component must be a real directory; symlink/reparse components,
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
Judge process. Experience Skill approval and human Replay Reading remain
separate lifecycle concerns. Tracking, Results, reports, evaluation lifecycle,
and benchmark publication are explicitly deferred.

## Real Judge triad adapter v1

`real_replay_adapter_v1` is a parallel, versioned input contract. It does not
weaken or reinterpret the synthetic schemas above. Its production trust chain
is:

```text
independently approved canonical real-manifest SHA-256
  -> bounded canonical manifest bytes
  -> bounded run-intent, .replay, .jsonl, and .result.json bytes + SHA-256
  -> Framework, adapter, enumerator, runner, Judge, and both player identities
  -> binary replay header/events/unique GameEnd/-1 trailer
  -> ordered Judge observations and AI operations
  -> regenerated complete ActionSupport for evaluated_camp only
  -> immutable RealJudgeReplayPacket / RealJudgeDecisionFrame values
```

The production allowlist lives in the independent
`real_replay_approvals_v1` data module and contains exactly the project-owner
approved H08 Manifest digest. It is not part of the adapter source identity,
so adding that separately reviewed digest did not invalidate the Manifest that
binds the stable adapter implementation. `preflight_real_judge_replay()`
rejects an unapproved manifest digest before JSON decoding or parsing.
`audit_real_judge_replay_candidate()` performs the same structural checks for
approval review but always returns `authoritative=false`; it is not a production
authorization shortcut. Its packet itself carries `authoritative=false` and
`evidence_scope=candidate_unapproved_real_judge_evidence`. Issued production
contexts are closure-registered, reopen both approved roots, and revalidate
every byte and identity. Only `open_real_judge_replay()` registers and returns
an `authoritative=true` packet. `require_authoritative_real_replay_packet()`
rejects candidate, forged, unregistered, or subsequently revoked packets.

All filesystem reads reuse the retained-handle, beneath-only path boundary of
the synthetic reader: Linux requires `openat2` with `RESOLVE_BENEATH`,
`RESOLVE_NO_SYMLINKS`, and `RESOLVE_NO_XDEV`; Windows uses handle-relative
`NtCreateFile`, rejects reparse points, and rechecks the approved root identity.
Platforms that cannot prove those guarantees fail closed. Manifest paths are
strict relative paths, and every input has a fixed size cap, strict digest, and
strict UTF-8/JSON boundary where applicable.

Source identities use `strict_utf8_canonical_lf_v1`: source bytes must be
strict UTF-8 without BOM or bare carriage returns, CRLF and LF are normalized
to LF, and the normalized bytes are hashed. This binds the same Git source
identity on Windows and Linux without treating arbitrary text rewrites as
equivalent.

The real schema accepts only `role=train`. It binds `evaluated_camp`, both
player tree identities, actual `map_type` and `day_time`, and explicitly records
`external_deterministic_seed_supported=false` with
`seed_status=unsupported_not_recorded`. It never substitutes a null, zero, or
random seed into the synthetic three-seed contract.

Only public Agent operations (`init`, `move`, `attack`, `summon`, `use`,
`endround`, and `surrender`) become real DecisionFrames. Judge framing and
internal candidate evaluation do not. Each chosen evaluated-camp operation is
canonicalized, checked against a newly regenerated complete ActionSupport, and
deep-frozen with its before-state and identity references. A next evaluated-camp
observation is a causal `state_after` only when no other `ai_operation`
intervenes. If another Agent acts first, `state_after=None`; the later state is
stored only as non-causal `next_observed_state`, together with its observation
trace line, the intervening action count, and every intervening operation line.
Consumers must not attribute that overall delta to the earlier chosen action.
If terminal evidence arrives first, the final frame records a proved terminal
event with `state_after=None`; it never invents a post-state or rationale.

Opponent operations are audited separately. An incompatible legacy opponent
cannot invalidate otherwise proven evaluated-camp frames when the manifest's
scope is `evaluated_camp_only`; it is excluded with exact compatibility counts.
An all-camp conversion request fails closed.

The H08 canonical Manifest is
`assets/24_miracle_h08_real_judge_manifest.v1.json`. It binds the real
`miracle_ifelse` versus `rank02` triad and expects 399 evaluated-camp frames,
399 complete supports, and 399 chosen-in-support actions. Of those frames, 348
have direct post-action observations, 50 cross one or more opponent decisions,
and one terminal frame has no ordinary post-state; the 50 non-causal intervals
contain 1510 intervening opponent operations. Project owner 龚恒 approved the
exact Manifest digest on 2026-08-02. The unmodified production preflight,
open, and lifecycle gate now issue and accept an authoritative packet with
`evidence_scope=approved_real_judge_training_replay`; the canonical result is
recorded in `assets/24_miracle_h08_production_replay_preflight.v1.json`.

The associated
`assets/24_miracle_h08_human_replay_skill.final-candidate.v1.json` preserves
the complete H03-B, H04-B, H07, H01–H08, evidence, scope, output, forbidden
inference, and causal-transition rules as `reader-v6-final-candidate`. Its
content is finalized, and project owner 龚恒 approved its exact canonical
digest on 2026-08-02. Authority is granted only by the external
`APPROVED_HUMAN_REPLAY_SKILL_SHA256` registry, which contains exactly that one
digest. The approved Skill bytes do not claim self-authority and were not
changed when the external approval was granted. The approval and successful
production-loader/replay-gate checks are recorded in
`assets/24_miracle_h08_human_replay_skill_approval.v1.json`.
