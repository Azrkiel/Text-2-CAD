# Text-2-CAD / Mirum — architecture hardening handoff

**Checkpoint date:** 2026-10-17  
**Status:** implementation and offline qualification complete locally; implementation not committed or published.  
**Next phase:** preserve/review the working tree, publish implementation separately when authorized, then qualify live-provider behavior if authorized.

## 1. Exactly where the project stands

| Item | State at handoff |
|---|---|
| Repository | `Azrkiel/Text-2-CAD` |
| Local checkout | `C:/Users/jiang/Text-2-CAD` |
| Local working branch | `architecture-hardening` |
| Local HEAD / original baseline | `9604df316c7051201be7689382498203906d5ced` |
| GitHub before this documentation upload | Only `main`, at the same baseline; no open PRs returned by the API |
| Implementation | All ten slices of the hardening plan implemented in the local working tree |
| Commit state | Many modified, deleted and untracked files; no implementation commit/push/merge |
| Final offline gate | **805 passed, 3234 warnings in 349.42 seconds**, exit 0 |
| Fresh independent fix reviews | Three passed; no blocking security or logic findings |
| Source identity | All **124 frozen code/configuration hashes** rechecked at handoff; no differences |
| Deployment | No production rollout, registry publication, or live-provider qualification |
| Temporary resources | Qualification servers stopped and disposable volume removed; absence rechecked |
| Retained local assets | Source, ignored `.hermes/` evidence, and all three qualified images by immutable ID; expected image tags are currently absent |

**Important:** this GitHub document is a documentation-only handoff. It does not publish the implementation. A fresh GitHub clone of `main` will not contain the hardened application, its new tests, or the local evidence. Preserve the existing checkout before switching machines or cleaning it. In particular, do not run `git clean`, `git reset --hard`, or an indiscriminate checkout/restore against this working tree.

The governing specification is local `docs/plans/2026-10-16-architecture-hardening.md`. Detailed local execution records are `docs/implementation/status.md` and `docs/implementation/release-qualification.md`. These paths are not assumed to exist on GitHub until the implementation is published; this handoff is self-contained for that reason.

## 2. What has been completed

| Slice | Delivered behavior |
|---|---|
| 1 — Shared contracts/test seam | Shared typed domain/API contracts and offline test boundaries; deterministic model fixtures instead of paid inference. |
| 2 — Manifest enforcement | Manifest IDs, references, anchors, connectivity and numeric constraints checked before CAD/model execution where applicable; immutable approved plans and canonical hashes. |
| 3 — Identifiable, isolated runs | SQLite-backed plan/run records, payload/operation-bound request IDs, per-run workspaces, opaque artifact IDs, immutable accepted artifacts, revision/head compare-and-swap. |
| 4 — Execution vs. acceptance | Supervised Linux CAD worker; successful Python execution alone cannot produce accepted geometry; real geometry validation and bounded trusted evidence. |
| 5 — Assembly acceptance | Static placement/joint rules, anchor resolution, body/collision checks, STEP reimport, and GLB evidence bound to effective geometry. |
| 6 — Bounded orchestration | One pipeline owner; bounded admission/concurrency, model/worker lifetimes, cancellation, drain/shutdown behavior and task ownership. |
| 7 — Plan/generate/refine connection | Stored-plan approval, typed SSE, terminal replay/conflict semantics, accepted-base refinement and edited-source child revisions. |
| 8 — Streamlit alignment | UI uses the current contracts, preserves the last accepted result on failure, tracks revisions, and routes edits through validation. |
| 9 — Restart evidence | Persistent accepted sources/artifacts, run state, diagnostics and telemetry; retention/recovery with active-base preservation; accepted-only training provenance. |
| 10 — Consolidation/deployment | Legacy production orchestration removed; startup separated from research helpers; pinned Linux runtime, hardened containers/Compose, and whole-suite native CI gate defined. |

### Supported scope, not an implied certification

- Tested deployment: trusted local, loopback-only, **single backend process**, Linux Docker.
- Solid parts and explicit named-body subassemblies are supported subject to validation.
- Six core **static** joints: FASTENED, REVOLUTE, SLIDER, CYLINDRICAL, PLANAR, BALL, within the documented reference/pose restrictions.
- GEAR/SCREW/RACK_PINION/CAM remain metadata-only, not silently enforced motion constraints or a source of placement connectivity.
- Seven mandatory acceptance checks: part validity, anchor resolution, assembly validity, mate residuals, collision check, STEP round-trip, GLB integrity.
- No full mechanism dynamics, swept-volume, manufacturing/strength/aerodynamic certification, general mesh-topology proof, public multi-tenant security, or hostile-code sandbox claim.

## 3. Verification already performed

### Final full gate

Ran the actual rebuilt test image with networking disabled, empty provider keys, UID/GID 10001, read-only root, dropped capabilities, no-new-privileges, resource limits, and tmpfs scratch/data:

```text
805 passed, 3234 warnings in 349.42s (0:05:49)
exit code 0
```

No skipped tests were reported. Warnings remain; passing does not mean warning-free. The native gate includes native-dependent tests under `tests/unit`, not just `tests/cad` and `tests/integration`.

Additional verification is overlapping evidence, **not counts to add to 805**:

- Parent's focused post-fix Linux run: 40 passed; source hashes unchanged during execution.
- Three fresh independent reviews: schema/retrieval, geometry/export, replay/API. All passed with before/after source-hash checks.
- Independent geometry review: 33 native tests plus nine unsupported-GLB mutation probes; the latter all rejected with `GLB_INVALID`.
- Independent replay/API review: 57 focused tests passed.
- Subsequent helper + SDK/core pytest selection: 10 passed in 22.34 seconds. Includes actual execution of the replay helper in the immutable release image, plus isolated verification of review-input generation.
- Earlier frontend subset: 71 passed; final frontend tests also ran in the full gate.
- Python 3.10 syntax scan, `git diff --check`, and `docker compose config --quiet` passed.

### Real runtime checks, with a clearly identified model fixture

The adapter returned deterministic box source. The CAD worker, API handlers, SQLite store, native assembly/export validation, artifact publication/downloads and shutdown were real.

1. Generated an accepted root revision with all seven checks completed and no issues.
2. Downloaded GLB/STEP/source/validation artifacts; sizes and SHA-256 matched their descriptors.
3. A new container reopened the persistent test volume and replayed the exact stored terminal result.
4. An editor change created a validated child revision; child geometry/source hashes changed while parent downloads remained byte-identical.
5. Restart/replay/editor phase made **zero model calls**.
6. Separate controlled-clock test expired the parent while retaining its successful child: matching request replay returned **200**, changed content with the same ID returned **409 REQUEST_ID_CONFLICT**, with zero model reexecution.
7. Actual backend entrypoint returned ready worker/storage and no telemetry degradation. A second backend on its volume failed startup with exclusive-ownership exit 3.
8. Actual frontend entrypoint returned HTTP 200/`ok`; Streamlit AppTest rendered idle UI with zero exceptions and HTTP calls forbidden. This was not a browser/WebGL visual test.
9. Real pinned SDK schema normalization and protobuf serialization passed offline: 7696 serialized bytes. This did not validate provider authentication/connectivity.

### Local images at qualification

| Local tag | Immutable local image ID |
|---|---|
| `mirum-architecture-tests` | `sha256:a1893b6849094d775ef42ccc41283c36fb13af8c7b327c94c68d7e13c6be028a` |
| `mirum-release-backend` | `sha256:7062a8b18847bef244e3bd4f82d6a78a4be9e617dc0e668ee4fde53f101a8c08` |
| `mirum-release-frontend` | `sha256:cfe96abf6c60e652a261c53d946d6059efb8d127e1a8d94198376e35206af48a` |

At handoff, all three immutable image IDs above still exist in Docker context `desktop-linux`, but the corresponding friendly tags do **not** resolve. The reason for tag removal is not established. Use the immutable IDs, explicitly restore tags after verifying identity, or rebuild; do not interpret missing tags as evidence that the old qualification failed. No tags were changed during this documentation task.

These are local Docker image IDs, not published registry digests. Compared image contents against the frozen source manifest: 38 backend/shared, 8 frontend/shared and 115 test-image source files; zero mismatches. Native stack: Python 3.10, CadQuery 2.6.1, cadquery-ocp 7.8.1.1.post1, pinned `requirements-linux.lock`.

## 4. Issues, fixes and lessons

The pre-fix candidate passed 786 tests but was **rejected by independent review**. Passing a large suite did not supersede a concrete failure. Four distinct defects then received actual RED/GREEN fixes and fresh review:

| Issue encountered | Root cause / implemented solution |
|---|---|
| Production planner schema rejected by pinned Google SDK | Raw Pydantic schema retained unsupported fields/unions and unresolved references. Recursively translate to supported SDK dialect, expand refs, preserve nullable/enum/required semantics, and retain full server-side domain validation. Test the real pinned encoder offline. |
| Retained child could not replay after parent expiry | Edited-source preparation fetched the parent before replay lookup, returning 410. Canonical payload/operation replay or conflict lookup now comes first; new invalid requests still validate before claiming an ID. |
| Corrupt optional corpus aborted generation | UTF-8 errors escaped file iteration handling. Decode inside per-record error handling, preserve valid accepted rows, initialize empty/partial indexes, and expose optional degradation without blocking generation. |
| GLB sparse geometry override passed base-geometry checks | Checking only base POSITION buffers missed effective deformed geometry. Fail closed on unsupported consumed sparse accessors, morph targets and skins; normal native exports remain accepted. |

Other nontrivial challenges addressed during implementation:

- **Cancellation/lifecycle races:** explicitly own and drain model tasks; abort/drain admitted work even for SSE disconnect/send failure before body iteration starts.
- **Request-ID poisoning:** stale preflight, bad source and capacity rejection must not consume an ID before actual admission; retain transactional claim/CAS checks.
- **False geometric evidence:** independently reimport STEP; bind source/body/export evidence; handle nested body transforms and static mate footprints/clearance; do not equate successful execution with valid geometry.
- **Tessellation changing BRep bounds:** native curved fixtures exposed mutable mesh-cache-dependent bounds; switched to tessellation-independent BRep bounding evidence.
- **Memory control not actually inherited:** real child address-space limit was unlimited; an exec wrapper now applies the configured/default cap before running the worker, with an actual excess-allocation regression.
- **Restart/recovery mismatches:** orphan-stage matching now recognizes the actual hyphenated publication-stage naming while preserving unrelated directories.
- **Training contamination:** execution-only successes were previously eligible. Positive examples now need explicit final acceptance, validation, run identity, original description and complete-source digest; accepted insertion rebuilds retrieval weights.
- **Host policy restrictions:** Windows Application Control blocked attempted native OCCT/PyArrow use. No policy was disabled/bypassed; real-kernel qualification used Linux Docker.
- **Windows tool behavior:** use `.venv/Scripts/python.exe -m pytest`, not the standalone pytest launcher that previously timed out. Git Bash Docker Linux paths need `MSYS_NO_PATHCONV=1`.
- **Late notifications:** delayed build/review completion messages were duplicates of already-inspected results. They did not represent new failures or justify repeating implementation.

## 5. What remains to be done

No blocking implementation/review defect is currently known **within the qualified offline scope**. The following are separate next steps, not completed work:

### Immediate continuation priorities

1. **Preserve and publish the implementation intentionally.** The local branch is still at the original baseline, with substantive changes uncommitted and many new files untracked. Inspect the complete diff and untracked files, audit for secrets, then create scoped implementation commits/PRs only when authorized. This handoff-only upload does not accomplish that.
2. **Check the worktree before changing anything.** Recheck branch/HEAD and source hashes against `.hermes/review-static.json`. Do not blindly rerun old fix agents or overwrite the fixes described above. Any code change invalidates prior source identity and requires relevant tests/fresh qualification.
3. **Resolve missing local image tags before using tag-based commands.** All three qualified image IDs remain present, but `mirum-release-backend`, `mirum-release-frontend` and `mirum-architecture-tests` tags do not currently resolve in `desktop-linux`. Use the recorded immutable IDs, verify/recreate the intended tags, or rebuild.
4. **Small nonblocking coverage follow-up:** permanently add sparse triangle-index accessor regressions, including no-base-bufferView, beside sparse POSITION tests. The common guard and independent native probes already reject these; this is a durable-test coverage suggestion, not an open bypass finding.
5. **Live-provider qualification requires explicit authorization and credentials.** Test real authentication/connectivity, structured response behavior and semantic output quality separately. Current fixture generation and SDK encoding do not prove those. Never introduce paid calls into the offline gate.
6. **Run remote CI after implementation publication.** Workflows exist locally, but no GitHub Actions success for this hardened implementation is claimed.
7. **Inspect visual behavior if needed.** A real browser/WebGL interaction session is still unqualified; idle Streamlit AppTest is narrower evidence.
8. **Only then consider release/publishing/deployment**, within the trusted local single-backend scope. No registry push or production rollout has occurred.

### Deliberately deferred scope

- Durable distributed queues, inference resumption, horizontal replicas/distributed admission/rate coordination.
- Full-motion mechanisms and coupling enforcement, manufacturing/process/FEA/aerodynamic/semantic-feature certification.
- GNN/RLAIF/SFT/constrained-decoding research rollout before trustworthy labels/evaluation baselines.
- Comprehensive metadata deletion, backup/privacy policy, secure erasure and disk-quota governance. Rotation/pruning are not secure-delete guarantees.
- Public authentication/authorization, stronger hostile-code isolation and a multi-tenant threat model.
- Dependency/base-image governance: lock refresh, hash-verified wheels and digest-pinned OS images; upstream Python 3.10 support/deprecation warnings require planned upgrades and new native/SDK qualification.
- Native Windows CAD execution while host policy blocks it; do not bypass host controls.

## 6. File map for the next person/agent

Paths below refer to the **existing local checkout**, not necessarily the docs-only GitHub branch.

| Area | Main paths |
|---|---|
| Specification / state | `docs/plans/2026-10-16-architecture-hardening.md`, `docs/implementation/status.md`, `docs/implementation/release-qualification.md`, `docs/TODOS.md` |
| Operator guide | `README.md`, `architecture.md` |
| Contracts | `shared/domain.py`, `shared/contracts.py` |
| HTTP / streaming | `backend/main.py`, `backend/streaming.py` |
| Pipeline / planning / revisions | `backend/pipeline.py`, `backend/planning.py`, `backend/refinement.py` |
| Store / artifacts / lifecycle | `backend/run_store.py`, `backend/artifacts.py`, `backend/lifecycle.py`, `backend/config.py` |
| Worker / validation | `backend/cad_worker.py`, `backend/_memory_entry.py`, `backend/validation.py`, `backend/assembly.py`, `backend/export_validation.py` |
| Source / provider boundary | `backend/source.py`, `backend/models.py` |
| Accepted-only analytics | `backend/retrieval.py`, `backend/telemetry.py`, `data_pipeline/prompt_stratification.py` |
| UI | `frontend/app.py`, `frontend/client.py`, `frontend/state.py` |
| Deployment / CI | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements-linux.lock`, `.github/workflows/architecture-checks.yml` |
| Final defect regressions | `tests/unit/test_planner_sdk_schema.py`, `tests/unit/test_review_core_regressions.py`, `tests/cad/test_glb_unsupported_features.py` |

### Evidence to preserve locally

`.hermes/` is ignored and **is not uploaded with this handoff**. It contains source hashes, test logs, fix/review reports, and disposable fixture scripts. Preserve it separately if moving the work; review/redact sensitive prompts/source before sharing logs.

- `.hermes/final-closeout.json`: machine-readable completed offline checkpoint.
- `.hermes/review-static.json`: final 124-file code/configuration hash manifest and static scan.
- `.hermes/review-r1-manifest.json`, `.hermes/review-r1-diff.patch`: focused fix-cycle snapshot.
- `.hermes/r1-review-schema-retrieval.json`, `.hermes/r1-review-geometry.json`, `.hermes/r1-review-replay.json`: independent passing reviews.
- `.hermes/fix-schema-report.json`, `.hermes/fix-glb-report.json`, `.hermes/fix-core-report.json`: actual defect/fix/RED/GREEN details.
- `.hermes/release-full-tests-r1.log`, `.hermes/fix-targets-parent.log`, `.hermes/build-*-r1.log`.
- `.hermes/release-image-evidence-r1.json`, `.hermes/release-cleanup-r1.json`.
- `.hermes/release_probe.py`, `.hermes/expiry_replay_probe.py`, associated smoke/expiry/SDK logs.
- `.hermes/prepare_fix_review.py`, `.hermes/test_release_helpers.py`: review-input generation and executable helper checks.

The named smoke volume was deliberately deleted; do not expect old fixture artifact URLs to remain downloadable. Regenerate test data via isolated probes if needed. Built image IDs are local and may be lost to a later Docker cleanup. At handoff their tags are already absent, although the immutable images remain present. Rebuild from the preserved source/lock, not from unchanged GitHub `main`.

## 7. Safe verification commands

Run from the existing checkout. These are verification commands, not authorization to make model calls or deploy.

```sh
git branch --show-current
git rev-parse HEAD
git status --short
git diff --check
docker compose config --quiet

# Existing Windows virtualenv: lightweight/helper verification.
.venv/Scripts/python.exe -m pytest \
  .hermes/test_release_helpers.py \
  tests/unit/test_review_core_regressions.py \
  tests/unit/test_planner_sdk_schema.py \
  -q -p no:cacheprovider --tb=short

# Build from preserved current source, then run the complete native offline gate.
docker build --target test -f backend/Dockerfile -t mirum-architecture-tests .
MSYS_NO_PATHCONV=1 docker run --rm --network none --read-only --init \
  --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 256 --memory 4g --cpus 2 \
  --tmpfs /tmp:rw,size=1g,mode=1777 \
  --tmpfs /data:rw,size=1g,uid=10001,gid=10001,mode=0700 \
  -e GEMINI_API_KEY= -e GOOGLE_API_KEY= \
  mirum-architecture-tests python -m pytest tests -q \
  -m "not live" -p no:cacheprovider --tb=short
```

The helper test uses the immutable local backend image listed above. If it is unavailable, restore/rebuild and verify the image/source identity rather than silently substituting another image. On non-Windows hosts use the appropriate existing Python environment instead of `.venv/Scripts/python.exe`. Registry/package downloads during a build may require network; actual tests use no provider network calls.

## 8. Publication and operating guardrails

- Publish **this handoff only** on a dedicated docs branch/PR; leave implementation, original working branch and index untouched.
- Do not merge, deploy, install/upgrade packages, configure secrets or initiate paid inference as an implied continuation of documentation work.
- Never publish `.env`, provider credentials, raw private telemetry, or unredacted agent transcripts.
- A configuration-presence sentinel and passing `/readyz` do not prove provider authentication.
- Preserve last accepted artifacts/revisions when a child fails. Preserve IDs for transport retry, use a new ID for a deliberate new attempt.
- Keep one backend owner of the data volume. Do not scale Uvicorn workers/replicas or expose the trusted-local service publicly without redesign and qualification.
