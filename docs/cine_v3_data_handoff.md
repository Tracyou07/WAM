# Cine v3 data-side handoff

Implemented interfaces below are the preparer/trainer contract.
Native type registration is `cine_v3_latent`; existing LIBERO builders remain available.

## Raw repository API

```python
from open_wam.data.cine_v3 import CineV3Repository, fit_cine_action_statistics, validate_cine_split_sources
repo = CineV3Repository(root)
records = repo.episode_records  # tuple of episode metadata dicts, sorted by episode_index
rows = repo.read_rows(episode_index, start=0, stop=None)  # sorted episode-local frame range [start,stop)
rgb = repo.read_rgb_window(episode_index, start, count)  # uint8 [count,3,H,W], one true color camera
stats = fit_cine_action_statistics(repo, episode_indices=[...])
validate_cine_split_sources(train_repo, val_repo, train_episode_indices, val_episode_indices)
```

`repo.info`, `repo.tasks` (task_index -> prompt), `repo.identity` (resolved root,
info/tasks SHA256 and combined episodes-parquet SHA256) are public. Parquet reader uses episode/frame
predicates across all data shards, never assumes an episode filename. Video
lookup uses episode metadata `videos/observation.images.color/{chunk_index,
file_index,from_timestamp,to_timestamp}` and row timestamp plus from_timestamp.
Reject non-v3, non-30-FPS, missing color, action/state dimensions other than 7,
non-contiguous/misaligned rows, and unresolved tasks. Raw values are not converted
to EEF, differentiated, resampled, padded to 8D or given a fabricated camera.

## Cache manifest v1

Place `manifest.json` in `data.latent_root`; payload paths are relative to this
cache root. The preparer writes new outputs only outside raw data. JSON schema:

```json
{
  "schema_version": "cine_v3_latent_v1",
  "complete": true,
  "fps": 30,
  "camera": "observation.images.color",
  "action_semantics": "raw_joint_command",
  "action_normalization": "none",
  "raw_window_frames": 33,
  "latent_frames": 9,
  "latent_stride_frames": 4,
  "canonical_size": [224, 448],
  "encoding": {"vae_identity": "explicit pinned identity", "text_encoder_identity": "explicit pinned identity"},
  "sources": {"train": {"root": "absolute raw train root", "info_sha256": "...", "tasks_sha256": "...", "episodes_sha256": "..."},
              "validation": {"root": "absolute separate raw validation root", "info_sha256": "...", "tasks_sha256": "...", "episodes_sha256": "..."}},
  "selection": {"scope": "subset", "window_stride": 1,
                "train": {"episode_indices": [0], "raw_starts": {"0": [1]}, "window_count": 1, "available_episode_count": 1},
                "validation": {"episode_indices": [0], "raw_starts": {"0": [1]}, "window_count": 1, "available_episode_count": 1}},
  "action_statistics": {"source": "train", "root": "absolute raw train root", "episode_indices": [0],
                        "count": 100, "mean": [0,0,0,0,0,0,0], "std": [1,1,1,1,1,1,1]},
  "samples": {"train": [{"path": "train/episode0-start1.pt", "episode_index": 0, "raw_start": 1}],
              "validation": [{"path": "validation/episode0-start1.pt", "episode_index": 0, "raw_start": 1}]}
}
```

Payload is a weights-only torch dict containing `video_latents` [C,9,Hl,Wl],
`condition_latents` [C,1,Hl,Wl], **`source_identity=repo.identity`**, `text_context` [L,D], optional
`negative_text_context` [Lneg,D], `frame_ids` exactly raw_start..raw_start+32,
`condition_frame_id=raw_start-1`, `episode_index`, and the same `encoding` identity
dict as the manifest. Do NOT declare 9 frames without checking the actual VAE
output shape. Spatial latent sizes come from the actual encoder, not a guessed
downsampling ratio. Encode the prefix independently as one observed RGB frame.

The loader derives raw action/state tensors from the matching repository rather
than trusting duplicated target tensors in cache files. The first four action
slots are zero/masked; subsequent eight groups map raw actions [start,start+32).
The ninth raw anchor is start+32; its outgoing command is not a target in this
window. States are xyz/quaternion7, with per-latent anchors from the native
`latent_anchor_positions` helper, and observed-prefix state at start-1.

Statistics are fitted only by `fit_cine_action_statistics` on selected TRAIN
episodes. Persist the returned dict; validation consumes the SAME dict and never
fits its own statistics. Numeric IDs are scoped by physical source root, so train
episode 0 and validation episode 0 are allowed. Identical train/val roots are
rejected. This is not an exhaustive physical trajectory duplication audit.

## Native dataset interface

`build_cine_latent_train_val_datasets(data_config)` returns two native latent
datasets. Configure one color camera, action_dim=7/state_dim=7, frame_stride=1,
`data.num_frames=9`, and `data.action_schema.action_horizon=36` for the 33-frame
cache window. Model inference chunk density must remain 4 actions per latent;
the trainer must keep its native inference horizon/chunk geometry consistent.
The sample metadata explicitly declares action_tokens_per_frame=4 and the leading
masked action group. Native external-prefix conditioning remains separate, so
nine targets become TEN native model-visible video frames. Freeze:
`history_frames=1`, `loss_frame_start/end=1/9`,
`latent_loss_frame_start/end=0/9`, `action_loss_frame_start/end=1/9`,
`chunk_origin_frame=1`, `singleton_chunk_frame=0`, `frame_shift=1`.
Video target 0 remains supervised under the native external-prefix contract;
its dummy action group is masked. Frame shift is a latent-grid offset, not raw s.
Provide states at the true nine target anchors; the existing native helper
selects the preceding chunk boundary, never the current chunk's endpoint.
The accepted factory profile has decoder/data horizon 36 and inference chunk 9,
while training chunk size is separately configured. No factory patch is needed.

`adapter_options` accepts `cache_manifest` (default manifest.json),
`action_semantics` (raw_joint_command default; joint_delta is an explicit metadata
declaration only), and `action_normalization` (none default; gaussian reuses
manifest TRAIN statistics). No implicit differences or value transforms occur
under either action-semantics label. Use native action target representation raw.
The normalization mode must equal the manifest mode; prepare a cache declaring
gaussian before requesting gaussian, rather than changing the flag on a none cache.

Preparer should use the existing real Wan VAE/T5 encode hooks, save their actual
outputs under this schema, and then call the registered native dataset builder.
No raw dataset metadata or video is rewritten into a v2 layout.

## Shared text and source guards (implemented)

Each payload MUST carry `source_identity` exactly matching its physical split's
repository identity. Copying a train payload into same-numbered val episode is
rejected. Within each split, repeated `(episode_index,raw_start)`, repeated resolved
payload paths and hard-linked duplicate payloads are rejected.

For reusable padded text, replace each inline field with `text_context_path`
and/or `negative_text_context_path` (root-relative, mutually exclusive with the
corresponding inline tensor). Files contain the Tensor directly and use
`prompt_cache/embeddings/<sha256(UTF8_prompt)>.pt`; positive filename must match
the actual raw task prompt, negative filename must match the empty prompt. Path
payloads include `task_text` and/or `task_sha256` matching that actual prompt.
The dataset holds an eight-entry LRU and still emits native Tensor fields.
Inline tensors remain supported for small fixtures. All payloads retain encoding
identities, source identities and precise episode/frame/prefix IDs.

`validate_cine_split_sources(train_repo,val_repo,train_ids,val_ids)` is now public;
call before any VAE/T5 load. It rejects selected train/val episodes whose video
paths resolve to the SAME `(st_dev,st_ino)` with overlapping half-open metadata
time intervals. The native dataset builder reuses it. Different disjoint intervals
in a shared physical file are allowed. Root names/episode IDs are not evidence
of disjoint trajectories. This bounded check does not compare independently
copied identical video content. `repo.identity` hashes info/tasks plus ALL episode
metadata Parquets (including shared-video offsets and row intervals);
it is not a full-content audit or a raw-data immutability guarantee.

`manifest.complete=true` means the declared selected cache is complete, not that
the full source root was encoded. The preparer records its exact scope, episode
IDs, raw starts, window counts/stride and available source episode counts in
`selection`. The cached entries and TRAIN statistic episode selection remain
the authoritative set; smaller subsets must be labelled as such.

Selection declarations are mandatory. Each split's IDs, raw_starts, window_count
and available_episode_count must agree with actual entries and source metadata.
window_stride must equal data.sample_stride. Subset and full selections must use
the `1+k*window_stride` grid and remain inside the true episode bounds.
`all_windows` reconstructs every valid start from every source episode's true
length using window_stride; partial lists cannot claim complete-root coverage.
Every payload path is checked before sampling; absent later samples fail even
when numerical check-data is limited to the first sample. `complete` must be the
boolean True. Every `sources` identity and every payload `source_identity` includes
the new **episodes_sha256**, so a changed episode/video mapping invalidates cache.

The single authoritative selection implementation is now public:
`open_wam.data.cine_v3_latent.validate_cine_selection_manifest(manifest, repositories=None)`.
The native builder passes its already-open train/validation repository dict.
The application layer can lazily delegate with just the manifest; do not maintain
a second set of selection rules in `gradientwam.cine_data_contract`.

## Verification and ownership closure

Final targeted CPU command, with CUDA hidden and both Hub/Transformers offline:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
.venv-cpu/bin/python -m pytest \
  tests/test_cine_v3.py tests/test_cine_review_data.py \
  tests/test_cine_review_temporal.py tests/test_cine_preparation.py \
  tests/test_lerobot_local_latent_dataset.py tests/test_latent_temporal.py \
  tests/test_data_config_defaults.py -q
```

Result: **102 passed in 65.91s**, recorded in
`outputs/cine-v3-checks/infra-final-regression.log`. This is a targeted suite,
not a claim that every upstream test passed. Initial feature checks failed on
the missing reader. Subsequent independently reproduced cache/state/metadata
failures were fixed and are included in the final green run.

The actual adapter-generated samples were passed to a real tiny native pipeline
for training chunk sizes 2 and 4. Checks cover 10 model-visible video frames,
36 action slots with the first 4 masked, exact raw command ordering, and true
previous-chunk pose selection after independent prefix insertion. Perturbing the
final future endpoint state leaves predictions unchanged. CAGrad composition,
finite backward, clipping and an AdamW update execute on CPU. Independent tests
also exercise real small Wan VAE and local native frontend preparation through
the published cache/prompt-path interface. These are synthetic small-model tests,
not pretrained/full-width encoding or robot-control gains.

Bounded H20 read-only receipt is in
`outputs/cine-v3-checks/infra-h20-readonly.log`: one episode from the selected
training root and one from the separate validation root; four rows and two
returned RGB frames each. Both episodes have length 399, state7/action7 and 36
available task prompts. Original row timestamps begin 0,1/30,2/30,3/30;
returned RGB shape is [2,3,480,640]. CUDA remains uninitialized. The real sampled
episodes have video offset zero; nonzero shared-file offsets and cross-shard reads
are additionally verified by the synthetic real-format fixture.

No raw assets were written, no full data/model downloads or environment installs
were performed by this owner, and no GPU training was launched. Only small reader
code/check scripts were uploaded to a separate temporary root-disk folder for
the bounded read check. Metadata fingerprints and the inode/time guard are not
an exhaustive content-duplication audit across all scales.

Owned changes: `cine_v3.py`, `cine_v3_latent.py`, the new finite
`CineActionSemantics` enum, the `cine_v3_latent` registration in `latent_factory.py`,
`tests/test_cine_v3.py` and this handoff. Existing LIBERO implementation files were
not changed. PM owns preparation/public configuration and training owns runtime
integration. No branch switch, staging, commit, push or subagent was performed.

Format background: [official LeRobot v3 file-based dataset description](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).
The Cine-specific shape/clock/semantic constraints come from the approved data
contract and local evidence above, not from a generic v3 benchmark claim.
