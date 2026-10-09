# Provenance and modifications

This is an independent GradientWAM modified distribution of OpenWAM.

- Upstream: https://github.com/OpenWAM/OpenWAM, base commit `4b3814a82268f3523df5fcdadcbeb2c021ac7737`.
- The delivered native source includes the round02 overlay (routing, evidence loss, public checkpoint loader, offline prompt cache and preparation fixes); it is not claimed to be an untouched upstream checkout.
- `native_source_manifest.json` records hashes of every native Python file. These files were copied unchanged from the round02 integration snapshot.
- The frozen `algorithm/method_contract_v02.md` has SHA256 `cb2f30413d8eedd9f307445ae603b4dbf0dffae4275957b39eb877af7ef6d64f`. It is a research specification with historical audit references, not an assertion of full-model validation.
- `src/gradientwam` adds portable explicit-path orchestration. It reuses the native factory, method attachment, loss, optimizer, scheduler and full-state checkpoint helper. Local resource leases, private receipts and deployment addresses are not runtime inputs.
- `checkpoint.py` preserves the reviewed helper implementation (original SHA256 `7d1bf3dc9a6a467d98b5c1e7f97e33b7d16721909e4a109dc5542c03271fd3f7`); only its module docstring was adapted.
- `algorithm/operator_v02.py` resolves native code from this checkout. The unrelated external FAMO probe was omitted; the weighted-loss, visibility and sensitivity checks remain. It does not require a private literature directory.
- Package metadata/entrypoint, four-arm configs, portable tests and documentation are delivery additions.

## Included fixtures

`fixture_allowlist.json` lists each retained small upstream synthetic safetensors regression fixture and two 86-byte text checkpoint placeholders by size and hash. They are deterministic test assets, not pretrained model weights, user demonstrations or training checkpoints. All matched the source snapshot. No real weights, raw data, videos, PDFs, private credentials or execution receipts are included.

## License

OpenWAM Team, Stanford Vision and Learning Lab (SVL), Stanford University. OpenWAM, version 0.2.0, 2026. https://github.com/OpenWAM/OpenWAM

Preserve `LICENSE` (AGPL-3.0-only), `NOTICE`, `LICENSES/`, `THIRD_PARTY_NOTICES.md` and `CITATION.bib`. This derivative does not claim official OpenWAM/Stanford authorship or endorsement. External model/dataset licenses apply separately.
