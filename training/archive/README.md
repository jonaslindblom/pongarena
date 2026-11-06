# Completed experiment archive

`experiments-2026-11-06.tar.gz` preserves the pre-cleanup `training/results/`, `training/runs/`, `training/scripts/` and `training/README.md` with their original repository-relative paths. This includes the uncommitted study outputs and scripts present before cleanup. Python bytecode caches were excluded.

All 5,804 archived files were read back and checked against SHA-256 hashes before loose copies were removed. [manifest.json](manifest.json) records every member hash and the archive hash. Retained deployed weights and selected checkpoints have a separate current [catalog](../catalog.json).

To inspect old files, extract into a separate directory, not over the maintained workflow:

```sh
mkdir -p /tmp/rally-training-history
tar -xzf training/archive/experiments-2026-11-06.tar.gz -C /tmp/rally-training-history
```

The extracted README and scripts describe historical experiments and their original graph contracts. Some expect retired public model files or the application source from their recorded commit/snapshot. Use the frozen source and dependency records for reproduction; extracting this archive alone does not recreate every historical app environment. Retired public model bytes are retained in `tests/fixtures/models/`.

This archive is local and ignored along with the training folder. It has not been uploaded or backed up elsewhere. Do not delete it if you need the original uncommitted experiments or exact historical training provenance.
