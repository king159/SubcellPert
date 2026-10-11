"""Validate a downloaded observation and frozen benchmark input bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from subcellpert.data.splits import SCHEDULE_MANIFEST, VALIDATION_MANIFEST, load_seed_specs
from subcellpert.data.storage import CELL_MAJOR_ARRAY_KEYS, load_population_index_manifest


EXPORT_SCHEMA = "portable_frozen_benchmark_v1"


def _json(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(root, relative):
    relative = Path(relative)
    path = root / relative
    if relative.is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Release path leaves its root: {relative}")
    return path


def _view_directories(root):
    validation = _json(root / VALIDATION_MANIFEST)
    return sorted(
        {
            relative
            for fits in validation["datasets"].values()
            for fit in fits.values()
            for relative in fit["index_views"].values()
        }
    )


def validate_public_release(root, *, verify_hashes=False):
    root = Path(root).resolve()
    release = _json(root / "manifest.json")
    portable = release.get("benchmark_inputs", {}).get("schema") == EXPORT_SCHEMA
    reports = []
    for entry in release["datasets"]:
        population = load_population_index_manifest(_inside(root, entry["population"]), expected_dataset=entry["id"])
        paths = [p for p in (root / "datasets" / entry["id"]).rglob("*") if p.is_file()]
        reports.append(
            {
                "dataset": entry["id"],
                "files": len(paths),
                "size_bytes": sum(p.stat().st_size for p in paths),
                "cells": population["totals"]["declared_cells"],
            }
        )
    instances = 0
    for code in _json(root / SCHEDULE_MANIFEST)["datasets"]:
        instances += len(load_seed_specs(root, code))
    if portable:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Portable release contains a symbolic link: {path}")
        for relative in _view_directories(root):
            view = load_population_index_manifest(_inside(root, relative))
            for shard in view["shards"]:
                for key in CELL_MAJOR_ARRAY_KEYS:
                    target = (_inside(root, relative) / shard[key]).resolve()
                    if not target.is_relative_to(root / "datasets" / view["dataset"] / "population"):
                        raise ValueError(f"View does not share its packaged dataset array: {target}")
        declared = set(release["publication_files"])
        for tree in release["publication_trees"]:
            declared.update(p.relative_to(root).as_posix() for p in _inside(root, tree).rglob("*") if p.is_file())
        # snapshot_download(local_dir=...) adds local transfer metadata, not
        # benchmark inputs. Keep every other unexpected file an error.
        actual = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).parts[:2] != (".cache", "huggingface")
        }
        if declared != actual:
            raise ValueError(f"Publication inventory differs: {sorted(declared ^ actual)[:8]}")
        checksums = {}
        for line in (root / release["checksums_file"]).read_text().splitlines():
            digest, relative = line.split("  ", 1)
            path = _inside(root, relative)
            if relative in checksums or not path.is_file():
                raise ValueError(f"Invalid checksum entry: {relative}")
            checksums[relative] = digest
            if verify_hashes and _sha256(path) != digest:
                raise ValueError(f"Checksum differs: {relative}")
        if set(checksums) != actual - {release["checksums_file"]}:
            raise ValueError("Checksum inventory does not cover the release")
    return {
        "status": "valid",
        "portable": portable,
        "datasets": reports,
        "materialized_datasets": len(reports),
        "pending_datasets": 0,
        "split_instances": instances,
        "files": sum(row["files"] for row in reports),
        "size_bytes": sum(row["size_bytes"] for row in reports),
    }
