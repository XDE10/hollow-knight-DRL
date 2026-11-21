"""Recover the historical teacher/student assignment directly from native game tapes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def records(root: Path, wanted: set[str]) -> dict[str, tuple[Path, int]]:
    result = {}
    for manifest in root.glob("development-*-student-*/manifest.json"):
        items = json.loads(manifest.read_bytes())
        if not isinstance(items, list):
            continue
        for i, item in enumerate(items):
            source = item["source"]
            if source not in wanted:
                continue
            receipt = json.loads(
                (manifest.parent / f"episode-{i:03d}.json").read_bytes()
            )
            if source != f"native-development/{receipt['seed']}":
                raise ValueError("Receipt seed differs from the development record")
            eid = int(receipt["episode_id"])
            native = (
                manifest.parent.with_name(manifest.parent.name + "-labels")
                / f"{eid}.native"
            )
            if source in result:
                raise ValueError(f"Duplicate raw tape for {source}")
            result[source] = (native, eid)
    if set(result) != wanted:
        raise ValueError("Missing original native recordings")
    return result


def recover_decisions(
    blob: bytes, episode: int, game: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if struct.unpack_from("<IQ", blob) != (0x32474144, episode):
        raise ValueError("Invalid timing tape header")
    rows = np.frombuffer(blob, "<f4", offset=12).reshape(-1, 259)
    if (
        not len(rows)
        or not np.isfinite(rows).all()
        or not np.isin(rows[:, 66], [0.0, 1.0]).all()
    ):
        raise ValueError("Invalid native trace")
    if not all(
        np.array_equal(rows[:, cols], game[key])
        for cols, key in (
            (0, "control_time"),
            (slice(57, 60), "controls"),
            (slice(60, 63), "teacher_controls"),
            (slice(67, None), "timing_state"),
        )
    ):
        raise ValueError("Native trace does not match the archived game")
    keys = rows[:, 63:66].astype("i2")
    if not np.array_equal(keys, rows[:, 63:66]):
        raise ValueError("Noninteger native student action")
    fields = {"teacher_acted": rows[:, 66].astype(bool), "student_controls": keys}
    check_assignment(game, fields)
    return fields


def check_assignment(
    game: dict[str, np.ndarray], fields: dict[str, np.ndarray]
) -> None:
    side, student = fields["teacher_acted"], fields["student_controls"]
    if (
        side.dtype != np.bool_
        or side.shape != game["control_time"].shape
        or student.shape != game["controls"].shape
    ):
        raise ValueError("Invalid controller assignments")
    if not np.array_equal(
        np.where(side[:, None], game["teacher_controls"], student), game["controls"]
    ):
        raise ValueError("Selected controller did not issue the executed action")


def packed(arrays: dict[str, np.ndarray]) -> bytes:
    buff = io.BytesIO()
    np.savez_compressed(buff, **arrays)
    return buff.getvalue()


def build(archive: Path, supplement: Path, output: Path, raw_root: Path | None) -> None:
    if output.exists() or (raw_root is not None and supplement.exists()):
        raise FileExistsError("Select fresh artifact paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    supplement.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(".pending")
    supplement_tmp = supplement.with_suffix(".pending")
    with zipfile.ZipFile(archive) as base:
        manifest_bytes = base.read("manifest.json")
        games = json.loads(manifest_bytes)
        ids = {g["source"] for g in games}
        if len(ids) != len(games) or any(
            not key.startswith("native-development/") for key in ids
        ):
            raise ValueError("Expected unique development fights")
        wanted = set()
        for g in games:
            with np.load(io.BytesIO(base.read(g["file"])), allow_pickle=False) as data:
                if "teacher_controls" in data and "teacher_acted" not in data:
                    wanted.add(g["source"])
        tapes = records(raw_root, wanted) if raw_root is not None else {}
        with (
            zipfile.ZipFile(
                supplement_tmp if raw_root is not None else supplement,
                "x" if raw_root is not None else "r",
            ) as patch,
            zipfile.ZipFile(output_tmp, "x") as dest,
        ):
            if raw_root is None:
                provenance = json.loads(patch.read("records.json"))
                if (
                    provenance["input_manifest_sha256"] != sha(manifest_bytes)
                    or set(provenance["records"]) != wanted
                ):
                    raise ValueError("Controller supplement belongs to another dataset")
            else:
                provenance = {
                    "input_manifest_sha256": sha(manifest_bytes),
                    "records": {},
                }
            for g in games:
                original = base.read(g["file"])
                if sha(original) != g["sha256"]:
                    raise ValueError("Damaged input replay")
                key = g["source"]
                if key not in wanted:
                    dest.writestr(g["file"], original)
                    continue
                with np.load(io.BytesIO(original), allow_pickle=False) as d:
                    data = dict(d)
                if raw_root is not None:
                    path, eid = tapes[key]
                    native = path.read_bytes()
                    addition = recover_decisions(native, eid, data)
                    sidecar = packed(addition)
                    patch.writestr(g["file"], sidecar)
                    provenance["records"][key] = {
                        "native_sha256": sha(native),
                        "input_sha256": sha(original),
                        "supplement_sha256": sha(sidecar),
                        "teacher_decisions": int(addition["teacher_acted"].sum()),
                        "rows": len(data["control_time"]),
                        "controller_handoffs": int(
                            np.count_nonzero(np.diff(addition["teacher_acted"]))
                        ),
                    }
                else:
                    entry = provenance["records"][key]
                    sidecar = patch.read(g["file"])
                    if (
                        sha(sidecar) != entry["supplement_sha256"]
                        or sha(original) != entry["input_sha256"]
                    ):
                        raise ValueError("Controller supplement failed checksum")
                    with np.load(io.BytesIO(sidecar), allow_pickle=False) as d:
                        addition = {
                            k: d[k] for k in ("teacher_acted", "student_controls")
                        }
                    check_assignment(data, addition)
                updated = packed({**data, **addition})
                dest.writestr(g["file"], updated)
                g["sha256"] = sha(updated)
                g["raw_native_sha256"] = provenance["records"][key]["native_sha256"]
            if raw_root is not None:
                patch.writestr("records.json", json.dumps(provenance, indent=2) + "\n")
            dest.writestr("manifest.json", json.dumps(games, indent=2) + "\n")
    if raw_root is not None:
        supplement_tmp.replace(supplement)
    output_tmp.replace(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("supplement", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--raw-root", type=Path)
    args = p.parse_args()
    build(args.archive, args.supplement, args.output, args.raw_root)
