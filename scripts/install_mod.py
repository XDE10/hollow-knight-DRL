#!/usr/bin/env python3
"""One-command HKRLEnvMod installer for Hollow Knight (Windows & Linux).

Auto-discovers Steam's Hollow Knight installation across any drive via the
Windows Registry + Steam libraryfolders.vdf (or Linux Steam paths), downloads
the prebuilt HKRLEnvMod.dll from GitHub Releases with SHA-256 verification,
and installs it into hollow_knight_Data/Managed/Mods/HKRLEnvMod/.

Usage:
    python scripts/install_mod.py
    python scripts/install_mod.py --game-root "/path/to/Hollow Knight"
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
MOD_RELEASE_URL = (
    "https://github.com/XDE10/hollow-knight-DRL/releases/download/"
    "v1.0.0-sisters-champion/HKRLEnvMod.dll"
)
MOD_RELEASE_SHA256 = "3e568d950fc3b71898fe4d1bd64806f84e22f107f1cc6d399e0f4fc3e9245ffe"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_steam_library_paths(vdf_path: Path) -> list[Path]:
    """Parse Steam's libraryfolders.vdf and return all Steam library root directories."""
    if not vdf_path.is_file():
        return []
    text = vdf_path.read_text(encoding="utf-8", errors="ignore")
    roots: list[Path] = []
    for match in re.finditer(r'"path"\s+"([^"]+)"', text):
        raw = match.group(1).replace("\\\\", "\\")
        p = Path(raw)
        if p not in roots:
            roots.append(p)
    return roots


def _windows_steam_roots() -> list[Path]:
    roots: list[Path] = []
    if os.name != "nt":
        return roots
    try:
        import winreg  # type: ignore[import-not-found]

        queries = [
            (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
        ]
        for hive, subkey, value_name in queries:
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    val, _ = winreg.QueryValueEx(key, value_name)
                    if isinstance(val, str) and val.strip():
                        p = Path(val.strip())
                        if p not in roots:
                            roots.append(p)
            except OSError:
                continue
    except Exception:
        pass
    return roots


def _has_game_exe(path: Path) -> bool:
    return (path / "hollow_knight.exe").is_file() or (path / "hollow_knight.x86_64").is_file()


def discover_hollow_knight(explicit: Path | None = None) -> Path:
    """Locate Hollow Knight installation across any drive without hardcoded paths."""
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not _has_game_exe(resolved):
            raise FileNotFoundError(f"Hollow Knight executable not found in: {resolved}")
        return resolved

    env_root = os.environ.get("HKRL_GAME_ROOT", "").strip()
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        if _has_game_exe(cand):
            return cand

    steam_roots: list[Path] = []
    steam_roots.extend(_windows_steam_roots())
    steam_roots.extend(
        [
            Path.home() / ".local/share/Steam",
            Path.home() / ".steam/steam",
            Path.home() / "Library/Application Support/Steam",
        ]
    )

    library_roots: list[Path] = []
    for sr in steam_roots:
        if sr.is_dir():
            if sr not in library_roots:
                library_roots.append(sr)
            for lib in parse_steam_library_paths(sr / "steamapps/libraryfolders.vdf"):
                if lib not in library_roots:
                    library_roots.append(lib)

    for lib in library_roots:
        game = lib / "steamapps/common/Hollow Knight"
        if _has_game_exe(game):
            return game.resolve()

    raise FileNotFoundError(
        "Could not auto-detect Hollow Knight via Steam. "
        "Please pass --game-root \"/path/to/Hollow Knight\" or set HKRL_GAME_ROOT."
    )


def install_mod(game_root: Path, *, local_dll: Path | None = None, force: bool = False) -> Path:
    managed = game_root / "hollow_knight_Data/Managed"
    if not managed.is_dir():
        raise FileNotFoundError(f"Managed directory missing: {managed}")

    # Verify HK Modding API is installed (MMHOOK_Assembly-CSharp.dll is created by Modding API)
    if not (managed / "MMHOOK_Assembly-CSharp.dll").is_file():
        print(
            "[WARN] MMHOOK_Assembly-CSharp.dll was not found in Managed/.\n"
            "       Make sure the Hollow Knight Modding API is installed first (e.g. via Lumafly).\n"
            "       See docs/setup_guide.md for the 2-minute setup instructions.",
            flush=True,
        )

    dst = managed / "Mods/HKRLEnvMod/HKRLEnvMod.dll"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_file() and not force and sha256_file(dst) == MOD_RELEASE_SHA256:
        print(f"[HKRL] HKRLEnvMod.dll is already up to date at {dst}", flush=True)
        return dst

    # Prefer local freshly-built DLL if provided or already built in repo
    repo_dll = local_dll or (REPO_ROOT / "mod/HKRLEnvMod/bin/Release/HKRLEnvMod.dll")
    if repo_dll.is_file() and (local_dll is not None or sha256_file(repo_dll) == MOD_RELEASE_SHA256):
        shutil.copy2(repo_dll, dst)
        print(f"[HKRL] Installed local HKRLEnvMod.dll -> {dst} (sha256={sha256_file(dst)[:12]})", flush=True)
        return dst

    tmp = dst.with_suffix(".dll.download")
    print(f"[HKRL] Downloading prebuilt HKRLEnvMod.dll from GitHub Releases -> {dst} ...", flush=True)
    urllib.request.urlretrieve(MOD_RELEASE_URL, tmp)
    digest = sha256_file(tmp)
    if digest != MOD_RELEASE_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded HKRLEnvMod.dll SHA-256 mismatch: {digest} != {MOD_RELEASE_SHA256}")
    tmp.replace(dst)
    print(f"[HKRL] Installed verified HKRLEnvMod.dll -> {dst} (sha256={digest[:12]})", flush=True)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Install prebuilt HKRLEnvMod.dll into Hollow Knight.")
    ap.add_argument("--game-root", type=Path, default=None, help="Path to Hollow Knight directory (auto-detected if omitted)")
    ap.add_argument("--dll", type=Path, default=None, help="Optional local HKRLEnvMod.dll path instead of downloading")
    ap.add_argument("--force", action="store_true", help="Overwrite existing HKRLEnvMod.dll even if hash matches")
    args = ap.parse_args()

    game_root = discover_hollow_knight(args.game_root)
    print(f"[HKRL] Detected Hollow Knight at: {game_root}", flush=True)
    install_mod(game_root, local_dll=args.dll, force=args.force)


if __name__ == "__main__":
    main()
