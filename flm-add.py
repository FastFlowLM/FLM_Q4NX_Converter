#!/usr/bin/env python3
"""flm-add.py - install a pre-converted FLM (Q4NX) model and register it with FastFlowLM.

Standalone, Python-3 stdlib only (no pip packages). Works with any repo that
already contains the runtime-ready files (config.json, model.q4nx,
tokenizer.json, tokenizer_config.json, optionally chat_template.jinja):

    python3 flm-add.py Atomic-Germ/Qwen3.5-9B-Claude-4.8-Opus-NPU2

Repo can be a Hugging Face repo id, a ModelScope repo id (--modelscope), a full
Hugging Face URL, or a local directory holding the model files. The tag is
derived from the repo name (e.g. Qwen3.5-9B-Claude-4.8-Opus-NPU2 ->
qwen3.5-claude:9b); override with --tag. Defaults for the registry entry
(family, engine, size, context length) are copied from the matching official
FastFlowLM entry.

The script never rewrites the system model list or the system xclbins; it
writes a user-level registry at ~/.config/flm/model_list.json and adds a single
symlink into ~/.config/flm/xclbins/ for the new model directory. Custom FLM
models never ship xclbins (they are closed source), so the kernel symlink is
always taken from the matching official model, keyed by family (engine) and
size -- e.g. Darwin-36B-Opus-NPU2 -> Qwen3.6-35B-A3B-NPU2. The only thing
you need in your shell rc afterwards is:

    export FLM_CONFIG_PATH="$HOME/.config/flm/model_list.json"
    export FLM_XCLBIN_PATH="$HOME/.config/flm"
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REQUIRED_FILES = ["config.json", "model.q4nx", "tokenizer.json", "tokenizer_config.json"]
OPTIONAL_FILES = ["chat_template.jinja", "vision_weight.q4nx", "audio_weight.q4nx"]
ALL_FILES = REQUIRED_FILES + OPTIONAL_FILES

SYSTEM_LIST_CANDIDATES = [
    "/opt/fastflowlm/share/flm/model_list.json",
    "/usr/share/flm/model_list.json",
    "/usr/local/share/flm/model_list.json",
]

SYSTEM_XCLBIN_PREFIXES = [
    Path("/opt/fastflowlm/share/flm"),
    Path("/usr/share/flm"),
    Path("/usr/local/share/flm"),
]

# Dir-name prefix -> runtime details.family, used only when no official entry
# can be matched by name. The official model_list.json is the primary source.
FAMILY_ALIASES = [
    ("qwen3.5-omni", "qwen3.5-omni"),
    ("qwen3.6", "qwen3.6-moe"),
    ("qwen3.5", "qwen3.5"),
    ("qwen3", "qwen3"),
    ("qwen2.5vl", "qwen2.5vl"),
    ("qwen2.5", "qwen2"),
    ("qwen2vl", "qwen2vl"),
    ("qwen2", "qwen2"),
    ("gemma4", "gemma4e"),
    ("gemma3", "gemma3"),
    ("llama3", "llama3"),
    ("deepseek-r1-0528", "deepseek-r1-0528"),
    ("deepseek-r1", "deepseek-r1"),
    ("deepseek", "deepseek-r1"),
    ("nanbeige4", "nanbeige"),
    ("nanbeige", "nanbeige"),
    ("gpt-oss", "gpt-oss"),
    ("lfm2.5", "lfm2.5-tk"),
    ("lfm2", "lfm2"),
    ("phi4", "phi4"),
    ("whisper-v3", "whisper-v3"),
    ("whisper", "whisper-v3"),
    ("embed-gemma", "embed-gemma"),
]


def log(msg):
    print(msg, file=sys.stderr)


def err(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_system_model_list():
    exe = shutil.which("flm")
    candidates = []
    if exe:
        candidates.append(Path(exe).parent / "model_list.json")
    candidates += [Path(p) for p in SYSTEM_LIST_CANDIDATES]
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        "Could not locate the system model_list.json (looked next to `flm` and in "
        "/opt,/usr,/usr/local share/flm). Pass --system-list."
    )


def find_system_xclbin_root():
    """Directory whose <root>/xclbins/ holds the per-model kernel folders."""
    exe = shutil.which("flm")
    candidates = []
    if exe:
        candidates.append(Path(exe).parent)
        candidates.append((Path(exe).parent / ".." / "share" / "flm").resolve())
    candidates += SYSTEM_XCLBIN_PREFIXES
    for c in candidates:
        if (c / "xclbins").is_dir():
            return c / "xclbins"
    return None


def user_xclbin_dir(arg):
    """Resolve the user-level xclbins directory (where symlinks are added)."""
    if arg:
        base = Path(arg)
    else:
        env = os.environ.get("FLM_XCLBIN_PATH")
        base = Path(env) if env else Path.home() / ".config" / "flm"
    return base if base.name == "xclbins" else base / "xclbins"


def user_registry_path(arg):
    if arg:
        return Path(arg)
    env = os.environ.get("FLM_CONFIG_PATH")
    if env:
        return Path(env)
    return Path.home() / ".config" / "flm" / "model_list.json"


def models_root_dir(arg):
    if arg:
        return Path(arg)
    env = os.environ.get("FLM_MODEL_PATH")
    if env:
        return Path(env) / "models"
    return Path.home() / ".config" / "flm" / "models"


# ---------------------------------------------------------------- tag derivation

def _strip_npu2(name):
    return re.sub(r"-NPU2$", "", name, flags=re.IGNORECASE)


def _extract_size(bare):
    m = re.search(r"(\d+(?:\.\d+)?[Bb](?:-[A-Za-z]+\d+(?:\.\d+)?)*)", bare)
    if not m:
        return None, bare
    size = m.group(1).lower()
    rest = (bare[: m.start()] + " " + bare[m.end():]).strip()
    return size, rest


def derive_tag(dir_name, explicit=None):
    if explicit:
        return explicit
    size, rest = _extract_size(_strip_npu2(dir_name))
    if not size:
        raise SystemExit(
            f"Could not derive a size from '{dir_name}' (no 'NNb' marker). "
            "Pass --tag name:size."
        )
    tokens = [t for t in re.split(r"[-_ ]+", rest) if t]
    if not tokens:
        raise SystemExit("Could not derive a tag from the repo name. Pass --tag name:size.")
    family = tokens[0].lower()
    variant = None
    for t in tokens[1:]:
        if re.fullmatch(r"\d+(\.\d+)?[MmKk]?", t):
            continue
        variant = t.lower()
        break
    return f"{family}-{variant}:{size}" if variant else f"{family}:{size}"


def match_official_entry(system_registry, dir_name):
    """Official entry whose directory name shares the longest token prefix."""
    best = None
    dir_tokens = re.split(r"[-_ ]+", dir_name)
    for bucket, sizes in system_registry.get("models", {}).items():
        for size, info in sizes.items():
            name = info.get("name")
            if not name:
                continue
            common = 0
            for x, y in zip(re.split(r"[-_ ]+", name), dir_tokens):
                if x.lower() != y.lower():
                    break
                common += 1
            if common >= 2 and (best is None or common > best[0]):
                best = (common, bucket, size, info)
    return best


def _official_entries(system_registry, family):
    return [
        (bucket, sz, info)
        for bucket, sizes in system_registry.get("models", {}).items()
        for sz, info in sizes.items()
        if (info.get("details") or {}).get("family") == family
    ]


def match_official_by_family_size(system_registry, family, size):
    """Official entry matching details.family and registry size (bytes).

    Used for repos that share an engine with an official model but not a
    name prefix (e.g. Huihui-Qwythos-9B-... -> qwen3.5 + 9B -> Qwen3.5-9B-NPU2).
    """
    if not family or not size:
        return None
    for bucket, sz, info in _official_entries(system_registry, family):
        if info.get("size") == size:
            return (0, bucket, sz, info)
    return None


def resolve_official(system_registry, dir_name, family, size):
    """Pick the official model that supplies the xclbins for this install.

    Custom FLM models never ship xclbins (closed source), so the kernels must
    be linked from the matching official model, keyed by family (engine) and
    size. Returns (official_4tuple, note) where note explains any size
    mismatch, or (None, None) when no official model matches.
    """
    official = match_official_entry(system_registry, dir_name)
    if official:
        return official, None
    official = match_official_by_family_size(system_registry, family, size)
    if official:
        return official, None
    entries = _official_entries(system_registry, family)
    if len(entries) == 1:
        bucket, sz, info = entries[0]
        note = None
        if size:
            official_size = info.get("size", 0)
            if official_size and official_size != size:
                note = f"tag size {size/1e9:g}B differs from official {official_size/1e9:g}B"
        return (0, bucket, sz, info), note
    if entries and size:
        best = min(entries, key=lambda e: abs(e[2].get("size", 0) - size))
        bucket, sz, info = best
        return (0, bucket, sz, info), (
            f"no exact size match for {size/1e9:g}B; using {info.get('size', 0)/1e9:g}B kernels"
        )
    return None, None


def derive_family(system_registry, dir_name, explicit=None, base_entry=None):
    if explicit:
        return explicit
    if base_entry:
        fam = base_entry.get("details", {}).get("family")
        if fam:
            return fam
    lower = dir_name.lower()
    for prefix, family in FAMILY_ALIASES:
        if lower.startswith(prefix.lower()):
            return family
    raise SystemExit(
        f"Could not determine details.family for '{dir_name}'. "
        "Pass --family (e.g. qwen3.5, qwen3.6-moe, nanbeige, llama3, ...)."
    )


# ------------------------------------------------------------------- asset fetch

def _hf_headers():
    headers = {"User-Agent": "flm-add/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get_json(url):
    req = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hf_file_tree(repo_id):
    return _http_get_json(f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true")


def ms_file_tree(repo_id):
    return _http_get_json(
        f"https://modelscope.cn/api/v1/models/{repo_id}/repo/files?Revision=master&Recursive=false"
    )


def hf_cache_snapshot(repo_id):
    roots = []
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        if os.environ.get(env):
            p = Path(os.environ[env])
            roots.append(p if p.name == "hub" else p / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    repo_dir_name = "models--" + repo_id.replace("/", "--")
    for root in roots:
        snapshots = root / repo_dir_name / "snapshots"
        if not snapshots.is_dir():
            continue
        for ref in (root / repo_dir_name / "refs").glob("*"):
            try:
                rev = ref.read_text().strip()
            except Exception:
                continue
            d = snapshots / rev
            if d.is_dir():
                return d
        first = next((d for d in snapshots.iterdir() if d.is_dir()), None)
        if first:
            return first
    return None


def download_file(url, dest, expected_size=None, expected_sha=None, verify=True, quiet=False):
    req = urllib.request.Request(url, headers=_hf_headers())
    tmp = str(dest) + ".part"
    written = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        length = int(resp.headers.get("Content-Length") or 0)
        total = expected_size or length or 0
        last_pct = -1
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if total and not quiet:
                pct = int(written * 100 / total)
                if pct != last_pct and pct % 5 == 0:
                    log(f"    {pct:3d}% ({written/1e9:.2f} GB / {total/1e9:.2f} GB)")
                    last_pct = pct
    if expected_size and written != expected_size:
        os.unlink(tmp)
        raise SystemExit(f"Size mismatch for {dest.name}: got {written}, expected {expected_size}")
    if expected_sha and verify:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        if h.hexdigest() != expected_sha:
            os.unlink(tmp)
            raise SystemExit(f"sha256 mismatch for {dest.name}")
    os.replace(tmp, dest)


def fetch_assets(repo_id, target, modelscope=False, verify=True, force=False, quiet=False):
    """Populate target/ with the model files; returns the list of files present."""
    obtained = []
    if modelscope:
        tree = ms_file_tree(repo_id)
        entries = {}
        for e in tree.get("Data", []):
            p = e.get("Path") or e.get("Name")
            if p and "/" not in p:
                entries[p] = {}
        for fname in ALL_FILES:
            if fname not in entries:
                continue
            dest = target / fname
            if dest.is_file() and not force:
                obtained.append(fname)
                continue
            log(f"Downloading {fname} from ModelScope...")
            download_file(
                f"https://modelscope.cn/models/{repo_id}/resolve/master/{fname}",
                dest,
                verify=verify,
                quiet=quiet,
            )
            obtained.append(fname)
        return obtained

    # Hugging Face: local cache first, then the tree API.
    entries = {}
    for e in hf_file_tree(repo_id):
        p = e.get("path")
        if p and "/" not in p:
            entries[p] = e
    for fname in ALL_FILES:
        if fname not in entries:
            continue
        dest = target / fname
        if dest.is_file() and not force:
            obtained.append(fname)
            continue
        lfs = entries[fname].get("lfs") or {}
        expected_sha = lfs.get("oid")
        expected_size = lfs.get("size") or entries[fname].get("size")
        log(f"Downloading {fname} ({expected_size/1e9:.2f} GB)...")
        download_file(
            f"https://huggingface.co/{repo_id}/resolve/main/{fname}",
            dest,
            expected_size=expected_size,
            expected_sha=expected_sha,
            verify=verify,
            quiet=quiet,
        )
        obtained.append(fname)
    return obtained


def copy_from_dir(src_dir, target, force=False):
    obtained = []
    for fname in ALL_FILES:
        src = src_dir / fname
        if src.is_file():
            dest = target / fname
            if dest.is_file() and not force:
                obtained.append(fname)
                continue
            shutil.copy2(src, dest)
            obtained.append(fname)
    return obtained


# ---------------------------------------------------------------- registry

def size_from_tag(tag):
    """Registry 'size' (bytes) from the tag size marker, e.g. '3b' -> 3000000000,
    '9b-claude-4.8' -> 9000000000, '0.8b' -> 800000000."""
    m = re.match(r".*:(\d+(?:\.\d+)?)b\b", tag, flags=re.IGNORECASE)
    if not m:
        return None
    return int(float(m.group(1)) * 1_000_000_000)


def estimate_size(config_path):
    try:
        cfg = load_json(config_path)
    except Exception:
        return None
    hidden = cfg.get("hidden_size")
    layers = cfg.get("num_hidden_layers")
    if not hidden or not layers:
        return None
    intermediate = cfg.get("intermediate_size")
    per_layer = 12 * hidden * hidden
    if intermediate:
        per_layer += 3 * hidden * intermediate
    total = per_layer * layers + 2 * hidden * (cfg.get("vocab_size") or hidden)
    return max(int(round(total / 1e9 * 2) / 2 * 1e9), 1_000_000_000)


def build_entry(base_entry, dir_name, files, size):
    entry = dict(base_entry) if base_entry else {}
    entry["name"] = dir_name
    entry["files"] = list(files)
    entry["url"] = ""
    entry["file_url"] = ""
    entry["ms_url"] = ""
    entry.setdefault("max_prefill_len", 4096)
    entry.setdefault("default_context_length", 8192)
    entry.setdefault("flm_min_version", "0.9.45")
    entry.setdefault("details", {}).setdefault("format", "NPU2")
    if size:
        entry["size"] = size
    entry["vlm"] = any(f.startswith("vision") for f in files)
    return entry


def register(user_list_path, tag, entry, system_registry):
    if user_list_path.is_file():
        registry = load_json(user_list_path)
    else:
        registry = json.loads(json.dumps(system_registry))
    registry.setdefault("model_path", "models")
    model_type, size = tag.split(":", 1)
    registry.setdefault("models", {}).setdefault(model_type, {})[size] = entry
    save_json(user_list_path, registry)


# ------------------------------------------------------------------- xclbins

def link_xclbins(system_root, user_root, dir_name, source_name, force=False, quiet=False):
    if not source_name:
        if not quiet:
            log("[WARN] No xclbin source; skipping symlink. Pass --xclbin-from NAME to link an official model's kernels.")
        return
    src = system_root / source_name
    if not src.is_dir():
        if not quiet:
            log(f"[WARN] Official model has no xclbins directory: {source_name}")
        return
    user_root.mkdir(parents=True, exist_ok=True)
    link = user_root / dir_name
    target = str(src)
    if link.is_symlink():
        if os.readlink(link) == target:
            if not quiet:
                log(f"[INFO] xclbins link already in place: {link}")
            return
        link.unlink()
    elif link.exists():
        if force:
            shutil.rmtree(link)
        else:
            raise SystemExit(
                f"{link} already exists and is not a symlink. Remove it or pass --force."
            )
    os.symlink(target, link)
    if not quiet:
        log(f"[INFO] Linked xclbins: {link} -> {target}")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        prog="flm-add.py",
        description="Install a pre-converted FLM (Q4NX) model and register it with FastFlowLM.",
    )
    ap.add_argument("repo", help="Hugging Face repo id (Org/Name), ModelScope id (with --modelscope), URL, or local directory")
    ap.add_argument("--tag", help="Registry tag (default: derived from the repo name, e.g. qwen3.5-claude:9b)")
    ap.add_argument("--family", help="details.family for engine dispatch (default: from matching official entry)")
    ap.add_argument("--config", help="model_list.json to update (default: $FLM_CONFIG_PATH or ~/.config/flm/model_list.json)")
    ap.add_argument("--models-root", help="models directory (default: $FLM_MODEL_PATH or ~/.config/flm/models)")
    ap.add_argument("--xclbin-dir", help="user xclbins directory (default: ~/.config/flm/xclbins)")
    ap.add_argument("--xclbin-from", help="official model directory name to link xclbins from (default: best match, e.g. Qwen3.6-35B-A3B-NPU2)")
    ap.add_argument("--system-list", help="official model_list.json used for defaults (default: auto-detect)")
    ap.add_argument("--modelscope", action="store_true", help="Treat REPO as a ModelScope repo id")
    ap.add_argument("--no-xclbin", action="store_true", help="Do not create the xclbins symlink")
    ap.add_argument("--no-verify", action="store_true", help="Skip sha256 verification of downloads")
    ap.add_argument("--force", action="store_true", help="Overwrite existing model files/links")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    ap.add_argument("--quiet", action="store_true", help="Less output")
    args = ap.parse_args()

    repo = args.repo
    if repo.startswith(("https://", "http://")):
        repo = repo.split("/", 3)[-1]
    local_dir = Path(repo) if os.path.isdir(repo) else None
    dir_name = local_dir.name if local_dir else repo.split("/")[-1]
    if not dir_name:
        raise SystemExit("Could not determine a model directory name from the repo.")

    system_list = find_system_model_list()
    system_registry = load_json(system_list)
    user_list = user_registry_path(args.config)
    models_root = models_root_dir(args.models_root)
    target = models_root / dir_name

    tag = derive_tag(dir_name, args.tag)
    bucket, size_token = tag.split(":", 1)
    official = match_official_entry(system_registry, dir_name)
    base_entry = official[3] if official else None
    family = derive_family(system_registry, dir_name, args.family, base_entry)
    size_value = (base_entry or {}).get("size") or size_from_tag(tag)
    official, official_note = resolve_official(system_registry, dir_name, family, size_value)
    base_entry = official[3] if official else None
    src_tag = f"{official[1]}:{official[2]}" if official else None
    xclbin_source = args.xclbin_from or (base_entry or {}).get("name")
    if not args.dry_run:
        if official:
            note = f" ({official_note})" if official_note else ""
            log(f"[INFO] xclbins from official {src_tag}{note}")
        else:
            log("[WARN] No official model matched; no xclbins link. Pass --xclbin-from NAME (or --no-xclbin).")

    if args.dry_run:
        print(f"repo directory : {dir_name}")
        print(f"tag            : {tag}")
        print(f"details.family : {family}")
        print(f"official match : {src_tag or '(none)'}")
        print(f"xclbin source  : {xclbin_source or '(none)'}")
        print(f"models dir     : {target}")
        print(f"registry       : {user_list}")
        return

    # --- acquire model files ---
    if local_dir:
        if not args.quiet:
            log(f"[INFO] Using local model directory: {local_dir}")
        target.mkdir(parents=True, exist_ok=True)
        files = copy_from_dir(local_dir, target, force=args.force)
    else:
        snapshot = None if args.modelscope else hf_cache_snapshot(repo)
        if snapshot:
            if not args.quiet:
                log(f"[INFO] Found local HF cache: {snapshot}")
            target.mkdir(parents=True, exist_ok=True)
            files = copy_from_dir(snapshot, target, force=args.force)
        else:
            if not args.quiet:
                log(f"[INFO] Downloading model files from {'ModelScope' if args.modelscope else 'Hugging Face'}: {repo}")
            target.mkdir(parents=True, exist_ok=True)
            files = fetch_assets(repo, target, args.modelscope, verify=not args.no_verify, force=args.force, quiet=args.quiet)

    missing = [f for f in REQUIRED_FILES if not (target / f).is_file()]
    if missing:
        raise SystemExit(f"Model is missing required files: {missing}")

    if not size_value:
        size_value = estimate_size(target / "config.json")
    entry = build_entry(base_entry, dir_name, files, size_value)
    entry.setdefault("details", {})["family"] = family

    register(user_list, tag, entry, system_registry)
    log(f"[INFO] Registered tag '{tag}' in {user_list}")

    if not args.no_xclbin:
        system_root = find_system_xclbin_root()
        if system_root is None:
            log("[WARN] Could not locate system xclbins; skipped symlink.")
        else:
            link_xclbins(
                system_root,
                user_xclbin_dir(args.xclbin_dir),
                dir_name,
                xclbin_source,
                force=args.force,
                quiet=args.quiet,
            )

    print()
    print(f"Done: {dir_name} installed to {target}")
    print(f"Run:  flm run {tag}   (or: flm serve {tag})")
    print()
    print("Make sure your shell has these exports (add to ~/.bashrc):")
    print('    export FLM_CONFIG_PATH="$HOME/.config/flm/model_list.json"')
    print('    export FLM_XCLBIN_PATH="$HOME/.config/flm"')


if __name__ == "__main__":
    main()
