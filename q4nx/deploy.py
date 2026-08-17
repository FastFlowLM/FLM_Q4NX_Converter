"""Deploy converted models into the FLM runtime and register them.

FLM resolves ``flm run <name>:<size>`` through ``model_list.json``: the tag maps to
an entry whose ``name`` is the directory inside ``<models_root>/models/``, and whose
``details.family`` selects the runtime engine (``modelFamilyMap``). The weights are
loaded from ``config.json`` in that directory.

Because the system registry (``/opt/fastflowlm/share/flm/model_list.json``) is
root-owned, we never write to it. Instead the converted model is copied into the
user's models directory and registered in a user-level copy of the registry at
``~/.config/flm/model_list.json``. Point ``FLM_CONFIG_PATH`` at that file so FLM
picks it up:

    export FLM_CONFIG_PATH=$HOME/.config/flm/model_list.json

The new entry is derived from the official entry for the source model (same
``details.family`` so the same engine is dispatched, same ``size`` for the memlock
reservation, same ``max_prefill_len`` which the runner reads unconditionally) but with
``url``/``file_url``/``ms_url`` left empty (it is a local model; several FLM code
paths read them as strings) and ``files`` set to exactly what was deployed so
``is_model_downloaded`` never triggers a download.
"""
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import ModelArch
from .arch_detect import ARCH_TO_FAMILY

ARCH_TO_SIZE: Dict[ModelArch, str] = {
    ModelArch.QWEN35_08B: "0.8b",
    ModelArch.QWEN35_2B: "2b",
    ModelArch.QWEN35_4B: "4b",
    ModelArch.QWEN35_9B: "9b",
    ModelArch.QWEN35MOE: "35b-a3b",
    ModelArch.QWEN3: "0.6b",
    ModelArch.QWEN3VL: "4b",
    ModelArch.QWEN2: "3b",
    ModelArch.QWEN2VL: "3b",
    ModelArch.GEMMA3: "4b",
    ModelArch.GEMMA4: "2b",
    ModelArch.LLAMA: "8b",
    ModelArch.LFM2: "1.2b",
    ModelArch.PHI4: "4b",
    ModelArch.GPT_OSS: "20b",
    ModelArch.NANBEIGE: "3b",
}

MODEL_FILES = [
    "config.json",
    "model.q4nx",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vision_weight.q4nx",
    "audio_weight.q4nx",
]

_INSTALL_PREFIX_CANDIDATES = [
    "/opt/fastflowlm/share/flm/model_list.json",
    "/usr/share/flm/model_list.json",
    "/usr/local/share/flm/model_list.json",
]

_XCLBIN_PREFIX_CANDIDATES = [
    Path("/opt/fastflowlm/share/flm"),
    Path("/usr/share/flm"),
    Path("/usr/local/share/flm"),
]


def _tag_parts(tag: str) -> Tuple[str, str]:
    tag = tag.strip().lstrip("/")
    if ":" in tag:
        name, size = tag.split(":", 1)
    else:
        name, size = tag, ""
    return name, size


def find_flm_executable() -> Optional[str]:
    return shutil.which("flm")


def find_system_model_list() -> Path:
    """Locate the model_list.json FLM currently resolves (FLM_CONFIG_PATH first)."""
    env = os.environ.get("FLM_CONFIG_PATH")
    if env and Path(env).is_file():
        return Path(env)
    exe = find_flm_executable()
    if exe:
        candidate = Path(exe).parent / "model_list.json"
        if candidate.is_file():
            return candidate
    for candidate in _INSTALL_PREFIX_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "Could not locate FLM's model_list.json. Set FLM_CONFIG_PATH to the file "
        "FLM reads before deploying."
    )


def get_models_root() -> Path:
    """Directory holding per-model folders, mirroring utils::get_models_directory."""
    base = Path(os.environ["FLM_MODEL_PATH"]) if os.environ.get("FLM_MODEL_PATH") else Path.home() / ".config" / "flm"
    return base / "models"


def get_user_registry_path() -> Path:
    """Where the user-level registry lives. Honors an already-set FLM_CONFIG_PATH."""
    env = os.environ.get("FLM_CONFIG_PATH")
    if env:
        return Path(env)
    return get_models_root().parent / "model_list.json"


def find_system_xclbin_root() -> Optional[Path]:
    """The system ``<prefix>/xclbins`` directory, mirroring utils::find_xclbin_path.

    FLM's model libraries resolve kernels as ``<xclbin_root>/<model_dir>/layer.xclbin``
    where ``<xclbin_root>`` is the directory that contains an ``xclbins/`` folder.
    """
    env = os.environ.get("FLM_XCLBIN_PATH")
    if env:
        path = Path(env)
        if path.name == "xclbins":
            path = path.parent
        if (path / "xclbins").is_dir():
            return path / "xclbins"
    exe = find_flm_executable()
    if exe and (Path(exe).parent / "xclbins").is_dir():
        return Path(exe).parent / "xclbins"
    for prefix in _XCLBIN_PREFIX_CANDIDATES:
        if (prefix / "xclbins").is_dir():
            return prefix / "xclbins"
    return None


def _symlink(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == str(target):
            return
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    os.symlink(target, link)


def link_model_xclbins(model_dir_name: str, source_dir_name: str) -> Optional[Path]:
    """Give a custom model directory its own xclbin directory, pointing at its parent's.

    The runtime derives the xclbin folder from the model's directory name, so a
    finetune deployed under its own name needs ``<xclbin_root>/<model_dir_name>/``.
    We maintain a user-level ``~/.config/flm/xclbins/`` that mirrors every system
    model's xclbins (so official models keep working under the FLM_XCLBIN_PATH
    override) and symlink the custom model's folder to its parent model's.
    """
    system_root = find_system_xclbin_root()
    if system_root is None or not source_dir_name:
        return None
    user_root = get_models_root().parent / "xclbins"
    user_root.mkdir(parents=True, exist_ok=True)
    for entry in system_root.iterdir():
        if entry.is_dir():
            _symlink(entry, user_root / entry.name)
    parent_dir = system_root / source_dir_name
    if not parent_dir.is_dir():
        print(f"[WARN] Parent model has no xclbins directory: {source_dir_name}")
        return None
    _symlink(parent_dir, user_root / model_dir_name)
    return user_root


def default_model_dir_name(tag: str) -> str:
    """Derive an official-style directory name, e.g. qwen3.5-claude:9b -> Qwen3.5-Claude-9B-NPU2."""
    name, size = _tag_parts(tag)
    cap_name = "-".join(part[:1].upper() + part[1:] for part in name.split("-") if part)
    if not size:
        return f"{cap_name}-NPU2"
    size_cap = "".join(
        chunk.upper() if chunk.isalpha() else chunk
        for chunk in re.findall(r"[a-zA-Z]+|\d+\.?\d*", size)
    )
    return f"{cap_name}-{size_cap}-NPU2"


def deployed_files_in(output_dir: Path) -> List[str]:
    return [name for name in MODEL_FILES if (output_dir / name).is_file()]


def find_source_entry(
    registry: dict, deploy_from: Optional[str], model_arch: ModelArch
) -> Tuple[Optional[str], dict]:
    """Find the official registry entry to derive defaults from.

    Prefers an explicit ``deploy_from`` tag, then auto-detects from the model
    architecture (family + preferred size, falling back to the family's largest
    entry).
    """
    models = registry.get("models", {})
    if deploy_from:
        name, size = _tag_parts(deploy_from)
        bucket = models.get(name)
        if not bucket:
            return None, {}
        if size in bucket:
            return f"{name}:{size}", bucket[size]
        first = next(iter(bucket.items()))
        return f"{name}:{first[0]}", first[1]
    family = ARCH_TO_FAMILY.get(model_arch)
    if not family or family not in models:
        return None, {}
    bucket = models[family]
    if not bucket:
        return None, {}
    preferred = ARCH_TO_SIZE.get(model_arch)
    if preferred in bucket:
        return f"{family}:{preferred}", bucket[preferred]
    largest = max(bucket, key=lambda k: bucket[k].get("size", 0))
    return f"{family}:{largest}", bucket[largest]


def _estimate_params(config_path: Path) -> Optional[int]:
    """Rough parameter count from config.json, used only as a last resort."""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
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


def build_registry_entry(
    base: dict, model_dir_name: str, deployed_files: List[str], size: Optional[int]
) -> dict:
    entry = dict(base) if base else {}
    entry["name"] = model_dir_name
    entry["files"] = list(deployed_files)
    entry["url"] = ""
    entry["file_url"] = ""
    entry["ms_url"] = ""
    if isinstance(size, int) and size > 0:
        entry["size"] = size
    entry.setdefault("max_prefill_len", 4096)
    entry.setdefault("default_context_length", 8192)
    details = entry.setdefault("details", {})
    details.setdefault("format", "NPU2")
    entry["vlm"] = any(name.startswith("vision") for name in deployed_files)
    return entry


def _write_user_registry(registry: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def register_model(tag: str, entry: dict, user_registry_path: Path, system_list: Path) -> None:
    if user_registry_path.is_file():
        with open(user_registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        with open(system_list, encoding="utf-8") as f:
            registry = json.load(f)
    models = registry.setdefault("models", {})
    bucket = models.setdefault(_tag_parts(tag)[0], {})
    bucket[_tag_parts(tag)[1]] = entry
    registry.setdefault("model_path", "models")
    _write_user_registry(registry, user_registry_path)


def deploy_model(
    output_dir,
    tag: str,
    model_arch: ModelArch,
    model_dir_name: Optional[str] = None,
    deploy_from: Optional[str] = None,
) -> dict:
    """Copy the assembled model into FLM's models dir and register the tag.

    Returns a dict with the tag, target directory, source entry tag (if any) and
    the user registry path.
    """
    output_dir = Path(output_dir)
    system_list = find_system_model_list()
    with open(system_list, encoding="utf-8") as f:
        system_registry = json.load(f)

    src_tag, base_entry = find_source_entry(system_registry, deploy_from, model_arch)

    name, size = _tag_parts(tag)
    if model_dir_name is None:
        model_dir_name = default_model_dir_name(tag)

    target = get_models_root() / model_dir_name
    files = deployed_files_in(output_dir)
    if not files:
        raise FileNotFoundError(f"No deployable model files found in {output_dir}")

    if target.exists():
        print(f"[WARN] Overwriting existing model directory: {target}")
    os.makedirs(target, exist_ok=True)
    for filename in files:
        shutil.copy2(output_dir / filename, target / filename)

    size_value = base_entry.get("size") if base_entry else None
    if size_value is None:
        size_value = _estimate_params(target / "config.json")

    entry = build_registry_entry(base_entry, model_dir_name, files, size_value)
    if not base_entry:
        family = ARCH_TO_FAMILY.get(model_arch, "")
        entry.setdefault("details", {}).setdefault("family", family)
        entry.setdefault("flm_min_version", "0.9.45")

    user_registry = get_user_registry_path()
    register_model(tag, entry, user_registry, system_list)

    print(f"[INFO] Deployed model files to: {target}")
    print(f"[INFO] Registered tag '{tag}' in: {user_registry}")
    if src_tag:
        print(f"[INFO] Registry defaults copied from official entry: {src_tag}")
    else:
        print("[WARN] No official registry entry found; wrote a minimal entry (family/size guesses).")
    if os.environ.get("FLM_CONFIG_PATH") != str(user_registry):
        print(f"[INFO] FLM does not read this registry yet. Add to your shell rc:\n"
              f"        export FLM_CONFIG_PATH={user_registry}")
    else:
        print(f"[INFO] FLM_CONFIG_PATH already points at this registry. Ready to 'flm run {tag}'.")

    source_dir_name = base_entry.get("name") if base_entry else None
    if source_dir_name:
        user_xclbin_root = link_model_xclbins(model_dir_name, source_dir_name)
        if user_xclbin_root is not None:
            print(f"[INFO] Linked xclbins for '{model_dir_name}' -> '{source_dir_name}'")
            if os.environ.get("FLM_XCLBIN_PATH") != str(user_xclbin_root.parent):
                print(f"[INFO] The runtime needs to find these xclbins. Add to your shell rc:\n"
                      f"        export FLM_XCLBIN_PATH={user_xclbin_root.parent}")
            else:
                print("[INFO] FLM_XCLBIN_PATH already points at the user xclbin tree.")

    return {
        "tag": tag,
        "name": name,
        "size": size,
        "dir": target,
        "src_tag": src_tag,
        "user_registry": user_registry,
    }
