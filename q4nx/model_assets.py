import json
import os
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .arch_detect import ARCH_TO_FAMILY, family_from_text, resolve_override_arch
from .constants import ModelArch

ASSET_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
REQUIRED_ASSETS = ["config.json", "tokenizer.json", "tokenizer_config.json"]


def _gguf_field(reader, name: str):
    """Return the python value of a GGUF metadata field (scalar string/array included)."""
    field = reader.fields.get(name)
    if field is None:
        return None
    try:
        return field.contents()
    except Exception:
        pass
    try:
        if len(field.data) == 0:
            return None
        return field.parts[field.data[0]].decode("utf-8", errors="replace")
    except Exception:
        return None


def _gguf_token_list(reader) -> List[str]:
    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None:
        return []
    try:
        return [field.parts[i].decode("utf-8", errors="replace") for i in field.data]
    except Exception:
        return []


def _repo_id_from_url(url) -> Optional[str]:
    if not url:
        return None
    m = re.search(
        r"(?:huggingface\.co|hf\.co|modelscope\.cn/models)/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        url,
    )
    return m.group(1) if m else None


def read_provenance(reader) -> dict:
    """Extract the provenance chain from GGUF metadata.

    llama.cpp converters record where a GGUF came from:
      - general.base_model.{i}.repo_url / .repository / .organization / .name
      - general.source.huggingface.repository
      - general.repo_url (the quantizer's repo)
      - tokenizer.chat_template
    """
    info: dict = {}
    base_models = []
    i = 0
    while i < 100:
        url = _gguf_field(reader, f"general.base_model.{i}.repo_url")
        repository = _gguf_field(reader, f"general.base_model.{i}.repository")
        if url is None and repository is None:
            break
        base_models.append(
            {
                "repo_url": url,
                "repository": repository,
                "organization": _gguf_field(reader, f"general.base_model.{i}.organization"),
                "name": _gguf_field(reader, f"general.base_model.{i}.name"),
            }
        )
        i += 1
    info["base_models"] = base_models
    info["source_hf_repository"] = _gguf_field(reader, "general.source.huggingface.repository")
    info["repo_url"] = _gguf_field(reader, "general.repo_url")
    info["chat_template"] = _gguf_field(reader, "tokenizer.chat_template")
    return info


def resolve_repo_candidates(reader, source_arg: Optional[str]) -> List[str]:
    """Ordered list of repos/local dirs to try for non-GGUF assets.

    An explicit -s source is always tried first (repo id or local dir): the
    GGUF's own repo often only ships weights, so tokenizer/config assets are
    pulled from the -s source whenever one is given. GGUF provenance follows.
    """
    candidates: List[str] = []
    if source_arg:
        if os.path.exists(source_arg):
            candidates.append(source_arg)  # explicit local dir
        elif "/" in source_arg:
            candidates.append(source_arg)  # explicit repo id
    provenance = read_provenance(reader)
    for base in provenance["base_models"]:
        rid = base.get("repository") or _repo_id_from_url(base.get("repo_url"))
        if rid and rid not in candidates:
            candidates.append(rid)
    rid = provenance.get("source_hf_repository")
    if rid and rid not in candidates:
        candidates.append(rid)
    rid = _repo_id_from_url(provenance.get("repo_url"))
    if rid and rid not in candidates:
        candidates.append(rid)
    return candidates


def _cache_roots() -> List[Path]:
    roots = []
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(env)
        if not value:
            continue
        path = Path(value)
        if path.name == "hub":
            roots.append(path)
        else:
            roots.append(path / "hub")
    home = Path.home()
    roots.append(home / ".cache" / "huggingface" / "hub")
    roots.append(home / "cache" / "huggingface" / "hub")
    seen, result = set(), []
    for root in roots:
        if str(root) not in seen:
            seen.add(str(root))
            result.append(root)
    return result


def repo_folder_name(repo_id: str) -> str:
    return "models--" + "--".join(repo_id.split("/"))


def cached_snapshot_dir(repo_id: str) -> Optional[Path]:
    """Return the snapshot directory of a repo in the local HF cache, if present."""
    for root in _cache_roots():
        repo_dir = root / repo_folder_name(repo_id)
        if not repo_dir.is_dir():
            continue
        snapshots = repo_dir / "snapshots"
        if snapshots.is_dir():
            candidates = [s for s in snapshots.iterdir() if s.is_dir()]
            refs = repo_dir / "refs"
            if refs.is_dir():
                for ref in refs.iterdir():
                    try:
                        rev = ref.read_text().strip()
                    except Exception:
                        continue
                    rev_dir = snapshots / rev
                    if rev_dir.is_dir():
                        return rev_dir
            if candidates:
                return candidates[0]
    return None


def _copy_from_dir(source_dir: Path, output_dir: Path, files: List[str]) -> List[str]:
    copied = []
    for filename in files:
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)
            copied.append(filename)
    return copied


def _hf_download_file(repo_id: str, filename: str) -> Optional[str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[WARN] huggingface_hub not installed; cannot download missing model assets")
        return None
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        print(f"[WARN] Could not download {filename} from {repo_id}: {e}")
        return None


# Quantized GGUF preference order when -i names an HF repo instead of a file.
# Most families convert best starting from q4_1; a few (e.g. LFM) prefer q4_0,
# and some (e.g. gpt-oss) additionally accept mxfp4 as a last resort. The
# per-family overrides below are keyed by runtime family name; anything not
# listed falls back to GGUF_QUANT_PRIORITY. The order is resolved per call from
# the -f flag (exact) or, as a best-effort fallback, the repo id / GGUF
# filenames (so the user only needs -f when auto-detection guesses wrong).
GGUF_QUANT_PRIORITY: Tuple[str, ...] = ("q4_1", "q4_0", "q8_0")

GGUF_QUANT_PRIORITY_BY_FAMILY: Dict[str, Tuple[str, ...]] = {
    "lfm2": ("q4_0", "q4_1", "q8_0"),
    "gpt-oss": ("q4_1", "q4_0", "q8_0", "mxfp4"),
}


def _gguf_quant_priority(override_model_arch: str, repo_id: str, gguf_filenames: List[str]) -> Tuple[str, ...]:
    """Resolve the GGUF quant fallback order for an HF repo.

    Precedence:
      1. -f flag (resolve override_model_arch -> family)
      2. best-effort keyword match against the repo id or any .gguf filename
      3. the global GGUF_QUANT_PRIORITY default
    """
    family: Optional[str] = None
    if override_model_arch:
        arch = resolve_override_arch(override_model_arch)
        if arch is not None:
            family = ARCH_TO_FAMILY.get(arch)
    if family is None:
        family = family_from_text(repo_id)
    if family is None:
        for fname in gguf_filenames:
            family = family_from_text(os.path.basename(fname))
            if family is not None:
                break
    if family and family in GGUF_QUANT_PRIORITY_BY_FAMILY:
        return GGUF_QUANT_PRIORITY_BY_FAMILY[family]
    return GGUF_QUANT_PRIORITY


def find_repo_gguf(repo_id: str, override_model_arch: str = "") -> Optional[Tuple[str, str]]:
    """Search an HF repo for a quantized GGUF, in family-preferred order.

    The quant fallback order is family-aware: most families prefer q4_1 then
    q4_0 then q8_0, but some differ (e.g. LFM prefers q4_0 first; gpt-oss also
    accepts mxfp4 last). It is driven by the -f flag when given, otherwise by a
    best-effort match on the repo id / GGUF filenames.

    Returns (local_path, repo_filename) using the HF cache (downloading if
    needed), or None if the repo has no matching GGUF.
    """
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        print("[WARN] huggingface_hub not installed; cannot search HF repos for a GGUF")
        return None
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"[WARN] Could not list files in {repo_id}: {e}")
        return None

    gguf_filenames = [
        f for f in files if f.lower().endswith(".gguf")
    ]
    priority = _gguf_quant_priority(override_model_arch, repo_id, gguf_filenames)

    matches = []  # (priority_index, filename)
    other_ggufs = []
    for fname in gguf_filenames:
        low = os.path.basename(fname).lower()
        found = next(
            (q for q in priority if q in low), None
        )
        if found is not None:
            matches.append((priority.index(found), fname))
        else:
            other_ggufs.append(fname)
    if not matches:
        if other_ggufs:
            print(
                f"[WARN] No {', '.join(priority)} GGUF in {repo_id}; "
                f"only found: {', '.join(sorted(other_ggufs))}"
            )
        return None

    _, filename = sorted(matches, key=lambda m: (m[0], m[1].lower()))[0]
    print(f"[INFO] Found GGUF in {repo_id}: {filename}")
    path = _hf_download_file(repo_id, filename)
    if path is None:
        return None
    return path, filename


def fetch_hf_repo_info(repo_id: str) -> dict:
    """Fetch HF repo metadata for README enrichment; {} if unavailable/offline."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {}
    try:
        info = HfApi().model_info(repo_id)
    except Exception as e:
        print(f"[WARN] Could not fetch repo info for {repo_id}: {e}")
        return {}
    card = getattr(info, "cardData", None) or {}
    config = getattr(info, "config", None) or {}
    quant_cfg = config.get("quantization_config") or {}
    result = {}
    sha = getattr(info, "sha", None)
    if sha:
        result["sha"] = sha
    downloads = getattr(info, "downloads", None)
    if downloads:
        result["downloads"] = downloads
    tags = getattr(info, "tags", None) or []
    if tags:
        result["tags"] = tags
    license = getattr(info, "license", None) or card.get("license")
    if license:
        result["license"] = license
    library = getattr(info, "library_name", None)
    if library:
        result["library"] = library
    pipeline = getattr(info, "pipeline_tag", None)
    if pipeline:
        result["pipeline_tag"] = pipeline
    base_model = card.get("base_model")
    if base_model:
        result["base_model"] = base_model
    language = card.get("language")
    if language:
        result["language"] = language
    if config.get("model_type"):
        result["model_type"] = config["model_type"]
    if config.get("architectures"):
        result["architectures"] = config["architectures"]
    if quant_cfg.get("quant_method"):
        result["quant_method"] = quant_cfg["quant_method"]
    return result


def _source_dir_for_candidate(candidate: str) -> Optional[Path]:
    """Resolve a candidate (local dir or repo id) to a directory holding model files."""
    if os.path.isdir(candidate):
        return Path(candidate)
    snapshot = cached_snapshot_dir(candidate)
    if snapshot is not None:
        return snapshot
    return None


def _fetch_assets(candidate: str, output_dir: Path, files: List[str]) -> List[str]:
    """Fetch files for a candidate repo into output_dir. Uses cache, then SDK download."""
    source_dir = _source_dir_for_candidate(candidate)
    if source_dir is not None:
        copied = _copy_from_dir(source_dir, output_dir, files)
        if copied:
            print(f"[INFO] Copied model assets from local source: {source_dir}")
            return copied
        return []
    if "/" in candidate:
        downloaded = []
        for filename in files:
            path = _hf_download_file(candidate, filename)
            if path:
                shutil.copy2(path, output_dir / filename)
                downloaded.append(filename)
        if downloaded:
            print(f"[INFO] Downloaded model assets from {candidate}")
        return downloaded
    return []


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------

def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.2f} {unit}".rstrip("0").rstrip(".")
        size /= 1024.0
    return f"{size:.2f} TB"


def _read_source_file(candidate: str, filename: str) -> Optional[str]:
    """Read a text file (e.g. README.md) from a candidate repo, without writing it."""
    source_dir = _source_dir_for_candidate(candidate)
    if source_dir is not None:
        path = source_dir / filename
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return None
    if "/" in candidate:
        path = _hf_download_file(candidate, filename)
        if path:
            try:
                return Path(path).read_text(encoding="utf-8", errors="replace")
            except Exception:
                return None
    return None


def _fetch_source_readme(candidates: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (source_id, readme_text) for the first candidate with a README.md."""
    for candidate in candidates:
        if not candidate:
            continue
        text = _read_source_file(candidate, "README.md")
        if text:
            return candidate, text
    return None, None


def _source_link(source: str) -> Optional[str]:
    """HF model-card URL for a repo id, None for local paths / non-repo strings."""
    if not source or os.path.isdir(source) or source.startswith(("/", ".", "\\")):
        return None
    parts = source.split("/")
    if len(parts) == 2 and all(parts) and "\\" not in source:
        return f"https://huggingface.co/{source}"
    return None


def _modality_label(output_dir: Path) -> str:
    labels = []
    if (output_dir / "model.q4nx").is_file():
        labels.append("language")
    if (output_dir / "vision_weight.q4nx").is_file():
        labels.append("vision")
    if (output_dir / "audio_weight.q4nx").is_file():
        labels.append("audio")
    return " / ".join(labels) or "language"


def build_readme_meta(output_dir: Path, flm_version: Optional[str]) -> dict:
    """Gather conversion metadata used to render the README banner."""
    meta = {
        "title": output_dir.name or None,
        "tag": output_dir.name,
        "modality": _modality_label(output_dir),
        "flm_version": flm_version,
        "date": date.today().isoformat(),
    }
    for filename in ("model.q4nx", "vision_weight.q4nx", "audio_weight.q4nx"):
        path = output_dir / filename
        if path.is_file():
            meta["weight_file"] = filename
            meta["weight_size"] = _human_size(path.stat().st_size)
            break
    return meta


def _readme_banner(meta: dict) -> str:
    title = meta.get("title") or meta.get("source") or "Model"
    source = meta.get("source")
    tag = meta.get("tag") or title
    weight_file = meta.get("weight_file", "model.q4nx")
    modality = meta.get("modality", "language")

    parts = [f"# {title}", ""]
    if source and meta.get("source_url"):
        parts.append(
            f"**FastFlowLM Q4NX conversion of [`{source}`]({meta['source_url']})** "
            "for AMD XDNA NPU inference."
        )
    elif source:
        parts.append(f"**FastFlowLM Q4NX conversion of `{source}`** for AMD XDNA NPU inference.")
    else:
        parts.append("**FastFlowLM Q4NX model** for AMD XDNA NPU inference.")
    parts += [
        "",
        "This repository contains a quantized **Q4NX** port of the model, compiled for the "
        "FastFlowLM (FLM) runtime. It is **not** a GGUF file.",
        "",
        "| Item | Value |",
        "|------|-------|",
    ]
    rows = []
    if source and meta.get("source_url"):
        rows.append(f"| Source model | [`{source}`]({meta['source_url']}) |")
    elif source:
        rows.append(f"| Source model | `{source}` |")
    if meta.get("source_file"):
        rows.append(f"| Source GGUF | `{meta['source_file']}` |")
    if meta.get("weight_size"):
        rows.append(f"| Weights | `{weight_file}` ({meta['weight_size']}) |")
    rows.append(f"| Modality | {modality} |")
    if meta.get("flm_version"):
        rows.append(f"| FLM version | `{meta['flm_version']}` |")
    if meta.get("date"):
        rows.append(f"| Converted | {meta['date']} |")
    parts.append("\n".join(rows))

    repo_rows = []
    if meta.get("license"):
        repo_rows.append(f"| License | {meta['license']} |")
    if meta.get("base_model"):
        repo_rows.append(f"| Base model | `{meta['base_model']}` |")
    if meta.get("library"):
        repo_rows.append(f"| Library | `{meta['library']}` |")
    if meta.get("model_type"):
        repo_rows.append(f"| Model type | `{meta['model_type']}` |")
    if meta.get("pipeline_tag"):
        repo_rows.append(f"| Pipeline | `{meta['pipeline_tag']}` |")
    if meta.get("quant_method"):
        repo_rows.append(f"| Upstream quant method | `{meta['quant_method']}` |")
    if meta.get("downloads"):
        repo_rows.append(f"| Downloads | {meta['downloads']:,} |")
    if meta.get("sha"):
        repo_rows.append(f"| Repo revision | `{meta['sha']}` |")
    if repo_rows:
        parts += [
            "",
            "## Source repository",
            "",
            "Metadata from the upstream Hugging Face repository:",
            "",
            "| Item | Value |",
            "|------|-------|",
            "\n".join(repo_rows),
            "",
        ]
    parts += [
        "",
        "## Usage",
        "",
        "Run it with FastFlowLM:",
        "",
        "```bash",
        f"flm run {tag}",
        "",
        "# or serve it as an OpenAI-compatible endpoint:",
        f"flm serve {tag}",
        "```",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    file_rows = []
    if "language" in modality or modality == "language":
        file_rows.append(("model.q4nx", "Quantized weights (Q8_0 / Q4_1 / BF16)"))
    if "vision" in modality:
        file_rows.append(("vision_weight.q4nx", "Vision encoder weights"))
    if "audio" in modality:
        file_rows.append(("audio_weight.q4nx", "Audio encoder weights"))
    file_rows += [
        ("config.json", "FLM runtime configuration"),
        ("tokenizer.json", "Tokenizer vocabulary"),
        ("tokenizer_config.json", "Tokenizer configuration"),
        ("chat_template.jinja", "Chat template"),
    ]
    parts.append("\n".join(f"| `{name}` | {desc} |" for name, desc in file_rows))
    return "\n".join(parts) + "\n\n"


def _adapt_source_readme(text: str) -> str:
    """Trim a source model card so it slots under our banner (single H1)."""
    text = text.lstrip("\ufeff \t\r\n")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:].lstrip("\n")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^#\s", line):
            lines[i] = "#" + line
            break
    return "\n".join(lines).rstrip() + "\n"


def generate_readme(readme_text: Optional[str], meta: dict) -> str:
    parts = [_readme_banner(meta)]
    if readme_text:
        parts.append("---\n\n## Source model card\n\n" + _adapt_source_readme(readme_text))
    else:
        parts.append(
            "---\n\n*Original model card not included; see the source repository for details.*\n"
        )
    return "\n".join(parts)


def assemble_readme(
    output_dir: Path,
    candidates: List[str],
    meta: dict,
    source_file: Optional[str] = None,
) -> None:
    """Generate a Q4NX-appropriate README.md in the output directory.

    The banner is rendered from conversion metadata; the body is the source
    repo's model card, adapted so this repo keeps a single top-level title.
    When the source resolves to an HF repo id, its metadata (license, base
    model, downloads, revision, ...) is fetched and added to the banner.
    """
    filtered = []
    for candidate in candidates:
        if not candidate:
            continue
        src = _source_dir_for_candidate(candidate)
        if src is not None and src.resolve() == output_dir.resolve():
            continue
        filtered.append(candidate)
    source_id, readme_text = _fetch_source_readme(filtered)
    if source_id and not meta.get("source"):
        meta["source"] = source_id
        meta["source_url"] = _source_link(source_id)
    if source_file and not meta.get("source_file"):
        meta["source_file"] = source_file
    if source_id and _source_link(source_id):
        for key, value in fetch_hf_repo_info(source_id).items():
            meta.setdefault(key, value)
    if readme_text:
        print(f"[INFO] Writing README.md based on {source_id}'s model card")
    else:
        print("[INFO] No source model card found; writing a minimal README.md")
    (output_dir / "README.md").write_text(generate_readme(readme_text, meta), encoding="utf-8")


def generate_config_from_gguf(reader) -> dict:
    """Best-effort HF-style config.json built from GGUF metadata (no-source fallback)."""
    cfg: dict = {}
    arch = _gguf_field(reader, "general.architecture")
    if arch:
        cfg["model_type"] = str(arch)
    mappings = [
        (".embedding_length", "hidden_size"),
        (".feed_forward_length", "intermediate_size"),
        (".block_count", "num_hidden_layers"),
        (".attention.head_count", "num_attention_heads"),
        (".attention.head_count_kv", "num_key_value_heads"),
        (".attention.layer_norm_rms_epsilon", "rms_norm_eps"),
        (".rope.dimension_count", "head_dim"),
        (".vocab_size", "vocab_size"),
    ]
    prefixes = [f"{arch}"] if arch else []
    for field_suffix, key in mappings:
        for prefix in prefixes:
            value = _gguf_field(reader, prefix + field_suffix)
            if value is not None:
                cfg[key] = value
                break
    general_vocab = _gguf_field(reader, "general.vocab_size")
    if "vocab_size" not in cfg and general_vocab is not None:
        cfg["vocab_size"] = general_vocab
    if "head_dim" in cfg and "num_attention_heads" in cfg and "hidden_size" in cfg:
        if cfg["head_dim"] * cfg["num_attention_heads"] == cfg["hidden_size"]:
            cfg.pop("head_dim", None)
    for key in list(cfg.keys()):
        if isinstance(cfg[key], bool):
            continue
        if isinstance(cfg[key], float) or isinstance(cfg[key], int):
            continue
        cfg[key] = str(cfg[key])
    return cfg


def generate_tokenizer_config(reader) -> dict:
    """Minimal tokenizer_config.json the FLM runtime can parse (no-source fallback)."""
    cfg: dict = {}
    tokens = _gguf_token_list(reader)
    id_map = {
        "bos_token_id": "tokenizer.ggml.bos_token_id",
        "eos_token_id": "tokenizer.ggml.eos_token_id",
        "unk_token_id": "tokenizer.ggml.unknown_token_id",
        "pad_token_id": "tokenizer.ggml.padding_token_id",
    }
    for hf_key, gguf_key in id_map.items():
        value = _gguf_field(reader, gguf_key)
        if value is not None:
            cfg[hf_key] = int(value)
    for token_key, id_key in [("bos_token", "bos_token_id"), ("eos_token", "eos_token_id")]:
        token_id = cfg.get(id_key)
        if token_id is not None and 0 <= token_id < len(tokens):
            cfg[token_key] = tokens[token_id]
    if "eos_token_id" in cfg:
        eos_id = cfg["eos_token_id"]
        cfg["eos_token_id"] = [eos_id] if not isinstance(eos_id, list) else eos_id
    chat_template = _gguf_field(reader, "tokenizer.chat_template")
    if chat_template:
        cfg["chat_template"] = chat_template
    return cfg


def ensure_runtime_tokenizer_ids(reader, output_dir: Path):
    """Backfill EOS/BOS/PAD token ids into the deployed tokenizer_config.json.

    The FLM runtime stops generation only when the sampled token id is in
    ``tokenizer_config["eos_token_id"]``. Several official sources (e.g.
    Qwen/Qwen3.5-9B) ship that field as null, which silently disables
    end-of-generation. Fill any null id from GGUF metadata, and normalize
    eos_token_id to a list, so converted models actually stop.
    """
    path = output_dir / "tokenizer_config.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    tokens = _gguf_token_list(reader)
    changed = False
    id_map = {
        "eos_token_id": "tokenizer.ggml.eos_token_id",
        "bos_token_id": "tokenizer.ggml.bos_token_id",
        "pad_token_id": "tokenizer.ggml.padding_token_id",
    }
    for hf_key, gguf_key in id_map.items():
        if cfg.get(hf_key) is None:
            value = _gguf_field(reader, gguf_key)
            if value is not None:
                cfg[hf_key] = int(value)
                changed = True
    for token_key, id_key in [("eos_token", "eos_token_id"), ("bos_token", "bos_token_id"), ("pad_token", "pad_token_id")]:
        token_id = cfg.get(id_key)
        if cfg.get(token_key) is None and isinstance(token_id, int) and 0 <= token_id < len(tokens):
            cfg[token_key] = tokens[token_id]
            changed = True
    if cfg.get("eos_token_id") is not None:
        eos = cfg["eos_token_id"]
        if not isinstance(eos, list):
            cfg["eos_token_id"] = [eos]
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Patched {path.name} with token ids from GGUF metadata")


def _tokenizer_id_lookup(tokenizer_path: Path) -> Dict[str, int]:
    """Map token text -> token id from tokenizer.json (added tokens first, then vocab)."""
    if not tokenizer_path.exists():
        return {}
    try:
        with open(tokenizer_path, encoding="utf-8") as f:
            tokenizer = json.load(f)
    except Exception:
        return {}
    lookup: Dict[str, int] = {}
    for added in tokenizer.get("added_tokens", []):
        content = added.get("content")
        if content is not None:
            lookup[content] = int(added["id"])
    vocab = tokenizer.get("model", {}).get("vocab", {})
    for text, token_id in vocab.items():
        lookup.setdefault(text, int(token_id))
    return lookup


def ensure_hf_tokenizer_ids(output_dir: Path) -> None:
    """Backfill EOS/BOS/PAD token ids into a copied HF tokenizer_config.json.

    The FLM runtime stops generation only when the sampled token id appears in
    ``tokenizer_config["eos_token_id"]``. HF Qwen-family repos ship that field as
    null (the end-of-turn token lives in ``eos_token``), which silently disables
    end-of-generation. Resolve each token string against the tokenizer and
    normalize ``eos_token_id`` to a list so converted models actually stop.
    """
    path = output_dir / "tokenizer_config.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    lookup = _tokenizer_id_lookup(output_dir / "tokenizer.json")
    config: dict = {}
    config_path = output_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}
    changed = False
    for token_key, id_key in [
        ("eos_token", "eos_token_id"),
        ("bos_token", "bos_token_id"),
        ("pad_token", "pad_token_id"),
    ]:
        token_text = cfg.get(token_key)
        current = cfg.get(id_key)
        if (current is None or current == [] or current == "") and isinstance(token_text, str):
            resolved = lookup.get(token_text)
            if resolved is None and config.get(id_key) is not None:
                resolved = int(config[id_key])
            if resolved is not None:
                cfg[id_key] = resolved
                changed = True
    if cfg.get("eos_token_id") is not None:
        eos = cfg["eos_token_id"]
        if not isinstance(eos, list):
            cfg["eos_token_id"] = [eos]
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Patched {path.name} with EOS/BOS/PAD token ids from HF tokenizer")


def inject_flm_keys(config: dict, q4nx_config: dict, output_dir: Path, flm_version: Optional[str]):
    """Restructure a source HF config.json into the shape the FLM runtime expects.

    - Flatten ``text_config`` into the top level (lm_config.hpp reads top-level
      keys such as hidden_size / num_attention_heads).
    - Keep the nested ``vision_config`` / ``audio_config`` objects (the runtime
      reads those via ``_vision_config`` / ``_audio_config``).
    - Inject flm_version and the weight-file names once the converted weights exist.
    """
    text_config = config.pop("text_config", None)
    if isinstance(text_config, dict):
        # Flatten text_config over top-level so LM_Config sees hidden_size etc.
        # Prefer text_config values (Ornith/VL wrappers put real LM hyperparams there).
        for key, value in text_config.items():
            if key in ("model_type",) or config.get(key) in (None, ""):
                config[key] = value
            else:
                config.setdefault(key, value)
        # Darwin-style text-only MoE: model_type becomes qwen3_5_moe_text.
        if text_config.get("model_type"):
            config["model_type"] = text_config["model_type"]
    # Drop HF vision blob when no vision weights were converted (text-only finetunes).
    if not (output_dir / "vision_weight.q4nx").exists():
        config.pop("vision_model_weight", None)
        # Keep Darwin parity: pure language configs omit nested vision_config.
        if config.get("model_type") in (
            "qwen3_5_moe_text", "qwen3_6_moe_text"
        ) or "text_config" not in config:
            # If original was a VL wrapper with nested text_config already popped,
            # strip unused vision_config so FLM stays text-only.
            vc = config.get("vision_config")
            if isinstance(vc, dict) and "vision_mm_engine_xclbin_name" not in vc:
                config.pop("vision_config", None)

    # qwen3.5/3.6 MoE: only moe_intermediate_size is declared upstream, but the
    # runtime LM_Config requires intermediate_size (== moe_intermediate_size).
    if (
        config.get("model_type") in (
            "qwen3_5_moe_text", "qwen3_5_moe", "qwen3_6_moe", "qwen3_6_moe_text"
        )
        and "moe_intermediate_size" in config
        and "intermediate_size" not in config
    ):
        config["intermediate_size"] = config["moe_intermediate_size"]
    # Engine memory-layout offsets (lm_config.hpp JSON_GETs addr_* with default 0).
    # Architecture-level, declared in the arch config. Darwin worked without them
    # on some FLM builds; still inject when present so MHA layouts are correct.
    for key in ("addr_qk", "addr_kv", "addr_kk", "addr_l_begin_mha", "addr_l_end_mha"):
        if key in q4nx_config:
            config.setdefault(key, q4nx_config[key])
    # Token ids: prefer text_config / generation defaults used by Darwin.
    if config.get("bos_token_id") is None and config.get("pad_token_id") is not None:
        config["bos_token_id"] = config["pad_token_id"]
    if config.get("eos_token_id") is None:
        config["eos_token_id"] = 248044
    # Darwin/Ornith engines need caching enabled at runtime.
    if config.get("model_type") in ("qwen3_5_moe", "qwen3_5_moe_text", "qwen3_6_moe", "qwen3_6_moe_text"):
        config["use_cache"] = True
    if flm_version:
        config["flm_version"] = flm_version
    vision_config = q4nx_config.get("vision_config", {})
    if vision_config:
        vision_file = vision_config.get("vision_file", "vision_weight.q4nx")
        if (output_dir / vision_file).exists():
            config["vision_model_weight"] = vision_file
            vc = config.setdefault("vision_config", {})
            if "vision_MM_K" in vision_config:
                vc["vision_MM_K"] = vision_config["vision_MM_K"]
            if "vision_MM_N" in vision_config:
                vc["vision_MM_N"] = vision_config["vision_MM_N"]
        else:
            config.pop("vision_model_weight", None)
    audio_config = q4nx_config.get("audio_config", {})
    if audio_config:
        audio_file = audio_config.get("audio_file", "audio_weight.q4nx")
        if (output_dir / audio_file).exists():
            config["audio_model_weight"] = audio_file
        else:
            config.pop("audio_model_weight", None)
    return config


def get_default_flm_version() -> Optional[str]:
    """Best-effort: read the installed FLM version for the config's flm_version."""
    try:
        output = subprocess.run(
            ["flm", "--version"], capture_output=True, text=True, timeout=3
        ).stdout
        m = re.search(r"v?(\d+\.\d+\.\d+)", output)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def assemble_model_assets_hf(
    hf_source: str,
    q4nx_config: dict,
    output_dir: str,
    source_model: Optional[str] = None,
    flm_version: Optional[str] = None,
    source_file: Optional[str] = None,
) -> None:
    """Build a complete model directory from an HF safetensors source.

    Same result as assemble_model_assets but sourced straight from an HF repo
    (no GGUF provenance involved): config.json / tokenizer.json /
    tokenizer_config.json / chat_template.jinja are copied from the HF model,
    then the config is restructured for the FLM runtime.
    """
    output_dir = Path(output_dir)
    if output_dir.suffix == ".q4nx":
        output_dir = output_dir.parent
    os.makedirs(output_dir, exist_ok=True)

    candidate = source_model or hf_source
    fetched = _fetch_assets(candidate, output_dir, ASSET_FILES)
    missing = [f for f in REQUIRED_ASSETS if f not in fetched]
    if missing:
        print(f"[WARN] HF source {candidate} missing required assets: {missing}")

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
    inject_flm_keys(config, q4nx_config, output_dir, flm_version)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    ensure_hf_tokenizer_ids(output_dir)

    assemble_readme(
        output_dir, [candidate], build_readme_meta(output_dir, flm_version), source_file
    )

    print(f"[INFO] Model directory ready: {output_dir}")


def assemble_model_assets(
    reader,
    q4nx_config: dict,
    output_dir: str,
    source_model: Optional[str] = None,
    flm_version: Optional[str] = None,
    source_file: Optional[str] = None,
) -> None:
    """Build a complete, uploadable model directory.

    Weights (model.q4nx / vision_weight.q4nx / audio_weight.q4nx) are produced by the
    converter itself. Everything else - tokenizer.json, tokenizer_config.json,
    config.json and optionally chat_template.jinja - is copied from the original
    HF/ModelScope model (local cache first, SDK download as backup), following the
    provenance recorded in the GGUF. If no source can be found, best-effort files
    are generated from GGUF metadata with a loud warning.
    """
    output_dir = Path(output_dir)
    if output_dir.suffix == ".q4nx":
        output_dir = output_dir.parent
    os.makedirs(output_dir, exist_ok=True)

    candidates = resolve_repo_candidates(reader, source_model)
    if source_model and not candidates:
        candidates = [source_model]

    fetched = []
    for candidate in candidates:
        fetched = _fetch_assets(candidate, output_dir, ASSET_FILES)
        missing = [f for f in REQUIRED_ASSETS if f not in fetched]
        if not missing:
            break
        if fetched:
            print(f"[WARN] Source {candidate} is missing required files: {missing}")

    if fetched:
        print(f"[INFO] Model assets present: {fetched}")

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    else:
        print("[WARN] No source model found; generating config.json from GGUF metadata.")
        print("[WARN] tokenizer files may not exactly match the official model.")
        config = generate_config_from_gguf(reader)

    inject_flm_keys(config, q4nx_config, output_dir, flm_version)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        generated = generate_tokenizer_config(reader)
        if generated:
            print("[WARN] Generating tokenizer_config.json from GGUF metadata (runtime-required).")
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(generated, f, indent=2, ensure_ascii=False)

    ensure_runtime_tokenizer_ids(reader, output_dir)

    chat_template_path = output_dir / "chat_template.jinja"
    if not chat_template_path.exists():
        chat_template = _gguf_field(reader, "tokenizer.chat_template")
        if chat_template:
            print("[INFO] Writing chat_template.jinja from GGUF metadata.")
            chat_template_path.write_text(chat_template, encoding="utf-8")

    assemble_readme(output_dir, candidates, build_readme_meta(output_dir, flm_version), source_file)

    print(f"[INFO] Model directory ready: {output_dir}")
