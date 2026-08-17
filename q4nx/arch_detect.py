"""Heuristic GGUF model-family detection and the family fingerprint chart.

gguf's ``general.architecture`` field is authoritative when present and
recognized. These profiles add best-effort *close enough* heuristics
(basename/general.name keywords, distinctive tensor names, metadata field
names) so unlisted or mislabeled GGUFs still land on a plausible family the
user can confirm or override with ``-f`` in convert.py.

The chart (``render_chart()``) maps every supported family to the signals used
to recognize it, so the heuristics can be audited and refined in one place.

Everything here is read-only with respect to the reader: it never mutates the
GGUF and never raises. Ranking is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gguf import GGUFReader

from .constants import ModelArch, ModelArchNames, QWEN35_VARIANT_DIMS

# confidence tiers, weakest to strongest (used for sorting only)
_CONF_RANK = {"low": 0, "medium": 1, "high": 2, "exact": 3}


@dataclass(frozen=True)
class FamilyProfile:
    """Everything the heuristics know about one model family.

    arch:      archetype arch the family maps to (variants resolved later)
    family:    flm runtime family name, e.g. "qwen3.6-moe"
    keywords:  substrings searched in general.basename / general.name
               (case-insensitive; the longest keyword wins across families)
    fingerprints: distinctive tensor-name substrings. Score = how many of
               these appear anywhere in the GGUF's tensor names.
    required_any: if non-empty, at least one of these substrings must appear
               somewhere or the family is ruled out (keeps a MoE file from
               matching its dense sibling and a text model from matching a
               vision family).
    excludes:  tensor-name substrings that rule the family out (score -> 0)
    field_prefixes: GGUF metadata field prefixes that hint at the family
               (weak signal, appended to reasons but not to the score)
    notes:     human-readable "how to spot it" text for the chart
    """

    arch: ModelArch
    family: str
    keywords: Tuple[str, ...] = ()
    fingerprints: Tuple[str, ...] = ()
    required_any: Tuple[str, ...] = ()
    excludes: Tuple[str, ...] = ()
    field_prefixes: Tuple[str, ...] = ()
    notes: str = ""


# Ordered most-specific first so ties in scoring/fingerprinting are broken
# toward the narrowest family.
FAMILY_PROFILES: Tuple[FamilyProfile, ...] = (
    FamilyProfile(
        arch=ModelArch.QWEN35MOE,
        family="qwen3.6-moe",
        keywords=(
            "qwen3.5moe", "qwen3.6moe", "qwen35moe",
            "qwen3.5-moe", "qwen3.6-moe", "qwen3.6-moe-text",
        ),
        fingerprints=(
            "ffn_gate_inp_shexp", "ffn_gate_shexp", "ffn_gate_exps",
            "ssm_conv1d", "ssm_alpha", "attn_qkv",
        ),
        required_any=("ffn_gate_inp_shexp", "ffn_gate_shexp", "ffn_gate_exps"),
        excludes=("attn_sinks",),
        field_prefixes=("qwen35moe.",),
        notes="MoE linear-attention (Qwen3.5/3.6-MoE). ssm_* tensors plus "
        "ffn_gate_inp_shexp / ffn_gate_shexp / ffn_gate_exps.",
    ),
    FamilyProfile(
        arch=ModelArch.QWEN35_2B,  # representative; real variant by embedding dim
        family="qwen3.5",
        keywords=("qwen3.5", "qwen35"),
        fingerprints=(
            "ssm_conv1d", "ssm_alpha", "ssm_out", "attn_gate", "attn_qkv",
        ),
        excludes=("ffn_gate_exps", "ffn_gate_inp_shexp", "shortconv"),
        field_prefixes=("qwen35.",),
        notes="Dense linear-attention Qwen3.5 (0.8B/2B/4B/9B). ssm_* tensors "
        "and fused attn_qkv. Variant resolved from qwen35.embedding_length: "
        "1536/2304/2560/4096.",
    ),
    FamilyProfile(
        arch=ModelArch.GPT_OSS,
        family="gpt-oss",
        keywords=("gpt-oss", "gpt_oss", "gptoss"),
        fingerprints=("attn_sinks", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"),
        required_any=("attn_sinks", "ffn_gate_exps"),
        excludes=("ssm_", "shortconv"),
        field_prefixes=("gpt-oss.",),
        notes="MoE with per-expert ffn_gate/up/down_exps and attention "
        "sinks; no ssm_ / shortconv tensors.",
    ),
    FamilyProfile(
        arch=ModelArch.LFM2,
        family="lfm2",
        keywords=("lfm2", "lfm-2"),
        fingerprints=("shortconv", "ssm_a"),
        excludes=("attn_sinks", "ffn_gate_exps"),
        field_prefixes=("lfm2.",),
        notes="Linear-attention with shortconv and ssm_a (also carries "
        "ssm_a_log / ssm_a_inv variants).",
    ),
    FamilyProfile(
        arch=ModelArch.PHI4,
        family="phi4",
        keywords=("phi-4", "phi4", "phi_4"),
        fingerprints=("rope_freqs_long", "rope_freqs_short"),
        excludes=("ssm_", "shortconv"),
        field_prefixes=("phi4.",),
        notes="Dense MHA without q/k norms or attn biases; rope_freqs_long/"
        "short are unique to Phi-4.",
    ),
    FamilyProfile(
        arch=ModelArch.QWEN3VL,
        family="qwen3vl",
        keywords=("qwen3-vl", "qwen3vl", "qwen3.5-vl", "qwen3.5vl"),
        fingerprints=("vision_patch_embd", "visual_deepstack", "q_norm", "k_norm"),
        required_any=("vision_patch_embd", "visual_deepstack"),
        excludes=("q_bias",),
        field_prefixes=("qwen3vl.",),
        notes="Qwen3-style text stack (q_norm/k_norm, no attn bias) plus "
        "vision_patch_embd / visual_* vision tower.",
    ),
    FamilyProfile(
        arch=ModelArch.QWEN3,
        family="qwen3",
        keywords=("qwen3", "qwen3moe"),
        fingerprints=("q_norm", "k_norm"),
        excludes=("q_bias", "vision_patch_embd", "post_ffn_norm", "ssm_", "shortconv"),
        field_prefixes=("qwen3.", "qwen3moe."),
        notes="Dense Qwen3 MHA with q_norm/k_norm, no attention biases. "
        "(Also matches Qwen3-MoE: same tensor skeleton.)",
    ),
    FamilyProfile(
        arch=ModelArch.QWEN2VL,
        family="qwen2.5vl",
        keywords=("qwen2.5-vl", "qwen2.5vl", "qwen2-vl", "qwen2vl"),
        fingerprints=("vision_patch_embd", "vision_merger", "q_bias"),
        required_any=("vision_patch_embd", "vision_merger"),
        excludes=("q_norm",),
        field_prefixes=("qwen2vl.",),
        notes="Qwen2-style text stack (q/k/v attention biases) plus "
        "vision_patch_embd / vision_merger vision tower.",
    ),
    FamilyProfile(
        arch=ModelArch.QWEN2,
        family="qwen2.5",
        keywords=("qwen2", "qwen2.5"),
        fingerprints=("q_bias", "k_bias", "v_bias"),
        excludes=("q_norm", "vision_patch_embd", "ssm_", "shortconv"),
        field_prefixes=("qwen2.",),
        notes="Dense Qwen2/Qwen2.5 MHA with q/k/v attention biases, no "
        "q/k norms. (Qwen2-MoE shares the skeleton.)",
    ),
    FamilyProfile(
        arch=ModelArch.GEMMA4,
        family="gemma4",
        keywords=("gemma4", "gemma-4"),
        fingerprints=(
            "inp_gate", "per_layer_token_embedding", "per_layer_projection",
            "layer_output_scale", "audio_",
        ),
        excludes=("ffn_gate_exps",),
        field_prefixes=("gemma4.",),
        notes="QAT'd Gemma 4: per_layer_token_embedding/per_layer_projection/"
        "layer_output_scale, inp_gate, audio_* + vision_* encoders.",
    ),
    FamilyProfile(
        arch=ModelArch.GEMMA3,
        family="gemma3",
        keywords=("gemma3", "gemma-3", "medgemma"),
        fingerprints=("post_ffn_norm", "q_proj_norm", "multi_modal_input_projection"),
        excludes=("inp_gate", "per_layer_token_embedding", "audio_", "q_bias", "q_norm"),
        field_prefixes=("gemma.",),
        notes="Gemma-3/MedGemma: q_proj_norm/k_proj_norm (naming differs "
        "from Qwen3's q_norm/k_norm) and post_ffn_norm.",
    ),
    FamilyProfile(
        arch=ModelArch.LLAMA,
        family="llama3.2",
        keywords=("llama", "meta-llama", "nemotron"),
        fingerprints=("rope_freqs",),
        excludes=(
            "q_norm", "k_norm", "q_bias", "ssm_", "shortconv", "ffn_gate_exps",
            "post_ffn_norm", "rope_freqs_long", "inp_gate", "vision_patch_embd",
        ),
        field_prefixes=("llama.",),
        notes="Generic Llama-style MHA: no q/k norms, no attn biases, plain "
        "rope_freqs. Deliberately the fallback family for any standard "
        "transformer (Mistral, etc.).",
    ),
    FamilyProfile(
        arch=ModelArch.NANBEIGE,
        family="nanbeige",
        keywords=("nanbeige",),
        field_prefixes=("nanbeige.",),
        notes="Tensor names are identical to Llama's, so it can only be told "
        "apart by general.architecture / basename / field prefix.",
    ),
)

# reverse map: embedding dim -> dense Qwen3.5 variant
_QWEN35_DIM_TO_ARCH: Dict[int, ModelArch] = {
    dim: arch for arch, dim in QWEN35_VARIANT_DIMS.items()
}


def _is_dense_qwen35_arch(arch: ModelArch) -> bool:
    return arch in (
        ModelArch.QWEN35_08B, ModelArch.QWEN35_2B,
        ModelArch.QWEN35_4B, ModelArch.QWEN35_9B,
    )


def _metadata_strings(reader: GGUFReader) -> Tuple[Optional[str], Optional[str]]:
    """Return (general.architecture, basename_or_name) from the reader."""
    arch_str: Optional[str] = None
    basename: Optional[str] = None
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch_str = str(field.parts[field.data[0]], encoding="utf-8") if field.data else None
        elif field.name in ("general.basename", "general.name") and basename is None:
            basename = str(field.parts[field.data[0]], encoding="utf-8") if field.data else None
    return arch_str, basename


def _match_arch_name(arch_str: str) -> Optional[ModelArch]:
    """Exact (case-insensitive) match of a GGUF architecture string."""
    needle = arch_str.lower()
    for arch_enum, names in ModelArchNames.items():
        for name in names:
            if needle == name.lower():
                return arch_enum
    return None


def _resolve_qwen35_variant(reader: GGUFReader) -> Tuple[ModelArch, List[str]]:
    """Resolve the dense Qwen3.5 variant from qwen35.embedding_length."""
    field = reader.fields.get("qwen35.embedding_length")
    if field is None:
        return ModelArch.QWEN35_2B, [
            "qwen35.embedding_length missing -> cannot pick 0.8B/2B/4B/9B; "
            "defaulting to 2B (use -f to force the size)"
        ]
    try:
        dim = int(field.contents())
    except Exception:  # pragma: no cover - defensive
        dim = -1
    if dim in _QWEN35_DIM_TO_ARCH:
        arch = _QWEN35_DIM_TO_ARCH[dim]
        return arch, [f"qwen35.embedding_length={dim} -> {arch.name}"]
    return ModelArch.QWEN35_2B, [
        f"qwen35.embedding_length={dim} not in {sorted(_QWEN35_DIM_TO_ARCH)}; "
        "defaulting to 2B (use -f to force the size)"
    ]


@dataclass
class Guess:
    """A candidate family with the evidence behind it.

    arch:        the ModelArch to use
    confidence:  "exact" (general.architecture), "high" (basename keyword),
                 "medium" (distinctive tensor fingerprint), "low" (weak hints)
    score:       internal evidence score, larger = stronger
    reasons:     human-readable lines of evidence
    """

    arch: ModelArch
    confidence: str
    score: int
    reasons: List[str]


def _exact_guess(reader: GGUFReader, arch_str: str) -> Optional[Guess]:
    # llama.cpp reports the dense Qwen3.5 architecture as bare 'qwen35' /
    # 'qwen3.5' with no size suffix; the variant is inferred from the
    # embedding dimension.
    if arch_str.lower() in ("qwen35", "qwen3.5"):
        arch, extra = _resolve_qwen35_variant(reader)
        return Guess(
            arch=arch,
            confidence="exact",
            score=1000,
            reasons=[f"general.architecture == {arch_str!r} (exact match)"] + extra,
        )
    arch = _match_arch_name(arch_str)
    if arch is None:
        return None
    return Guess(
        arch=arch,
        confidence="exact",
        score=1000,
        reasons=[f"general.architecture == {arch_str!r} (exact match)"],
    )


def _keyword_guess(text: str) -> Optional[Guess]:
    """Longest-keyword-wins match against basename/general.name."""
    lowered = text.lower()
    best_profile: Optional[FamilyProfile] = None
    best_keyword = ""
    for profile in FAMILY_PROFILES:
        for keyword in profile.keywords:
            if keyword.lower() in lowered and len(keyword) > len(best_keyword):
                best_profile = profile
                best_keyword = keyword
    if best_profile is None:
        return None
    return Guess(
        arch=best_profile.arch,
        confidence="high",
        score=100,
        reasons=[f"basename/name contains {best_keyword!r}"],
    )


def _fingerprint_guesses(tensor_names: Iterable[str]) -> List[Guess]:
    """Score every family against distinctive tensor-name substrings."""
    names = list(tensor_names)
    guesses: List[Guess] = []
    for profile in FAMILY_PROFILES:
        if not profile.fingerprints:
            continue
        if profile.required_any and not any(
            any(r in n for n in names) for r in profile.required_any
        ):
            continue
        excluded = [e for e in profile.excludes if any(e in n for n in names)]
        if excluded:
            continue
        hits = [f for f in profile.fingerprints if any(f in n for n in names)]
        if not hits:
            continue
        reasons = [f"tensor name matches {h!r} (e.g. {_example(names, h)})" for h in hits]
        guesses.append(
            Guess(
                arch=profile.arch,
                confidence="medium",
                score=10 * len(hits),
                reasons=reasons,
            )
        )
    return guesses


def _example(names: Sequence[str], needle: str) -> str:
    for name in names:
        if needle in name:
            return name
    return "?"


def _field_hints(reader: GGUFReader) -> List[Tuple[FamilyProfile, str]]:
    """Weak metadata-field hints, appended to reasons but not the score."""
    field_names = set(reader.fields.keys())
    hints: List[Tuple[FamilyProfile, str]] = []
    for profile in FAMILY_PROFILES:
        for prefix in profile.field_prefixes:
            for fname in field_names:
                if fname.startswith(prefix):
                    hints.append((profile, f"metadata field {fname} present"))
                    break
    return hints


def detect_model_family(reader: GGUFReader) -> List[Guess]:
    """Ranked, never-raising guesses for which family a GGUF belongs to.

    Passes, in order of authority:

    1. ``general.architecture`` exact match  (confidence "exact")
    2. basename / general.name keyword        (confidence "high")
    3. distinctive tensor-name fingerprint    (confidence "medium")
    4. metadata field prefixes                (confidence "low")

    Keyword and fingerprint evidence for the same family are merged. A dense
    Qwen3.5 guess is resolved to its size variant from ``qwen35.embedding_length``.
    """
    arch_str, basename = _metadata_strings(reader)
    tensor_names = [t.name for t in reader.tensors]

    if arch_str:
        exact = _exact_guess(reader, arch_str)
        if exact is not None:
            return [exact]

    kw_guess = _keyword_guess(basename or "") if basename else None
    fp_guesses = _fingerprint_guesses(tensor_names)
    field_hints = _field_hints(reader)

    merged: Dict[ModelArch, Guess] = {}
    for guess in [kw_guess] + fp_guesses:
        if guess is None:
            continue
        if guess.arch in merged:
            existing = merged[guess.arch]
            existing.reasons.extend(guess.reasons)
            existing.score += guess.score
        else:
            merged[guess.arch] = guess

    for profile, reason in field_hints:
        guess = merged.get(profile.arch)
        if guess is None:
            guess = merged[profile.arch] = Guess(
                arch=profile.arch, confidence="low", score=1, reasons=[]
            )
        guess.reasons.append(reason)

    # resolve dense Qwen3.5 to a concrete variant
    for arch, guess in list(merged.items()):
        if _is_dense_qwen35_arch(arch):
            variant, extra = _resolve_qwen35_variant(reader)
            guess.arch = variant
            guess.reasons.extend(extra)

    ranked = sorted(
        merged.values(),
        key=lambda g: (_CONF_RANK[g.confidence], g.score),
        reverse=True,
    )
    return ranked


def resolve_override_arch(override: str) -> Optional[ModelArch]:
    """Map a user-supplied -f arch string to a ModelArch, or None.

    Mirrors the override branch of get_model_arch_from_gguf: flm tag names use
    a colon (e.g. 'qwen3.5:9b') while arch names use a dash (e.g. 'qwen3.5-9B').
    Normalize and prefer the longest matching arch name so a shorthand still
    resolves to the right converter.
    """
    if not override:
        return None
    normalized = override.replace(":", "-").lower()
    best_match: Optional[ModelArch] = None
    best_len = 0
    for arch_enum, arch_names in ModelArchNames.items():
        for arch_name in arch_names:
            if normalized.startswith(arch_name.lower()) and len(arch_name) > best_len:
                best_match = arch_enum
                best_len = len(arch_name)
    return best_match


def family_from_text(text: str) -> Optional[str]:
    """Best-effort family name from an arbitrary string (repo id, file name).

    Longest keyword across FAMILY_PROFILES wins. Returns None if nothing
    matches, so callers can fall back to the default quant priority. Used to
    pick a sensible GGUF quant order for an HF repo before any GGUF is
    downloaded (the user can always override with -f).
    """
    if not text:
        return None
    lowered = text.lower()
    best_family: Optional[str] = None
    best_keyword = ""
    for profile in FAMILY_PROFILES:
        for keyword in profile.keywords:
            if keyword.lower() in lowered and len(keyword) > len(best_keyword):
                best_family = profile.family
                best_keyword = keyword
    return best_family


# ModelArch -> runtime family name. Also used to pick a per-family GGUF quant
# fallback order in model_assets.find_repo_gguf.
ARCH_TO_FAMILY: Dict[ModelArch, str] = {
    ModelArch.QWEN35_08B: "qwen3.5",
    ModelArch.QWEN35_2B: "qwen3.5",
    ModelArch.QWEN35_4B: "qwen3.5",
    ModelArch.QWEN35_9B: "qwen3.5",
    ModelArch.QWEN35MOE: "qwen3.6-moe",
    ModelArch.QWEN3: "qwen3",
    ModelArch.QWEN3VL: "qwen3vl",
    ModelArch.QWEN2: "qwen2.5",
    ModelArch.QWEN2VL: "qwen2.5vl",
    ModelArch.GEMMA3: "gemma3",
    ModelArch.GEMMA4: "gemma4",
    ModelArch.LLAMA: "llama3.2",
    ModelArch.LFM2: "lfm2",
    ModelArch.PHI4: "phi4",
    ModelArch.GPT_OSS: "gpt-oss",
    ModelArch.NANBEIGE: "nanbeige",
}


def render_chart() -> str:
    """Render the family -> detection-heuristic chart as markdown."""
    lines = [
        "# GGUF model family detection chart",
        "",
        "How each supported family is recognized. `general.architecture` is "
        "always the strongest signal; the other columns are the *close "
        "enough* heuristics used when it is missing or unrecognized.",
        "",
        "| Family | GGUF arch names | Tensor fingerprints | Basename/name keywords | Notes |",
        "|---|---|---|---|---|",
    ]
    for profile in FAMILY_PROFILES:
        family = profile.family
        if _is_dense_qwen35_arch(profile.arch):
            family += " (dense)"
            arch_names = ", ".join(
                name
                for variant in QWEN35_VARIANT_DIMS
                for name in ModelArchNames.get(variant, [])
            )
        else:
            arch_names = ", ".join(ModelArchNames.get(profile.arch, []))
        fp = ", ".join(f"`{f}`" for f in profile.fingerprints) or "*(none: use metadata)*"
        kws = ", ".join(f"`{k}`" for k in profile.keywords) or "-"
        if profile.excludes:
            fp += f"  <br>*(excludes: {', '.join(f'`{e}`' for e in profile.excludes)})*"
        lines.append(f"| **{family}** | {arch_names} | {fp} | {kws} | {profile.notes} |")
    lines.extend(
        [
            "",
            "Confidence tiers: `exact` = general.architecture matches, "
            "`high` = basename keyword, `medium` = tensor fingerprint, "
            "`low` = metadata field hint only.",
            "",
            "Run `tools/detect_family.py <model.gguf>` to see what an "
            "individual file resolves to.",
        ]
    )
    return "\n".join(lines)
