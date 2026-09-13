from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable

from system_analysis.analysis.component_diagnosis import ComponentDiagnosisProfile, ComponentValue, format_component_value, parse_component_value


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JLC_DIR = Path("D:/嵌入式大赛")
DEFAULT_SCHEMATIC_PDF = DEFAULT_JLC_DIR / "SCH_Schematic1_2026-06-14.pdf"
DEFAULT_PCB_PDF = DEFAULT_JLC_DIR / "PCB_PCB1_2026-06-14.pdf"
DEFAULT_CLEAR_SCHEMATIC_DIR = PROJECT_ROOT / "测量数据" / "电路图_清晰版"

REF_PATTERN = r"[RC](?:[1-9]\d{0,2})"
DESIGNATOR_RE = re.compile(rf"\b({REF_PATTERN})\b")
DESIGNATOR_VALUE_RE = re.compile(
    rf"\b(?P<designator>{REF_PATTERN})\b\s+"
    r"(?P<value>\d+(?:\.\d+)?\s*(?:meg|Mohm|MOhm|kΩ|KΩ|kOhm|KOhm|k|K|Ω|ohm|nF|uF|μF|pF|n|u|μ|p))\b"
)
VALUE_DESIGNATORS_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?\s*(?:meg|Mohm|MOhm|kΩ|KΩ|kOhm|KOhm|k|K|Ω|ohm|nF|uF|μF|pF|n|u|μ|p))\s+"
    rf"(?P<designators>{REF_PATTERN}(?:\s*,\s*{REF_PATTERN})*)\b"
)

LABEL_FROM_TYPE_ORDER = {
    ("lowpass", 1): "一阶低通",
    ("lowpass", 2): "二阶低通",
    ("lowpass", 3): "三阶低通",
    ("highpass", 1): "一阶高通",
    ("highpass", 2): "二阶高通",
    ("highpass", 3): "三阶高通",
    ("bandpass", 2): "带通",
    ("bandstop", 2): "带阻",
}


@dataclass(frozen=True)
class SchematicComponent:
    designator: str
    value: float
    kind: str
    raw_value: str = ""
    source: str = ""

    def label(self) -> str:
        return f"{self.designator}={format_component_value(self.value, self.kind)}"


@dataclass(frozen=True)
class CircuitTemplate:
    label: str
    circuit_type: str
    order: int
    input_node: str
    output_node: str
    resistors: tuple[str, ...]
    capacitors: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def designators(self) -> tuple[str, ...]:
        return (*self.resistors, *self.capacitors)


@dataclass(frozen=True)
class CircuitLibrary:
    components: dict[str, SchematicComponent] = field(default_factory=dict)
    circuits: dict[str, CircuitTemplate] = field(default_factory=dict)
    source_paths: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()

    def labels(self) -> list[str]:
        return list(self.circuits.keys())

    def profile_for_label(self, label: str, *, enabled: bool = True, calibrated: bool = False) -> ComponentDiagnosisProfile | None:
        template = self.circuits.get(label)
        if template is None:
            return None
        resistors = [self._component_value(name, "R") for name in template.resistors]
        capacitors = [self._component_value(name, "C") for name in template.capacitors]
        if any(item is None for item in resistors) or any(item is None for item in capacitors):
            return None
        return ComponentDiagnosisProfile(
            enabled=enabled,
            circuit_label=template.label,
            circuit_type=template.circuit_type,
            order=template.order,
            resistors=[item for item in resistors if item is not None],
            capacitors=[item for item in capacitors if item is not None],
            calibrated=calibrated,
        )

    def profile_for_type_order(
        self,
        circuit_type: str,
        order: int | None,
        *,
        enabled: bool = True,
        calibrated: bool = False,
    ) -> ComponentDiagnosisProfile | None:
        label = label_from_type_order(circuit_type, order)
        if label is None:
            return None
        return self.profile_for_label(label, enabled=enabled, calibrated=calibrated)

    def template_for_profile(self, profile: ComponentDiagnosisProfile | None) -> CircuitTemplate | None:
        if profile is None:
            return None
        return self.circuits.get(profile.circuit_label)

    def _component_value(self, designator: str, kind: str) -> ComponentValue | None:
        component = self.components.get(designator)
        if component is None:
            return None
        return ComponentValue(component.designator, component.value, kind)

    def source_summary(self) -> str:
        if not self.source_paths:
            return "内置八电路原理图参数"
        names = [path.name for path in self.source_paths]
        return " / ".join(names)


def label_from_type_order(circuit_type: str | None, order: int | None) -> str | None:
    if circuit_type in ("bandpass", "bandstop"):
        return LABEL_FROM_TYPE_ORDER.get((str(circuit_type), 2))
    try:
        normalized_order = int(order or 1)
    except (TypeError, ValueError):
        normalized_order = 1
    return LABEL_FROM_TYPE_ORDER.get((str(circuit_type), normalized_order))


def label_from_analysis_result(result: object | None) -> str | None:
    if result is None:
        return None
    circuit_type = getattr(result, "filter_type", None)
    order = getattr(result, "order_estimate", None)
    return label_from_type_order(circuit_type, order)


def read_schematic_text(paths: Iterable[Path | str] | None = None) -> tuple[str, tuple[Path, ...], tuple[str, ...]]:
    selected = tuple(Path(path) for path in paths) if paths is not None else _default_source_paths()
    text_parts: list[str] = []
    used_paths: list[Path] = []
    warnings: list[str] = []
    for path in selected:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".pdf":
                text = _extract_pdf_text(path)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            warnings.append(f"读取 {path.name} 失败: {exc}")
            continue
        if text.strip():
            text_parts.append(text)
            used_paths.append(path)
    return "\n".join(text_parts), tuple(used_paths), tuple(warnings)


@lru_cache(maxsize=4)
def build_default_circuit_library() -> CircuitLibrary:
    text, source_paths, warnings = read_schematic_text()
    components = parse_components_from_text(text)
    components = {**_fallback_components(), **components}
    circuits = _default_circuit_templates()
    return CircuitLibrary(components=components, circuits=circuits, source_paths=source_paths, warnings=warnings)


def parse_components_from_text(text: str) -> dict[str, SchematicComponent]:
    normalized = _normalize_component_text(text)
    components: dict[str, SchematicComponent] = {}

    for match in VALUE_DESIGNATORS_RE.finditer(normalized):
        value_text = _normalize_value_text(match.group("value"))
        for designator in DESIGNATOR_RE.findall(match.group("designators")):
            _add_component(components, designator, value_text, source="bom")

    for match in DESIGNATOR_VALUE_RE.finditer(normalized):
        designator = match.group("designator")
        value_text = _normalize_value_text(match.group("value"))
        _add_component(components, designator, value_text, source="schematic")

    return components


def format_schematic_profile_lines(library: CircuitLibrary | None = None) -> list[str]:
    library = library or build_default_circuit_library()
    lines = [
        "=== 原理图读取结果 ===",
        f"来源: {library.source_summary()}",
        f"识别到元件: {', '.join(library.components[name].label() for name in sorted(library.components))}",
        f"生成电路模板: {', '.join(library.labels())}",
    ]
    if library.warnings:
        lines.append("读取提示: " + "；".join(library.warnings))
    lines.append("")
    return lines


def _add_component(components: dict[str, SchematicComponent], designator: str, value_text: str, source: str) -> None:
    kind = designator[:1]
    if not _value_matches_kind(value_text, kind):
        return
    try:
        value = parse_component_value(_normalize_value_for_parser(value_text, kind))
    except Exception:
        return
    previous = components.get(designator)
    if previous is not None and source != "schematic":
        return
    components[designator] = SchematicComponent(
        designator=designator,
        value=value,
        kind=kind,
        raw_value=value_text,
        source=source,
    )


def _normalize_component_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("Ω", "Ω ").replace("μ", "μ")
    text = re.sub(r"(?<=\d)(nF|uF|μF|pF|kΩ|KΩ|Ω|K|k)\b", r" \1", text)
    return re.sub(r"\s+", " ", text)


def _normalize_value_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def _normalize_value_for_parser(text: str, kind: str) -> str:
    raw = _normalize_value_text(text)
    lower = raw.lower()
    if kind == "C":
        for suffix in ("nf", "uf", "μf", "pf"):
            if lower.endswith(suffix):
                return raw[: -1]
    return raw


def _value_matches_kind(text: str, kind: str) -> bool:
    raw = _normalize_value_text(text)
    lower = raw.lower()
    if kind == "C":
        return lower.endswith(("pf", "nf", "uf", "μf", "p", "n", "u", "μ"))
    if "f" in lower:
        return False
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:meg|mohm|kohm|kω|k|ω|ohm|m)?", lower))


def _default_source_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in (DEFAULT_SCHEMATIC_PDF, DEFAULT_PCB_PDF):
        if path.exists():
            paths.append(path)
    if DEFAULT_CLEAR_SCHEMATIC_DIR.exists():
        paths.extend(sorted(DEFAULT_CLEAR_SCHEMATIC_DIR.glob("*原理图.svg")))
    return tuple(paths)


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _default_circuit_templates() -> dict[str, CircuitTemplate]:
    templates = [
        CircuitTemplate(
            label="一阶低通",
            circuit_type="lowpass",
            order=1,
            input_node="Vin",
            output_node="OUT1",
            resistors=("R1",),
            capacitors=("C1",),
            notes=("低通连接: R 串入信号, C 接 GND。",),
        ),
        CircuitTemplate(
            label="二阶低通",
            circuit_type="lowpass",
            order=2,
            input_node="Vin",
            output_node="OUT2",
            resistors=("R1", "R2"),
            capacitors=("C1", "C2"),
            notes=("两级一阶低通级联。",),
        ),
        CircuitTemplate(
            label="三阶低通",
            circuit_type="lowpass",
            order=3,
            input_node="Vin",
            output_node="OUT3/VOUT",
            resistors=("R1", "R2", "R3"),
            capacitors=("C1", "C2", "C3"),
            notes=("三极点由 R1-C1、R2-C2、R3-C3 级联形成。",),
        ),
        CircuitTemplate(
            label="一阶高通",
            circuit_type="highpass",
            order=1,
            input_node="Vin",
            output_node="OUT1",
            resistors=("R1",),
            capacitors=("C1",),
            notes=("高通连接: C 串入信号, R 接 VREF。",),
        ),
        CircuitTemplate(
            label="二阶高通",
            circuit_type="highpass",
            order=2,
            input_node="Vin",
            output_node="OUT2",
            resistors=("R1", "R2"),
            capacitors=("C1", "C2"),
            notes=("两级一阶高通级联。",),
        ),
        CircuitTemplate(
            label="三阶高通",
            circuit_type="highpass",
            order=3,
            input_node="Vin",
            output_node="OUT3/VOUT",
            resistors=("R1", "R2", "R3"),
            capacitors=("C1", "C2", "C3"),
            notes=("三极点由 R1-C1、R2-C2、R3-C3 级联形成。",),
        ),
        CircuitTemplate(
            label="带通",
            circuit_type="bandpass",
            order=2,
            input_node="BP_IN",
            output_node="BP_OUT",
            resistors=("R4", "R5"),
            capacitors=("C4", "C5"),
            notes=("R4-C4 约束低端拐点, R5-C5 约束高端拐点。",),
        ),
        CircuitTemplate(
            label="带阻",
            circuit_type="bandstop",
            order=2,
            input_node="BR_IN",
            output_node="BR_OUT",
            resistors=("R8", "R11"),
            capacitors=("C7", "C9"),
            notes=("R12/C8 是 Twin-T 平衡支路, V2 先用等效二阶陷波模板拟合。",),
        ),
    ]
    return {template.label: template for template in templates}


def _fallback_components() -> dict[str, SchematicComponent]:
    raw = {
        "R1": "10k",
        "R2": "10k",
        "R3": "10k",
        "R4": "3.3k",
        "R5": "10k",
        "R6": "20k",
        "R7": "10k",
        "R8": "10k",
        "R9": "20k",
        "R10": "10k",
        "R11": "10k",
        "R12": "5.1k",
        "C1": "10n",
        "C2": "10n",
        "C3": "10n",
        "C4": "100n",
        "C5": "1n",
        "C6": "100n",
        "C7": "10n",
        "C8": "10n",
        "C9": "10n",
    }
    components: dict[str, SchematicComponent] = {}
    for designator, value_text in raw.items():
        value = parse_component_value(value_text)
        components[designator] = SchematicComponent(
            designator=designator,
            value=value,
            kind=designator[:1],
            raw_value=value_text,
            source="fallback",
        )
    return components
