from __future__ import annotations

import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = OUT_DIR.parent / "visio_system_block_diagram"
SOURCE_TEMPLATE = SYSTEM_DIR / "visio_base_template.vsdx"
TEMPLATE_VSDX = OUT_DIR / "visio_base_template.vsdx"

PAGE_W = 16.54
PAGE_H = 13.20
TITLE = "STM32G431 频响测量与自动控制校正程序流程图"
SUBTITLE = "当前代码主线：扫频/导入 -> 建立 Session -> 频响分析 -> AI 诊断 -> 控制校正 -> 补扫或导出"

NS = "http://schemas.microsoft.com/office/visio/2012/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

BG = "#fbfcff"
INK = "#172033"
MUTED = "#4b5b70"
LINE = "#526072"


@dataclass(frozen=True)
class Lane:
    title: str
    x: float
    y: float
    w: float
    h: float
    fill: str
    stroke: str


@dataclass(frozen=True)
class Node:
    key: str
    x: float
    y: float
    w: float
    h: float
    title: str
    body: tuple[str, ...]
    fill: str
    stroke: str
    shape: str = "process"


@dataclass(frozen=True)
class Edge:
    start: str | tuple[float, float]
    end: str | tuple[float, float]
    label: str = ""
    color: str = LINE
    width: float = 1.8
    dashed: bool = False


def model() -> tuple[list[Lane], list[Node], list[Edge]]:
    lanes = [
        Lane("输入与采集", 0.75, 9.35, 15.05, 2.85, "#f7fbff", "#b8d2ed"),
        Lane("后台分析主链路", 0.75, 4.35, 15.05, 4.55, "#fbf8ff", "#c9b8ec"),
        Lane("结果处理", 0.75, 1.00, 15.05, 2.85, "#f0fbf8", "#98d1bf"),
    ]

    nodes = [
        Node("start", 6.82, 11.30, 2.30, 0.70, "启动上位机", ("main.py -> app.py",), "#e8f3ff", "#4f86c6", "terminator"),
        Node("input_mode", 6.70, 10.18, 2.55, 0.94, "选择数据来源", ("串口扫频 或 文本导入",), "#fff7df", "#cc8c24", "decision"),
        Node("serial_path", 1.85, 10.27, 3.00, 0.76, "串口扫频", ("校验参数", "启动 ThreeLineReader"), "#e8f3ff", "#4f86c6"),
        Node("stm32", 1.85, 9.42, 3.00, 0.76, "STM32 固件", ("SWEEP -> DAC/ADC/FFT", "输出三数组"), "#e8faf4", "#40a277"),
        Node("text_path", 11.50, 10.27, 3.00, 0.76, "文本导入", ("解析 omega / Magnitude", "Phase 数据"), "#e8f3ff", "#4f86c6", "io"),
        Node("frame", 6.55, 9.28, 2.85, 0.90, "handle_frame()", ("建立 MeasurementSession", "投递后台分析线程"), "#e8f3ff", "#4f86c6"),
        Node("settings", 6.55, 8.00, 2.85, 0.82, "读取分析设置", ("平滑、P、反接", "控制校正参数"), "#f2ebff", "#8066c6"),
        Node("analyze", 6.55, 7.02, 2.85, 0.82, "analyze_system_v2()", ("Bode / Nyquist", "稳定性判断"), "#f2ebff", "#8066c6"),
        Node("expected", 6.55, 6.04, 2.85, 0.82, "期望电路校验", ("Auto 或指定电路类型",), "#f2ebff", "#8066c6"),
        Node("ai", 6.55, 4.92, 2.85, 0.94, "run_intelligent_diagnosis()", ("特征提取、模板拟合", "故障规则与补扫计划"), "#eef9fb", "#39a0ad"),
        Node("control", 6.55, 3.93, 2.85, 0.90, "自动控制校正", ("PM / GM 计算", "超前 / 滞后校正器"), "#eef9fb", "#39a0ad"),
        Node("resweep_decision", 6.70, 3.00, 2.55, 0.94, "是否需要补扫？", ("置信度低或关键频段不足",), "#fff7df", "#cc8c24", "decision"),
        Node("merge", 11.20, 3.08, 3.00, 0.80, "SmartResweepReader", ("执行 1-3 条 SWEEP", "合并新旧测量帧"), "#e8faf4", "#40a277"),
        Node("report_lines", 6.55, 2.02, 2.85, 0.86, "生成报告文本", ("频响结论 + AI 诊断", "控制校正建议"), "#f2ebff", "#8066c6"),
        Node("done", 6.55, 1.05, 2.85, 0.86, "更新界面", ("历史列表、图表", "右侧摘要"), "#f2ebff", "#8066c6"),
        Node("export_decision", 10.00, 1.05, 2.35, 0.90, "是否导出？", ("报告、CSV 或图片",), "#fff7df", "#cc8c24", "decision"),
        Node("export", 12.72, 1.10, 2.60, 0.80, "导出结果", ("HTML 报告、CSV、PNG",), "#e8faf4", "#40a277", "io"),
        Node("finish", 2.00, 1.12, 2.40, 0.74, "流程结束", ("保存或继续下一次测量",), "#e8faf4", "#40a277", "terminator"),
    ]

    edges = [
        Edge("start", "input_mode"),
        Edge("input_mode", "serial_path", "串口扫频"),
        Edge("input_mode", "text_path", "文本导入"),
        Edge("serial_path", "stm32", "SWEEP"),
        Edge("stm32", "frame", "三数组"),
        Edge("text_path", "frame", "数组文本"),
        Edge("frame", "settings"),
        Edge("settings", "analyze"),
        Edge("analyze", "expected"),
        Edge("expected", "ai"),
        Edge("ai", "control"),
        Edge("control", "resweep_decision"),
        Edge("resweep_decision", "merge", "是"),
        Edge("merge", (15.10, 3.48), "", "#3fa578", 1.7, True),
        Edge((15.10, 3.48), (15.10, 9.75), "", "#3fa578", 1.7, True),
        Edge((15.10, 9.75), "frame", "合并后重新分析", "#3fa578", 1.7, True),
        Edge("resweep_decision", "report_lines", "否：结果可信"),
        Edge("report_lines", "done"),
        Edge("done", "export_decision"),
        Edge("export_decision", "export", "是"),
        Edge("export_decision", "finish", "否"),
        Edge("export", "finish"),
    ]
    return lanes, nodes, edges


def node_center(node: Node) -> tuple[float, float]:
    return node.x + node.w / 2, node.y + node.h / 2


def anchor(node: Node, toward: tuple[float, float]) -> tuple[float, float]:
    cx, cy = node_center(node)
    dx = toward[0] - cx
    dy = toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (node.w / 2) / abs(dx) if dx else float("inf")
    sy = (node.h / 2) / abs(dy) if dy else float("inf")
    t = min(sx, sy)
    return cx + dx * t, cy + dy * t


def edge_points(edge: Edge, by_key: dict[str, Node]) -> tuple[tuple[float, float], tuple[float, float]]:
    def center(value: str | tuple[float, float]) -> tuple[float, float]:
        return value if isinstance(value, tuple) else node_center(by_key[value])

    start_center = center(edge.start)
    end_center = center(edge.end)
    start = anchor(by_key[edge.start], end_center) if isinstance(edge.start, str) else start_center
    end = anchor(by_key[edge.end], start_center) if isinstance(edge.end, str) else end_center
    return start, end


def cell(name: str, value: str | float, formula: str | None = None) -> str:
    attrs = f"N='{name}' V='{escape(str(value))}'"
    if formula is not None:
        attrs += f" F='{escape(formula)}'"
    return f"<Cell {attrs}/>"


def char_section(color: str, size: float = 0.105, bold: bool = False) -> str:
    return (
        "<Section N='Character'><Row IX='0'>"
        f"{cell('Color', color)}{cell('Size', size)}{cell('Style', 17 if bold else 0)}{cell('Font', 0)}"
        "</Row></Section>"
    )


def para_section(align: int = 1) -> str:
    return f"<Section N='Paragraph'><Row IX='0'>{cell('HorzAlign', align)}</Row></Section>"


def geometry(shape: str) -> str:
    if shape == "diamond":
        rows = (
            "<Row T='RelMoveTo' IX='1'><Cell N='X' V='0.5'/><Cell N='Y' V='1'/></Row>"
            "<Row T='RelLineTo' IX='2'><Cell N='X' V='1'/><Cell N='Y' V='0.5'/></Row>"
            "<Row T='RelLineTo' IX='3'><Cell N='X' V='0.5'/><Cell N='Y' V='0'/></Row>"
            "<Row T='RelLineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='0.5'/></Row>"
            "<Row T='RelLineTo' IX='5'><Cell N='X' V='0.5'/><Cell N='Y' V='1'/></Row>"
        )
    else:
        rows = (
            "<Row T='RelMoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            "<Row T='RelLineTo' IX='2'><Cell N='X' V='1'/><Cell N='Y' V='0'/></Row>"
            "<Row T='RelLineTo' IX='3'><Cell N='X' V='1'/><Cell N='Y' V='1'/></Row>"
            "<Row T='RelLineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='1'/></Row>"
            "<Row T='RelLineTo' IX='5'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
        )
    return (
        "<Section N='Geometry' IX='0'>"
        f"{cell('NoFill', 0)}{cell('NoLine', 0)}{cell('NoShow', 0)}{cell('NoSnap', 0)}{cell('NoQuickDrag', 0)}"
        f"{rows}</Section>"
    )


def make_shape(
    sid: int,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fill: str,
    stroke: str,
    *,
    shape: str = "rect",
    text_color: str = INK,
    font_size: float = 0.105,
    bold: bool = False,
    line_weight: float = 0.012,
    rounding: float = 0.08,
    transparency: int = 0,
    align: int = 1,
) -> str:
    round_value = 0.25 if shape == "terminator" else rounding
    return "".join(
        [
            f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>",
            cell("PinX", x + w / 2),
            cell("PinY", y + h / 2),
            cell("Width", w),
            cell("Height", h),
            cell("LocPinX", w / 2, "Width*0.5"),
            cell("LocPinY", h / 2, "Height*0.5"),
            cell("FillForegnd", fill),
            cell("LineColor", stroke),
            cell("LineWeight", line_weight),
            cell("Rounding", round_value),
            cell("FillForegndTrans", transparency),
            cell("LeftMargin", 0.06),
            cell("RightMargin", 0.06),
            cell("TopMargin", 0.04),
            cell("BottomMargin", 0.04),
            cell("VerticalAlign", 1),
            char_section(text_color, font_size, bold),
            para_section(align),
            geometry("diamond" if shape == "decision" else "rect"),
            f"<Text>{escape(text)}</Text>",
            "</Shape>",
        ]
    )


def make_text(sid: int, x: float, y: float, w: float, h: float, text: str, size: float, color: str, bold: bool = False, align: int = 0) -> str:
    return make_shape(
        sid,
        x,
        y,
        w,
        h,
        text,
        BG,
        BG,
        text_color=color,
        font_size=size,
        bold=bold,
        line_weight=0,
        transparency=100,
        align=align,
    )


def make_connector(sid: int, start: tuple[float, float], end: tuple[float, float], label: str, color: str, width: float, dashed: bool) -> str:
    x1, y1 = start
    x2, y2 = end
    w = x2 - x1
    h = y2 - y1
    txt = ""
    if label:
        label_width = max(0.70, min(2.20, len(label) * 0.095))
        txt = (
            cell("TxtPinX", w / 2)
            + cell("TxtPinY", h / 2 + 0.10)
            + cell("TxtWidth", label_width)
            + cell("TxtHeight", 0.20)
            + cell("TxtLocPinX", label_width / 2)
            + cell("TxtLocPinY", 0.10)
        )
    return "".join(
        [
            f"<Shape ID='{sid}' NameU='Dynamic connector' Name='Dynamic connector' Type='Shape' Master='2'>",
            cell("PinX", (x1 + x2) / 2),
            cell("PinY", (y1 + y2) / 2),
            cell("Width", w, "GUARD(EndX-BeginX)"),
            cell("Height", h, "GUARD(EndY-BeginY)"),
            cell("LocPinX", w / 2),
            cell("LocPinY", h / 2),
            cell("BeginX", x1),
            cell("BeginY", y1),
            cell("EndX", x2),
            cell("EndY", y2),
            cell("LineColor", color),
            cell("LineWeight", width / 72),
            cell("LinePattern", 2 if dashed else 1),
            cell("EndArrow", 4),
            cell("EndArrowSize", 2),
            txt,
            char_section("#475569", 0.075, False),
            para_section(1),
            "<Section N='Geometry' IX='0'>"
            "<Row T='MoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            f"<Row T='LineTo' IX='2'><Cell N='X' V='{escape(str(w))}'/><Cell N='Y' V='{escape(str(h))}'/></Row>"
            "</Section>",
            f"<Text>{escape(label)}</Text>" if label else "",
            "</Shape>",
        ]
    )


def node_text(node: Node) -> str:
    return "\n".join((node.title, *node.body))


def page_xml(lanes: list[Lane], nodes: list[Node], edges: list[Edge]) -> str:
    shapes: list[str] = []
    sid = 1
    shapes.append(make_shape(sid, 0, 0, PAGE_W, PAGE_H, "", BG, BG, line_weight=0))
    sid += 1
    shapes.append(make_text(sid, 0.55, PAGE_H - 0.62, 9.8, 0.34, TITLE, 0.185, INK, True, 0))
    sid += 1
    shapes.append(make_text(sid, 8.90, PAGE_H - 0.52, 7.05, 0.24, SUBTITLE, 0.075, "#5d687a", False, 2))
    sid += 1

    for lane in lanes:
        shapes.append(make_shape(sid, lane.x, lane.y, lane.w, lane.h, "", lane.fill, lane.stroke, line_weight=0.007, rounding=0.04, transparency=15))
        sid += 1
        shapes.append(make_text(sid, lane.x + 0.13, lane.y + lane.h - 0.30, 2.80, 0.22, lane.title, 0.090, "#334155", True, 0))
        sid += 1

    by_key = {node.key: node for node in nodes}
    for edge in edges:
        start, end = edge_points(edge, by_key)
        shapes.append(make_connector(sid, start, end, edge.label, edge.color, edge.width, edge.dashed))
        sid += 1

    for node in nodes:
        shapes.append(
            make_shape(
                sid,
                node.x,
                node.y,
                node.w,
                node.h,
                node_text(node),
                node.fill,
                node.stroke,
                shape=node.shape,
                font_size=0.105 if node.shape == "decision" else 0.112,
                line_weight=0.014,
                rounding=0.08,
            )
        )
        sid += 1

    page_sheet = (
        "<PageSheet LineStyle='0' FillStyle='0' TextStyle='0'>"
        f"{cell('PageWidth', PAGE_W)}{cell('PageHeight', PAGE_H)}"
        f"{cell('PageScale', 0.03937007874015748)}{cell('DrawingScale', 0.03937007874015748)}"
        f"{cell('DrawingSizeType', 0)}{cell('DrawingScaleType', 0)}{cell('DrawingResizeType', 1)}"
        "</PageSheet>"
    )
    return (
        "<?xml version='1.0' encoding='utf-8'?>"
        f"<PageContents xmlns='{NS}' xmlns:r='{R_NS}' xml:space='preserve'>"
        f"{page_sheet}<Shapes>{''.join(shapes)}</Shapes></PageContents>"
    )


def pages_xml() -> str:
    page_sheet = (
        "<PageSheet LineStyle='0' FillStyle='0' TextStyle='0'>"
        f"{cell('PageWidth', PAGE_W)}{cell('PageHeight', PAGE_H)}"
        f"{cell('PageScale', 0.03937007874015748)}{cell('DrawingScale', 0.03937007874015748)}"
        f"{cell('DrawingSizeType', 0)}{cell('DrawingScaleType', 0)}{cell('DrawingResizeType', 1)}"
        "</PageSheet>"
    )
    return (
        "<?xml version='1.0' encoding='utf-8'?>"
        f"<Pages xmlns='{NS}' xmlns:r='{R_NS}' xml:space='preserve'>"
        f"<Page ID='0' NameU='程序流程图' Name='程序流程图' ViewScale='1' ViewCenterX='{PAGE_W / 2}' ViewCenterY='{PAGE_H / 2}'>"
        f"{page_sheet}<Rel r:id='rId1'/></Page></Pages>"
    )


def ensure_template() -> None:
    if TEMPLATE_VSDX.exists():
        return
    if not SOURCE_TEMPLATE.exists():
        raise FileNotFoundError(f"缺少 Visio 模板: {SOURCE_TEMPLATE}")
    shutil.copyfile(SOURCE_TEMPLATE, TEMPLATE_VSDX)


def write_vsdx(lanes: list[Lane], nodes: list[Node], edges: list[Edge], path: Path) -> None:
    ensure_template()
    with zipfile.ZipFile(TEMPLATE_VSDX, "r") as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "visio/pages/page1.xml":
                data = page_xml(lanes, nodes, edges).encode("utf-8")
            elif item.filename == "visio/pages/pages.xml":
                data = pages_xml().encode("utf-8")
            zout.writestr(item, data)


def find_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def to_px(point: tuple[float, float], scale: float) -> tuple[int, int]:
    x, y = point
    return round(x * scale), round((PAGE_H - y) * scale)


def rect_px(x: float, y: float, w: float, h: float, scale: float) -> tuple[int, int, int, int]:
    left, top = to_px((x, y + h), scale)
    right, bottom = to_px((x + w, y), scale)
    return left, top, right, bottom


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        current = ""
        for ch in raw:
            test = current + ch
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_w or not current:
                current = test
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str, width: int, dashed: bool) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        parts = max(10, int(math.dist(start, end) / 22))
        for index in range(parts):
            if index % 2 == 0:
                a = index / parts
                b = min(1, (index + 0.70) / parts)
                draw.line(
                    (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
                    fill=color,
                    width=width,
                )
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = max(12, width * 5)
    spread = math.radians(24)
    p1 = (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread))
    p2 = (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread))
    draw.polygon([(x2, y2), p1, p2], fill=color)


def draw_node(draw: ImageDraw.ImageDraw, node: Node, scale: float, title_font: ImageFont.ImageFont, body_font: ImageFont.ImageFont) -> None:
    xy = rect_px(node.x, node.y, node.w, node.h, scale)
    shadow = (xy[0] + 5, xy[1] + 5, xy[2] + 5, xy[3] + 5)
    draw.rounded_rectangle(shadow, radius=12, fill="#d8dee8")
    if node.shape == "decision":
        cx = (xy[0] + xy[2]) / 2
        cy = (xy[1] + xy[3]) / 2
        points = [(cx, xy[1]), (xy[2], cy), (cx, xy[3]), (xy[0], cy)]
        draw.polygon(points, fill=node.fill, outline=node.stroke)
    else:
        radius = 22 if node.shape == "terminator" else 10
        draw.rounded_rectangle(xy, radius=radius, fill=node.fill, outline=node.stroke, width=3)

    max_w = xy[2] - xy[0] - 20
    title_lines = wrap_text(draw, node.title, title_font, max_w)
    body_lines: list[str] = []
    for raw in node.body:
        body_lines.extend(wrap_text(draw, raw, body_font, max_w))

    total_h = len(title_lines) * (title_font.size + 2) + max(0, len(body_lines)) * (body_font.size + 1)
    if body_lines:
        total_h += 3
    y = max(xy[1] + 8, (xy[1] + xy[3] - total_h) / 2)

    for line in title_lines:
        draw.text(((xy[0] + xy[2]) / 2, y), line, fill=INK, font=title_font, anchor="ma")
        y += title_font.size + 2
    y += 3
    for line in body_lines:
        draw.text(((xy[0] + xy[2]) / 2, y), line, fill=MUTED, font=body_font, anchor="ma")
        y += body_font.size + 1


def render_png(lanes: list[Lane], nodes: list[Node], edges: list[Edge], path: Path) -> None:
    scale = 185
    image = Image.new("RGB", (round(PAGE_W * scale), round(PAGE_H * scale)), BG)
    draw = ImageDraw.Draw(image)

    title_font = find_font(38, True)
    subtitle_font = find_font(17)
    lane_font = find_font(20, True)
    node_title_font = find_font(20, True)
    node_body_font = find_font(17)
    edge_font = find_font(15)

    draw.text(to_px((0.55, PAGE_H - 0.34), scale), TITLE, fill=INK, font=title_font, anchor="la")
    draw.text(to_px((15.95, PAGE_H - 0.39), scale), SUBTITLE, fill="#5d687a", font=subtitle_font, anchor="ra")

    for lane in lanes:
        xy = rect_px(lane.x, lane.y, lane.w, lane.h, scale)
        draw.rounded_rectangle(xy, radius=12, fill=lane.fill, outline=lane.stroke, width=2)
        draw.text(to_px((lane.x + 0.14, lane.y + lane.h - 0.14), scale), lane.title, fill="#334155", font=lane_font, anchor="la")

    by_key = {node.key: node for node in nodes}
    for edge in edges:
        start, end = edge_points(edge, by_key)
        s = to_px(start, scale)
        e = to_px(end, scale)
        draw_arrow(draw, s, e, edge.color, max(2, round(edge.width * 1.35)), edge.dashed)
        if edge.label:
            mx = (s[0] + e[0]) / 2
            my = (s[1] + e[1]) / 2 - 10
            bbox = draw.textbbox((0, 0), edge.label, font=edge_font)
            pad = 5
            draw.rounded_rectangle(
                (mx - (bbox[2] - bbox[0]) / 2 - pad, my - 10, mx + (bbox[2] - bbox[0]) / 2 + pad, my + 10),
                radius=5,
                fill=BG,
                outline="#e6ebf2",
            )
            draw.text((mx, my - 1), edge.label, fill="#475569", font=edge_font, anchor="mm")

    for node in nodes:
        draw_node(draw, node, scale, node_title_font, node_body_font)

    image.save(path, quality=95)


def svg_text(text: str, x: float, y: float, size: int, color: str, weight: int = 400, anchor_text: str = "middle") -> str:
    return f"<text x='{x:.1f}' y='{y:.1f}' fill='{color}' font-size='{size}' font-weight='{weight}' text-anchor='{anchor_text}' font-family='Microsoft YaHei, SimHei, Arial'>{escape(text)}</text>"


def render_svg(lanes: list[Lane], nodes: list[Node], edges: list[Edge], path: Path) -> None:
    scale = 92
    w = PAGE_W * scale
    h = PAGE_H * scale

    def sx(value: float) -> float:
        return value * scale

    def sy(value: float) -> float:
        return (PAGE_H - value) * scale

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w:.0f}' height='{h:.0f}' viewBox='0 0 {w:.0f} {h:.0f}'>",
        "<defs><marker id='arrow' markerWidth='10' markerHeight='8' refX='9' refY='4' orient='auto'><path d='M0,0 L10,4 L0,8 z' fill='#526072'/></marker></defs>",
        f"<rect x='0' y='0' width='{w:.0f}' height='{h:.0f}' fill='{BG}'/>",
        svg_text(TITLE, sx(0.55), sy(PAGE_H - 0.36), 25, INK, 700, "start"),
        svg_text(SUBTITLE, sx(15.95), sy(PAGE_H - 0.40), 11, "#5d687a", 400, "end"),
    ]

    for lane in lanes:
        parts.append(f"<rect x='{sx(lane.x):.1f}' y='{sy(lane.y + lane.h):.1f}' width='{sx(lane.w):.1f}' height='{sx(lane.h):.1f}' rx='9' fill='{lane.fill}' stroke='{lane.stroke}'/>")
        parts.append(svg_text(lane.title, sx(lane.x + 0.14), sy(lane.y + lane.h - 0.14), 13, "#334155", 700, "start"))

    by_key = {node.key: node for node in nodes}
    for edge in edges:
        start, end = edge_points(edge, by_key)
        dash = " stroke-dasharray='8,6'" if edge.dashed else ""
        parts.append(f"<line x1='{sx(start[0]):.1f}' y1='{sy(start[1]):.1f}' x2='{sx(end[0]):.1f}' y2='{sy(end[1]):.1f}' stroke='{edge.color}' stroke-width='{edge.width:.1f}' marker-end='url(#arrow)'{dash}/>")
        if edge.label:
            parts.append(svg_text(edge.label, (sx(start[0]) + sx(end[0])) / 2, (sy(start[1]) + sy(end[1])) / 2 - 6, 10, "#475569"))

    for node in nodes:
        x = sx(node.x)
        y = sy(node.y + node.h)
        nw = sx(node.w)
        nh = sx(node.h)
        if node.shape == "decision":
            points = f"{x + nw / 2:.1f},{y:.1f} {x + nw:.1f},{y + nh / 2:.1f} {x + nw / 2:.1f},{y + nh:.1f} {x:.1f},{y + nh / 2:.1f}"
            parts.append(f"<polygon points='{points}' fill='{node.fill}' stroke='{node.stroke}' stroke-width='1.5'/>")
        else:
            rx = 16 if node.shape == "terminator" else 8
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{nw:.1f}' height='{nh:.1f}' rx='{rx}' fill='{node.fill}' stroke='{node.stroke}' stroke-width='1.7'/>")
        ty = y + 22
        parts.append(svg_text(node.title, x + nw / 2, ty, 12, INK, 700))
        ty += 15
        for raw in node.body[:3]:
            parts.append(svg_text(raw, x + nw / 2, ty, 10, MUTED))
            ty += 13

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_vdx(lanes: list[Lane], nodes: list[Node], edges: list[Edge], path: Path) -> None:
    xml = [
        "<?xml version='1.0' encoding='utf-8'?>",
        "<VisioDocument xmlns='urn:schemas-microsoft-com:office:visio'>",
        f"<DocumentProperties><Title>{escape(TITLE)}</Title></DocumentProperties>",
        f"<Pages><Page ID='0' Name='程序流程图'><PageSheet><PageProps><PageWidth>{PAGE_W:.2f}</PageWidth><PageHeight>{PAGE_H:.2f}</PageHeight></PageProps></PageSheet><Shapes>",
    ]
    sid = 1
    for lane in lanes:
        xml.append(f"<Shape ID='{sid}' Type='Shape'><XForm><PinX>{lane.x+lane.w/2:.2f}</PinX><PinY>{lane.y+lane.h/2:.2f}</PinY><Width>{lane.w:.2f}</Width><Height>{lane.h:.2f}</Height></XForm><Text>{escape(lane.title)}</Text></Shape>")
        sid += 1
    for node in nodes:
        xml.append(f"<Shape ID='{sid}' Type='Shape'><XForm><PinX>{node.x+node.w/2:.2f}</PinX><PinY>{node.y+node.h/2:.2f}</PinY><Width>{node.w:.2f}</Width><Height>{node.h:.2f}</Height></XForm><Text>{escape(node_text(node))}</Text></Shape>")
        sid += 1
    xml.append("</Shapes></Page></Pages></VisioDocument>")
    path.write_text("".join(xml), encoding="utf-8")


def main() -> None:
    lanes, nodes, edges = model()
    vsdx = OUT_DIR / "STM32G431程序流程图.vsdx"
    vdx = OUT_DIR / "STM32G431程序流程图.vdx"
    png = OUT_DIR / "STM32G431程序流程图.png"
    svg = OUT_DIR / "STM32G431程序流程图.svg"
    write_vsdx(lanes, nodes, edges, vsdx)
    write_vdx(lanes, nodes, edges, vdx)
    render_png(lanes, nodes, edges, png)
    render_svg(lanes, nodes, edges, svg)
    print(vsdx)
    print(vdx)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
