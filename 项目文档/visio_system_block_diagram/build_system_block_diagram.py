from __future__ import annotations

import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
TEMPLATE_VSDX = OUT_DIR / "visio_base_template.vsdx"

PAGE_W = 16.54
PAGE_H = 9.30
TITLE = "STM32G431 智能频响测量与 AI 诊断系统框图"
SUBTITLE = "硬件扫频采集、三数组协议、上位机频响分析、AI诊断与控制校正闭环"

NS = "http://schemas.microsoft.com/office/visio/2012/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

BG = "#f6f8fb"
BOARD = "#ffffff"
INK = "#172033"
MUTED = "#667085"
BORDER = "#d8e0ea"
LINE = "#8794a8"
NAVY = "#1f4e79"


@dataclass(frozen=True)
class Group:
    title: str
    x: float
    y: float
    w: float
    h: float
    accent: str
    fill: str


@dataclass(frozen=True)
class Card:
    key: str
    x: float
    y: float
    w: float
    h: float
    title: str
    lines: tuple[str, ...]
    accent: str
    fill: str = BOARD
    tag: str = ""


@dataclass(frozen=True)
class Arrow:
    start: str | tuple[float, float]
    end: str | tuple[float, float]
    label: str = ""
    color: str = LINE
    width: float = 1.5
    dashed: bool = False
    arrow: bool = True


def model() -> tuple[list[Group], list[Card], list[Arrow]]:
    groups = [
        Group("主信号链路", 0.82, 4.95, 14.90, 2.36, "#1f4e79", "#eef6ff"),
        Group("关键能力支撑", 0.82, 2.02, 14.90, 1.86, "#00a884", "#f0fbf6"),
    ]

    cards = [
        Card(
            "dut",
            1.00,
            5.54,
            2.06,
            1.16,
            "被测电路",
            ("滤波器 / DUT", "输入、输出接入"),
            "#2f9e44",
            tag="01",
        ),
        Card(
            "stm32",
            3.46,
            5.54,
            2.06,
            1.16,
            "STM32G431 测量",
            ("DAC 激励", "ADC 采样 + FFT"),
            "#2f80ed",
            tag="02",
        ),
        Card(
            "serial",
            5.92,
            5.54,
            2.06,
            1.16,
            "三数组串口协议",
            ("omega", "Magnitude_data / Phase_data_rad"),
            "#d08a1d",
            tag="03",
        ),
        Card(
            "pc",
            8.38,
            5.54,
            2.06,
            1.16,
            "上位机分析",
            ("Bode / Nyquist", "稳定性与类型识别"),
            "#1592a6",
            tag="04",
        ),
        Card(
            "ai",
            10.84,
            5.54,
            2.06,
            1.16,
            "AI 诊断校正",
            ("故障候选排序", "PM/GM 补偿"),
            "#7a5cff",
            tag="05",
        ),
        Card(
            "output",
            13.30,
            5.54,
            2.06,
            1.16,
            "报告与闭环",
            ("HTML / CSV / PNG", "补扫 SWEEP"),
            "#2f9e44",
            tag="06",
        ),
        Card(
            "firmware",
            1.10,
            2.48,
            3.16,
            0.90,
            "下位机固件",
            ("SWEEP 命令解析，定时器/DMA/DAC/ADC 协同",),
            "#2f80ed",
        ),
        Card(
            "analysis",
            4.62,
            2.48,
            3.16,
            0.90,
            "频响分析算法",
            ("平滑、截止/中心频率、Nyquist 稳定性判断",),
            "#00a884",
        ),
        Card(
            "diagnosis",
            8.14,
            2.48,
            3.16,
            0.90,
            "AI 规则库",
            ("特征提取、模板拟合、故障规则、补扫计划",),
            "#7a5cff",
        ),
        Card(
            "comp",
            11.66,
            2.48,
            3.16,
            0.90,
            "自动控制校正",
            ("目标 PM/GM，超前/滞后，输出 Gc(s) 与测试建议",),
            "#d17b2f",
        ),
        Card(
            "resweep",
            1.10,
            1.02,
            4.16,
            0.62,
            "智能补扫闭环",
            ("置信度不足或关键频段缺失时，回写 SWEEP 加密测量",),
            "#2f9e44",
        ),
        Card(
            "visual",
            6.18,
            1.02,
            3.58,
            0.62,
            "图表与报告",
            ("Bode / Nyquist / HTML 报告 / CSV 数据 / PNG 图",),
            "#1592a6",
        ),
        Card(
            "advice",
            10.68,
            1.02,
            4.16,
            0.62,
            "诊断结论与调试建议",
            ("电路类型、故障候选、校正器参数、下一步测试",),
            "#d17b2f",
        ),
    ]

    arrows = [
        Arrow("dut", "stm32", "信号接入"),
        Arrow("stm32", "serial", "三数组帧"),
        Arrow("serial", "pc", "串口读取"),
        Arrow("pc", "ai", "分析结果"),
        Arrow("ai", "output", "结论/参数"),
        Arrow("output", (14.33, 7.03), "", "#2f9e44", 1.10, True, False),
        Arrow((14.33, 7.03), (4.49, 7.03), "低置信度触发补扫", "#2f9e44", 1.10, True, False),
        Arrow((4.49, 7.03), "stm32", "SWEEP", "#2f9e44", 1.10, True, True),
    ]
    return groups, cards, arrows


def card_center(card: Card) -> tuple[float, float]:
    return card.x + card.w / 2, card.y + card.h / 2


def anchor(card: Card, toward: tuple[float, float]) -> tuple[float, float]:
    cx, cy = card_center(card)
    dx = toward[0] - cx
    dy = toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (card.w / 2) / abs(dx) if dx else float("inf")
    sy = (card.h / 2) / abs(dy) if dy else float("inf")
    t = min(sx, sy)
    return cx + dx * t, cy + dy * t


def arrow_points(arrow: Arrow, by_key: dict[str, Card]) -> tuple[tuple[float, float], tuple[float, float]]:
    def center(value: str | tuple[float, float]) -> tuple[float, float]:
        return value if isinstance(value, tuple) else card_center(by_key[value])

    start_center = center(arrow.start)
    end_center = center(arrow.end)
    start = anchor(by_key[arrow.start], end_center) if isinstance(arrow.start, str) else start_center
    end = anchor(by_key[arrow.end], start_center) if isinstance(arrow.end, str) else end_center
    return start, end


def cell(name: str, value: str | float, formula: str | None = None) -> str:
    attrs = f"N='{name}' V='{escape(str(value))}'"
    if formula is not None:
        attrs += f" F='{escape(formula)}'"
    return f"<Cell {attrs}/>"


def char_section(color: str, size: float = 0.09, bold: bool = False) -> str:
    return (
        "<Section N='Character'><Row IX='0'>"
        f"{cell('Color', color)}{cell('Size', size)}{cell('Style', 17 if bold else 0)}{cell('Font', 0)}"
        "</Row></Section>"
    )


def para_section(align: int = 1) -> str:
    return f"<Section N='Paragraph'><Row IX='0'>{cell('HorzAlign', align)}</Row></Section>"


def geometry_rect() -> str:
    return (
        "<Section N='Geometry' IX='0'>"
        f"{cell('NoFill', 0)}{cell('NoLine', 0)}{cell('NoShow', 0)}{cell('NoSnap', 0)}{cell('NoQuickDrag', 0)}"
        "<Row T='RelMoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
        "<Row T='RelLineTo' IX='2'><Cell N='X' V='1'/><Cell N='Y' V='0'/></Row>"
        "<Row T='RelLineTo' IX='3'><Cell N='X' V='1'/><Cell N='Y' V='1'/></Row>"
        "<Row T='RelLineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='1'/></Row>"
        "<Row T='RelLineTo' IX='5'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
        "</Section>"
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
    text_color: str = INK,
    font_size: float = 0.082,
    bold: bool = False,
    line_weight: float = 0.010,
    rounding: float = 0.06,
    transparency: int = 0,
    align: int = 1,
) -> str:
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
            cell("Rounding", rounding),
            cell("FillForegndTrans", transparency),
            cell("LeftMargin", 0.07),
            cell("RightMargin", 0.07),
            cell("TopMargin", 0.04),
            cell("BottomMargin", 0.04),
            cell("VerticalAlign", 1),
            char_section(text_color, font_size, bold),
            para_section(align),
            geometry_rect(),
            f"<Text>{escape(text)}</Text>",
            "</Shape>",
        ]
    )


def make_connector(
    sid: int,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    color: str = LINE,
    width: float = 1.5,
    dashed: bool = False,
    arrow: bool = True,
) -> str:
    x1, y1 = start
    x2, y2 = end
    w = x2 - x1
    h = y2 - y1
    label_xml = ""
    if label:
        label_width = max(0.70, min(1.80, len(label) * 0.08))
        label_xml = (
            cell("TxtPinX", w / 2)
            + cell("TxtPinY", h / 2 + 0.10)
            + cell("TxtWidth", label_width)
            + cell("TxtHeight", 0.18)
            + cell("TxtLocPinX", label_width / 2)
            + cell("TxtLocPinY", 0.09)
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
            cell("EndArrow", 4 if arrow else 0),
            cell("EndArrowSize", 2),
            label_xml,
            char_section("#475569", 0.060, False),
            para_section(1),
            "<Section N='Geometry' IX='0'>"
            "<Row T='MoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            f"<Row T='LineTo' IX='2'><Cell N='X' V='{escape(str(w))}'/><Cell N='Y' V='{escape(str(h))}'/></Row>"
            "</Section>",
            f"<Text>{escape(label)}</Text>" if label else "",
            "</Shape>",
        ]
    )


def card_text(card: Card) -> str:
    tag = f"{card.tag}  " if card.tag else ""
    return "\n".join((f"{tag}{card.title}", *card.lines))


def page_xml(groups: list[Group], cards: list[Card], arrows: list[Arrow]) -> str:
    shapes: list[str] = []
    sid = 1
    shapes.append(make_shape(sid, 0, 0, PAGE_W, PAGE_H, "", BG, BG, line_weight=0))
    sid += 1
    shapes.append(make_shape(sid, 0.55, 0.72, 15.44, 7.56, "", BOARD, "#e8edf4", line_weight=0.006, rounding=0.10))
    sid += 1
    shapes.append(make_shape(sid, 0.58, PAGE_H - 0.68, 9.80, 0.36, TITLE, BG, BG, text_color=INK, font_size=0.18, bold=True, line_weight=0, transparency=100, align=0))
    sid += 1
    shapes.append(make_shape(sid, 8.42, PAGE_H - 0.56, 7.44, 0.25, SUBTITLE, BG, BG, text_color=MUTED, font_size=0.064, line_weight=0, transparency=100, align=2))
    sid += 1
    shapes.append(make_shape(sid, 0.72, 7.88, 15.10, 0.06, "", NAVY, NAVY, line_weight=0, rounding=0.02))
    sid += 1

    for group in groups:
        shapes.append(make_shape(sid, group.x, group.y, group.w, group.h, "", group.fill, group.accent, line_weight=0.005, rounding=0.08, transparency=12))
        sid += 1
        shapes.append(make_shape(sid, group.x + 0.15, group.y + group.h - 0.34, 2.25, 0.22, group.title, group.fill, group.fill, text_color=group.accent, font_size=0.067, bold=True, line_weight=0, transparency=100, align=0))
        sid += 1

    by_key = {card.key: card for card in cards}
    for arrow in arrows:
        start, end = arrow_points(arrow, by_key)
        shapes.append(make_connector(sid, start, end, arrow.label, arrow.color, arrow.width, arrow.dashed, arrow.arrow))
        sid += 1

    for card in cards:
        shapes.append(make_shape(sid, card.x + 0.04, card.y - 0.04, card.w, card.h, "", "#d9e1ec", "#d9e1ec", line_weight=0, rounding=0.08, transparency=18))
        sid += 1
        shapes.append(make_shape(sid, card.x, card.y, card.w, card.h, card_text(card), card.fill, BORDER, text_color=INK, font_size=0.070, bold=False, line_weight=0.010, rounding=0.08))
        sid += 1
        shapes.append(make_shape(sid, card.x, card.y + card.h - 0.08, card.w, 0.08, "", card.accent, card.accent, line_weight=0, rounding=0.02))
        sid += 1

    chips = [
        ("测量协议稳定：SWEEP -> 三数组输出", 0.90, 0.42, 3.38),
        ("算法链路完整：频响、稳定性、滤波器类型、故障候选", 4.58, 0.42, 4.78),
        ("闭环能力突出：AI补扫 + 控制校正 + 报告导出", 9.70, 0.42, 4.38),
    ]
    for text, x, y, w in chips:
        shapes.append(make_shape(sid, x, y, w, 0.32, text, BOARD, "#d8e0ea", text_color=MUTED, font_size=0.057, line_weight=0.006, rounding=0.14))
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
        f"<Page ID='0' NameU='系统框图' Name='系统框图' ViewScale='1' ViewCenterX='{PAGE_W / 2}' ViewCenterY='{PAGE_H / 2}'>"
        f"{page_sheet}<Rel r:id='rId1'/></Page></Pages>"
    )


def ensure_template() -> None:
    if TEMPLATE_VSDX.exists():
        return
    raise FileNotFoundError(f"缺少 Visio 模板: {TEMPLATE_VSDX}")


def write_vsdx(groups: list[Group], cards: list[Card], arrows: list[Arrow], path: Path) -> None:
    ensure_template()
    with zipfile.ZipFile(TEMPLATE_VSDX, "r") as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "visio/pages/page1.xml":
                data = page_xml(groups, cards, arrows).encode("utf-8")
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


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int,
    *,
    dashed: bool = False,
    arrow: bool = True,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        parts = max(8, int(math.dist(start, end) / 18))
        for index in range(parts):
            if index % 2 == 0:
                a = index / parts
                b = (index + 0.70) / parts
                draw.line(
                    (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
                    fill=color,
                    width=width,
                )
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    if not arrow:
        return
    angle = math.atan2(y2 - y1, x2 - x1)
    length = max(10, width * 4)
    spread = math.radians(24)
    p1 = (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread))
    p2 = (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread))
    draw.polygon([(x2, y2), p1, p2], fill=color)


def draw_label(draw: ImageDraw.ImageDraw, text: str, point: tuple[int, int], font: ImageFont.ImageFont) -> None:
    if not text:
        return
    x, y = point
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.rounded_rectangle((x - w / 2 - 7, y - h / 2 - 5, x + w / 2 + 7, y + h / 2 + 5), radius=6, fill="#ffffff", outline="#e4e9f0")
    draw.text((x, y - 1), text, fill="#475569", font=font, anchor="mm")


def draw_card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: tuple[str, ...],
    accent: str,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    *,
    tag: str = "",
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle((x1 + 6, y1 + 7, x2 + 6, y2 + 7), radius=14, fill="#dce3ec")
    draw.rounded_rectangle(xy, radius=14, fill=BOARD, outline=BORDER, width=2)
    draw.rounded_rectangle((x1, y1, x2, y1 + 12), radius=14, fill=accent)
    draw.rectangle((x1, y1 + 8, x2, y1 + 13), fill=accent)

    max_w = x2 - x1 - 26
    y = y1 + 25
    if tag:
        badge = (x1 + 14, y1 + 22, x1 + 52, y1 + 52)
        draw.ellipse(badge, fill=accent)
        draw.text(((badge[0] + badge[2]) / 2, (badge[1] + badge[3]) / 2), tag, fill="#ffffff", font=body_font, anchor="mm")
        text_x = (x1 + x2) / 2 + 8
        available = max_w - 42
    else:
        text_x = (x1 + x2) / 2
        available = max_w

    for line in wrap_text(draw, title, title_font, available):
        draw.text((text_x, y), line, fill=INK, font=title_font, anchor="ma")
        y += title_font.size + 2
    y += 2
    for raw in lines:
        for line in wrap_text(draw, raw, body_font, available):
            draw.text((text_x, y), line, fill=MUTED, font=body_font, anchor="ma")
            y += body_font.size + 1


def render_png(groups: list[Group], cards: list[Card], arrows: list[Arrow], path: Path) -> None:
    scale = 160
    image = Image.new("RGB", (round(PAGE_W * scale), round(PAGE_H * scale)), BG)
    draw = ImageDraw.Draw(image)

    title_font = find_font(38, True)
    subtitle_font = find_font(15)
    group_font = find_font(16, True)
    card_title = find_font(18, True)
    card_body = find_font(13)
    small_title = find_font(16, True)
    small_body = find_font(12)
    label_font = find_font(12)
    chip_font = find_font(13)

    board_xy = rect_px(0.55, 0.72, 15.44, 7.56, scale)
    draw.rounded_rectangle((board_xy[0] + 8, board_xy[1] + 10, board_xy[2] + 8, board_xy[3] + 10), radius=18, fill="#dfe6ef")
    draw.rounded_rectangle(board_xy, radius=18, fill=BOARD, outline="#e8edf4", width=1)

    draw.text(to_px((0.58, PAGE_H - 0.30), scale), TITLE, fill=INK, font=title_font, anchor="la")
    draw.text(to_px((15.90, PAGE_H - 0.36), scale), SUBTITLE, fill=MUTED, font=subtitle_font, anchor="ra")
    draw.rounded_rectangle(rect_px(0.72, 7.88, 15.10, 0.06, scale), radius=6, fill=NAVY)

    for group in groups:
        xy = rect_px(group.x, group.y, group.w, group.h, scale)
        draw.rounded_rectangle(xy, radius=16, fill=group.fill, outline=group.accent, width=1)
        draw.text(to_px((group.x + 0.16, group.y + group.h - 0.13), scale), group.title, fill=group.accent, font=group_font, anchor="la")

    by_key = {card.key: card for card in cards}
    for arrow in arrows:
        start, end = arrow_points(arrow, by_key)
        s = to_px(start, scale)
        e = to_px(end, scale)
        draw_arrow(draw, s, e, arrow.color, max(2, round(arrow.width * 2)), dashed=arrow.dashed, arrow=arrow.arrow)
        if arrow.label:
            draw_label(draw, arrow.label, ((s[0] + e[0]) // 2, (s[1] + e[1]) // 2 - 12), label_font)

    for card in cards:
        main = card.tag != ""
        draw_card(
            draw,
            rect_px(card.x, card.y, card.w, card.h, scale),
            card.title,
            card.lines,
            card.accent,
            card_title if main else small_title,
            card_body if main else small_body,
            tag=card.tag,
        )

    chips = [
        ("测量协议稳定：SWEEP -> 三数组输出", 0.90, 0.42, 3.38),
        ("算法链路完整：频响、稳定性、滤波器类型、故障候选", 4.58, 0.42, 4.78),
        ("闭环能力突出：AI补扫 + 控制校正 + 报告导出", 9.70, 0.42, 4.38),
    ]
    for text, x, y, w in chips:
        xy = rect_px(x, y, w, 0.32, scale)
        draw.rounded_rectangle(xy, radius=18, fill=BOARD, outline="#d8e0ea", width=1)
        draw.text(((xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2 - 1), text, fill=MUTED, font=chip_font, anchor="mm")

    image.save(path, quality=95)


def svg_text(text: str, x: float, y: float, size: int, color: str, weight: int = 400, anchor_text: str = "middle") -> str:
    return f"<text x='{x:.1f}' y='{y:.1f}' fill='{color}' font-size='{size}' font-weight='{weight}' text-anchor='{anchor_text}' font-family='Microsoft YaHei, SimHei, Arial'>{escape(text)}</text>"


def render_svg(groups: list[Group], cards: list[Card], arrows: list[Arrow], path: Path) -> None:
    scale = 92
    width = PAGE_W * scale
    height = PAGE_H * scale

    def sx(value: float) -> float:
        return value * scale

    def sy(value: float) -> float:
        return (PAGE_H - value) * scale

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' height='{height:.0f}' viewBox='0 0 {width:.0f} {height:.0f}'>",
        "<defs><filter id='shadow' x='-20%' y='-20%' width='140%' height='140%'><feDropShadow dx='3' dy='4' stdDeviation='3' flood-color='#b8c2d0' flood-opacity='.45'/></filter><marker id='arrow' markerWidth='10' markerHeight='8' refX='9' refY='4' orient='auto'><path d='M0,0 L10,4 L0,8 z' fill='#8794a8'/></marker></defs>",
        f"<rect x='0' y='0' width='{width:.0f}' height='{height:.0f}' fill='{BG}'/>",
        f"<rect x='{sx(0.55):.1f}' y='{sy(0.72 + 7.56):.1f}' width='{sx(15.44):.1f}' height='{sx(7.56):.1f}' rx='16' fill='{BOARD}' stroke='#e8edf4' filter='url(#shadow)'/>",
        svg_text(TITLE, sx(0.58), sy(PAGE_H - 0.30), 23, INK, 700, "start"),
        svg_text(SUBTITLE, sx(15.90), sy(PAGE_H - 0.36), 9, MUTED, 400, "end"),
        f"<rect x='{sx(0.72):.1f}' y='{sy(7.94):.1f}' width='{sx(15.10):.1f}' height='{sx(0.06):.1f}' rx='4' fill='{NAVY}'/>",
    ]

    for group in groups:
        parts.append(f"<rect x='{sx(group.x):.1f}' y='{sy(group.y + group.h):.1f}' width='{sx(group.w):.1f}' height='{sx(group.h):.1f}' rx='13' fill='{group.fill}' stroke='{group.accent}' opacity='.88'/>")
        parts.append(svg_text(group.title, sx(group.x + 0.16), sy(group.y + group.h - 0.14), 10, group.accent, 700, "start"))

    by_key = {card.key: card for card in cards}
    for arrow in arrows:
        start, end = arrow_points(arrow, by_key)
        dash = " stroke-dasharray='7,6'" if arrow.dashed else ""
        marker = " marker-end='url(#arrow)'" if arrow.arrow else ""
        parts.append(f"<line x1='{sx(start[0]):.1f}' y1='{sy(start[1]):.1f}' x2='{sx(end[0]):.1f}' y2='{sy(end[1]):.1f}' stroke='{arrow.color}' stroke-width='{arrow.width:.1f}'{dash}{marker}/>")
        if arrow.label:
            parts.append(svg_text(arrow.label, (sx(start[0]) + sx(end[0])) / 2, (sy(start[1]) + sy(end[1])) / 2 - 5, 8, "#475569"))

    for card in cards:
        x = sx(card.x)
        y = sy(card.y + card.h)
        w = sx(card.w)
        h = sx(card.h)
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='10' fill='{BOARD}' stroke='{BORDER}' filter='url(#shadow)'/>")
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{sx(0.08):.1f}' rx='10' fill='{card.accent}'/>")
        if card.tag:
            parts.append(f"<circle cx='{x + 24:.1f}' cy='{y + 34:.1f}' r='14' fill='{card.accent}'/>")
            parts.append(svg_text(card.tag, x + 24, y + 38, 8, "#fff", 700))
        tx = x + w / 2 + (8 if card.tag else 0)
        ty = y + 32
        parts.append(svg_text(card.title, tx, ty, 10 if card.tag else 9, INK, 700))
        ty += 15
        for line in card.lines:
            parts.append(svg_text(line, tx, ty, 8, MUTED, 400))
            ty += 13

    chips = [
        ("测量协议稳定：SWEEP -> 三数组输出", 0.90, 0.42, 3.38),
        ("算法链路完整：频响、稳定性、滤波器类型、故障候选", 4.58, 0.42, 4.78),
        ("闭环能力突出：AI补扫 + 控制校正 + 报告导出", 9.70, 0.42, 4.38),
    ]
    for text, x, y, w in chips:
        parts.append(f"<rect x='{sx(x):.1f}' y='{sy(y + 0.32):.1f}' width='{sx(w):.1f}' height='{sx(0.32):.1f}' rx='14' fill='{BOARD}' stroke='#d8e0ea'/>")
        parts.append(svg_text(text, sx(x + w / 2), sy(y + 0.15), 8, MUTED))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_vdx(groups: list[Group], cards: list[Card], arrows: list[Arrow], path: Path) -> None:
    xml = [
        "<?xml version='1.0' encoding='utf-8'?>",
        "<VisioDocument xmlns='urn:schemas-microsoft-com:office:visio'>",
        f"<DocumentProperties><Title>{escape(TITLE)}</Title></DocumentProperties>",
        f"<Pages><Page ID='0' Name='系统框图'><PageSheet><PageProps><PageWidth>{PAGE_W:.2f}</PageWidth><PageHeight>{PAGE_H:.2f}</PageHeight></PageProps></PageSheet><Shapes>",
    ]
    sid = 1
    for group in groups:
        xml.append(f"<Shape ID='{sid}' Type='Shape'><XForm><PinX>{group.x+group.w/2:.2f}</PinX><PinY>{group.y+group.h/2:.2f}</PinY><Width>{group.w:.2f}</Width><Height>{group.h:.2f}</Height></XForm><Text>{escape(group.title)}</Text></Shape>")
        sid += 1
    for card in cards:
        xml.append(f"<Shape ID='{sid}' Type='Shape'><XForm><PinX>{card.x+card.w/2:.2f}</PinX><PinY>{card.y+card.h/2:.2f}</PinY><Width>{card.w:.2f}</Width><Height>{card.h:.2f}</Height></XForm><Text>{escape(card_text(card))}</Text></Shape>")
        sid += 1
    xml.append("</Shapes></Page></Pages></VisioDocument>")
    path.write_text("".join(xml), encoding="utf-8")


def main() -> None:
    groups, cards, arrows = model()
    vsdx = OUT_DIR / "STM32G431智能频响测量与AI诊断系统框图.vsdx"
    vdx = OUT_DIR / "STM32G431智能频响测量与AI诊断系统框图.vdx"
    svg = OUT_DIR / "STM32G431智能频响测量与AI诊断系统框图.svg"
    png = OUT_DIR / "STM32G431智能频响测量与AI诊断系统框图.png"
    write_vsdx(groups, cards, arrows, vsdx)
    write_vdx(groups, cards, arrows, vdx)
    render_svg(groups, cards, arrows, svg)
    render_png(groups, cards, arrows, png)
    print(vsdx)
    print(vdx)
    print(svg)
    print(png)


if __name__ == "__main__":
    main()
