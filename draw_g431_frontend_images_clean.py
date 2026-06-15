from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(r"D:\ai")


def get_font(size, bold=False):
    paths = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = {
    "title": get_font(42, True),
    "section": get_font(28, True),
    "part": get_font(22, True),
    "text": get_font(20),
    "small": get_font(17),
    "tiny": get_font(14),
    "net": get_font(18, True),
}

BG = "#fffdf7"
GRID = "#efe7da"
GRID2 = "#ded2bf"
WIRE = "#13803a"
RED = "#b00000"
BLUE = "#0d50c9"
TEXT = "#111111"
NOTE = "#3d3d3d"
BOX = "#fff7e8"
BOX2 = "#f8fff2"
PCB = "#1f7f45"
TOP = "#cf3a2c"
BOT = "#2458c9"
GND = "#2d9658"
SILK = "#f8f2e7"
PAD = "#ffd15a"


def txt(d, x, y, s, style="text", fill=TEXT, anchor=None):
    d.text((x, y), s, font=FONT[style], fill=fill, anchor=anchor)


def draw_grid(d, w, h):
    for x in range(0, w + 1, 25):
        d.line((x, 0, x, h), fill=GRID, width=1)
    for y in range(0, h + 1, 25):
        d.line((0, y, w, y), fill=GRID, width=1)
    for x in range(0, w + 1, 125):
        d.line((x, 0, x, h), fill=GRID2, width=1)
    for y in range(0, h + 1, 125):
        d.line((0, y, w, y), fill=GRID2, width=1)


def rrect(d, x, y, w, h, fill=BOX, outline="#c9ad78", width=3, radius=12):
    d.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)


def block(d, x, y, w, h, title, lines, fill=BOX):
    rrect(d, x, y, w, h, fill=fill)
    txt(d, x + 14, y + 12, title, "part", RED)
    yy = y + 48
    for line in lines:
        txt(d, x + 16, yy, line, "small", TEXT)
        yy += 26


def arrow(d, x1, y1, x2, y2, label=None):
    d.line((x1, y1, x2, y2), fill=WIRE, width=5)
    if x2 >= x1:
        pts = [(x2, y2), (x2 - 16, y2 - 9), (x2 - 16, y2 + 9)]
    else:
        pts = [(x2, y2), (x2 + 16, y2 - 9), (x2 + 16, y2 + 9)]
    d.polygon(pts, fill=WIRE)
    if label:
        txt(d, (x1 + x2) / 2 - 55, y1 - 30, label, "net", BLUE)


def connector(d, x, y, title, pins, w=250):
    h = 54 + len(pins) * 38
    d.rectangle((x, y, x + w, y + h), fill=BG, outline=RED, width=3)
    txt(d, x + 16, y + 14, title, "part", RED)
    for i, p in enumerate(pins):
        yy = y + 55 + i * 38
        d.ellipse((x + 17, yy - 7, x + 31, yy + 7), fill=BG, outline=RED, width=3)
        txt(d, x + 44, yy - 12, p, "small", BLUE)
    return h


def opamp_box(d, x, y, name, desc):
    pts = [(x, y), (x, y + 110), (x + 145, y + 55)]
    d.polygon(pts, fill=BG)
    d.line((x, y, x, y + 110, x + 145, y + 55, x, y), fill=RED, width=3)
    txt(d, x + 18, y - 28, name, "part", RED)
    txt(d, x + 38, y + 43, desc, "small", TEXT)
    txt(d, x + 17, y + 28, "+", "part")
    txt(d, x + 17, y + 76, "-", "part")


def schematic():
    w, h = 2300, 1500
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    draw_grid(d, w, h)

    txt(d, 55, 38, "STM32G431CBT6 最小系统板频响仪转接板 - 原理图图片", "title")
    txt(d, 55, 92, "目标：把 PA4 DAC、PA0/PA1 ADC 变成可量程切换、AC/DC、带保护和缓冲的低压共地测量前端。", "text", NOTE)

    connector(d, 60, 145, "H1 接 G431 最小系统板", ["3V3_MCU", "GND", "PA4_DAC", "PA0_ADC_VIN", "PA1_ADC_VOUT"], 310)

    block(d, 430, 145, 560, 245, "电源 + VMID 偏置",
          ["3V3_MCU -> FB1 磁珠/10R -> 3V3_A",
           "3V3_A 对 GND：C101=10uF, C102=100nF",
           "3V3_A -> R101=10k -> VMID_RAW -> R102=10k -> GND",
           "VMID_RAW -> U1A 电压跟随器 -> VMID=1.65V",
           "U1 四运放供电：V+ 接 3V3_A，V- 接 GND"], BOX2)

    block(d, 1060, 145, 640, 245, "PA4 DAC 激励输出",
          ["PA4_DAC -> R301=100R -> U1B 跟随缓冲",
           "U1B_OUT -> R302=100R -> H2 激励 OUT(SIG)",
           "H2_GND 接公共 GND",
           "JP401/JP402：校准时 H2_OUT 经 1k 接 CH1/CH2 输入",
           "正常测量时校准跳帽断开"], BOX)
    connector(d, 1780, 178, "H2 激励 OUT", ["SIG", "GND"], 210)
    arrow(d, 370, 238, 430, 238, "3V3/GND")
    txt(d, 1060, 112, "输入网络：PA4_DAC 来自 H1", "net", BLUE)
    arrow(d, 1700, 240, 1780, 240, "DAC_OUT")

    block(d, 60, 430, 640, 130, "U1 器件建议",
          ["优先：TLV9064 / TLV9054 / MCP6004 / OPA4340",
           "条件：3.3V 单电源、轨到轨输入输出、四运放封装",
           "U1A=VMID，U1B=DAC，U1C=CH1，U1D=CH2"], BOX)

    def channel(y, ch, conn, adc, hname):
        connector(d, 60, y, hname, ["SIG", "GND"], 245)
        block(d, 350, y - 8, 270, 145, f"CH{ch} 输入保护",
              [f"{conn}_SIG -> R{1 if ch == 1 else 21}=1k -> CH{ch}_RAW",
               f"D{1 if ch == 1 else 21}=BAT54S",
               "钳位到 3V3_A 和 GND",
               "保护件尽量靠近输入接口"], BOX2)
        block(d, 700, y - 8, 440, 180, f"CH{ch} 量程选择 SW{ch}",
              ["1x：CH_RAW 直接到 CH_SCALE",
               f"10x：R{2 if ch == 1 else 22}=90k 上臂，R{3 if ch == 1 else 23}=10k 下臂",
               f"100x：330k+330k+330k 上臂，R{7 if ch == 1 else 27}=10k 下臂",
               "1x / 10x / 100x 三选一，不能同时闭合"], BOX)
        block(d, 1220, y - 8, 430, 180, f"CH{ch} AC/DC 耦合",
              ["DC：CH_SCALE 直通缓冲输入",
               f"AC：CH_SCALE 串 C{2 if ch == 1 else 22}=1uF 到缓冲输入",
               f"AC 时 R{8 if ch == 1 else 28}=100k 把输入拉到 VMID",
               "测滤波器频响建议用 AC 档"], BOX)
        opamp_box(d, 1745, y + 20, f"U1{'C' if ch == 1 else 'D'}", f"CH{ch}缓冲")
        block(d, 1945, y + 0, 310, 145, f"到 STM32 {adc}",
              [f"U1{'C' if ch == 1 else 'D'}_OUT -> R{9 if ch == 1 else 29}=220R -> ADC脚",
               f"ADC脚对 GND：C{3 if ch == 1 else 23}=1nF",
               "R/C 尽量靠近 G431 ADC 引脚"], BOX2)

        arrow(d, 305, y + 58, 350, y + 58, "SIG")
        arrow(d, 620, y + 58, 700, y + 58, f"CH{ch}_RAW")
        arrow(d, 1140, y + 58, 1220, y + 58, f"CH{ch}_SCALE")
        arrow(d, 1650, y + 58, 1745, y + 75, f"CH{ch}_BUF_IN")
        arrow(d, 1890, y + 75, 1945, y + 75, f"CH{ch}_BUF")
        txt(d, 1320, y + 134, "VMID=1.65V", "net", BLUE)
        txt(d, 92, y + 138, "GND 与被测低压电路共地", "small", BLUE)

    channel(680, 1, "H3", "PA0_ADC_VIN", "H3 Vin/参考输入")
    channel(1110, 2, "H4", "PA1_ADC_VOUT", "H4 Vout/响应输入")

    rrect(d, 55, 1440, 2190, 42, fill="#fff0df", outline="#d08a42", width=3)
    txt(d, 75, 1450, "安全边界：这块板只用于低压共地电路。市电、高压、开关电源一次侧、浮地高共模信号不要直接接。", "text", "#7a2e00")

    im.save(OUT / "G431CBT6_frontend_schematic.png")


def pad(d, x, y, r=16, label=None):
    d.ellipse((x - r, y - r, x + r, y + r), fill=PAD, outline="#7a5a00", width=3)
    if label:
        txt(d, x + r + 8, y - 12, label, "small", SILK)


def comp(d, x, y, w, h, label, fill="#2f9659"):
    d.rounded_rectangle((x, y, x + w, y + h), radius=9, fill=fill, outline=SILK, width=3)
    txt(d, x + 10, y + h / 2 - 12, label, "small", SILK)


def chip(d, x, y, w, h, label):
    d.rounded_rectangle((x, y, x + w, y + h), radius=8, fill="#171717", outline=SILK, width=3)
    txt(d, x + 18, y + h / 2 - 14, label, "part", SILK)
    for i in range(7):
        py = y + 18 + i * (h - 36) / 6
        d.rectangle((x - 12, py - 7, x, py + 7), fill=PAD)
        d.rectangle((x + w, py - 7, x + w + 12, py + 7), fill=PAD)


def trace(d, pts, color, width=10):
    d.line(pts, fill=color, width=width, joint="curve")


def pcb():
    w, h = 1900, 1250
    im = Image.new("RGB", (w, h), "#f5efe1")
    d = ImageDraw.Draw(im)
    txt(d, 55, 38, "STM32G431CBT6 频响仪转接板 - PCB 图片/布局参考", "title")
    txt(d, 55, 92, "不是 Gerber，不是可投板文件；用于照着在嘉立创 EDA 摆件和走线。红=顶层信号，蓝=底层信号，绿=铺地。", "text", NOTE)

    bx, by, bw, bh = 110, 150, 1680, 1000
    d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=28, fill=PCB, outline="#0c3c22", width=8)
    d.rounded_rectangle((bx + 35, by + 35, bx + bw - 35, by + bh - 35), radius=18, outline="#63bd7d", width=2)
    for x, y in [(170, 210), (1730, 210), (170, 1090), (1730, 1090)]:
        d.ellipse((x - 24, y - 24, x + 24, y + 24), fill="#f5efe1", outline=SILK, width=4)

    comp(d, 155, 405, 230, 300, "H1 接 G431")
    for i, lab in enumerate(["3V3", "GND", "PA4", "PA0", "PA1"]):
        pad(d, 210, 455 + i * 50, 15, lab)

    for y, title in [(270, "H2 DAC OUT"), (610, "H3 Vin IN"), (930, "H4 Vout IN")]:
        d.rounded_rectangle((1540, y - 65, 1715, y + 65), radius=14, fill="#206f42", outline=SILK, width=3)
        txt(d, 1560, y - 52, title, "part", SILK)
        pad(d, 1588, y + 10, 21, "SIG")
        pad(d, 1665, y + 10, 21, "GND")

    chip(d, 850, 515, 225, 175, "U1 四运放")
    comp(d, 690, 260, 290, 95, "VMID: R101/R102 C101/C102")
    comp(d, 650, 395, 150, 70, "FB1/10R")
    comp(d, 1035, 250, 170, 70, "R301")
    comp(d, 1230, 250, 170, 70, "R302")
    comp(d, 1215, 360, 275, 75, "JP401/JP402 校准")

    comp(d, 1180, 535, 125, 65, "D1/R1")
    comp(d, 1045, 620, 270, 95, "CH1 量程")
    comp(d, 1045, 740, 270, 95, "CH1 AC/DC")
    comp(d, 520, 710, 210, 70, "R9/C3 靠近PA0")

    comp(d, 1180, 855, 125, 65, "D21/R21")
    comp(d, 1045, 940, 270, 95, "CH2 量程")
    comp(d, 1045, 1055, 270, 70, "CH2 AC/DC")
    comp(d, 520, 855, 210, 70, "R29/C23 靠近PA1")

    trace(d, [(210, 455), (650, 455), (650, 395), (760, 395), (760, 355), (860, 355), (860, 515)], TOP, 9)
    trace(d, [(210, 505), (350, 505), (350, 1110), (1715, 1110)], GND, 13)
    trace(d, [(210, 555), (510, 555), (510, 300), (1035, 300), (1215, 300), (1588, 280)], TOP, 10)
    trace(d, [(1588, 620), (1305, 620), (1305, 565), (1180, 565), (1180, 665), (1045, 665), (1045, 785), (960, 785), (960, 690)], TOP, 10)
    trace(d, [(850, 610), (720, 745), (520, 745), (210, 605)], BOT, 10)
    trace(d, [(1588, 940), (1305, 940), (1305, 890), (1180, 890), (1180, 985), (1045, 985), (1045, 1090), (980, 1090), (980, 690)], TOP, 10)
    trace(d, [(850, 665), (720, 890), (520, 890), (210, 655)], BOT, 10)
    trace(d, [(1455, 400), (1490, 400), (1490, 620), (1305, 620)], "#f5b532", 7)
    trace(d, [(1455, 420), (1510, 420), (1510, 940), (1305, 940)], "#f5b532", 7)

    txt(d, 420, 430, "PA4 DAC 走线短，远离输入前端", "small", SILK)
    txt(d, 400, 745, "PA0 端 R/C 靠近 G431", "small", SILK)
    txt(d, 400, 890, "PA1 端 R/C 靠近 G431", "small", SILK)
    txt(d, 1370, 715, "输入接口旁边先放保护和量程", "small", SILK)
    txt(d, 650, 1120, "整板铺 GND；H2/H3/H4 地就近打孔接地；CH1/CH2 输入走线不要贴着 DAC 输出走", "part", SILK)

    x, y = 125, 1185
    d.rectangle((x, y, x + 32, y + 16), fill=TOP)
    txt(d, x + 44, y - 6, "顶层信号", "small")
    d.rectangle((x + 180, y, x + 212, y + 16), fill=BOT)
    txt(d, x + 224, y - 6, "底层信号", "small")
    d.rectangle((x + 360, y, x + 392, y + 16), fill=PAD)
    txt(d, x + 404, y - 6, "焊盘/跳帽", "small")

    im.save(OUT / "G431CBT6_frontend_pcb.png")


if __name__ == "__main__":
    schematic()
    pcb()
    print(OUT / "G431CBT6_frontend_schematic.png")
    print(OUT / "G431CBT6_frontend_pcb.png")
