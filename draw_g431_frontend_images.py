from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(r"D:\ai")


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


F = {
    "title": font(42, True),
    "h1": font(30, True),
    "h2": font(24, True),
    "txt": font(22),
    "small": font(18),
    "tiny": font(15),
    "net": font(19, True),
}

BG = "#fffdf7"
GRID = "#eee7dc"
GRID2 = "#ded3c0"
WIRE = "#147a36"
RED = "#b00000"
BLUE = "#174fbc"
TEXT = "#111111"
NOTE = "#3a3a3a"
BOX = "#fff8e8"
PCB = "#1e7e45"
COPPER_TOP = "#c9362b"
COPPER_BOT = "#2457c6"
SILK = "#f7f2e8"
PAD = "#ffd35a"
MASK = "#1c6c3c"


def grid(draw, w, h):
    for x in range(0, w + 1, 25):
        draw.line((x, 0, x, h), fill=GRID, width=1)
    for y in range(0, h + 1, 25):
        draw.line((0, y, w, y), fill=GRID, width=1)
    for x in range(0, w + 1, 125):
        draw.line((x, 0, x, h), fill=GRID2, width=1)
    for y in range(0, h + 1, 125):
        draw.line((0, y, w, y), fill=GRID2, width=1)


def text(draw, xy, s, f="txt", fill=TEXT, anchor=None):
    draw.text(xy, s, font=F[f], fill=fill, anchor=anchor)


def box(draw, xy, title, lines=None, w=300, h=160, fill=BOX, outline="#c7ad7c"):
    x, y = xy
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=fill, outline=outline, width=3)
    text(draw, (x + 16, y + 14), title, "h2", RED)
    if lines:
        yy = y + 52
        for line in lines:
            text(draw, (x + 18, yy), line, "small", TEXT)
            yy += 28


def connector(draw, x, y, title, pins, w=230):
    h = 62 + 40 * len(pins)
    draw.rectangle((x, y, x + w, y + h), fill=BG, outline=RED, width=3)
    text(draw, (x + 16, y + 14), title, "h2", RED)
    for i, pin in enumerate(pins):
        yy = y + 60 + i * 40
        draw.ellipse((x + 16, yy - 7, x + 30, yy + 7), fill=BG, outline=RED, width=3)
        text(draw, (x + 42, yy - 12), pin, "small", BLUE)
    return h


def opamp(draw, x, y, name, label):
    pts = [(x, y), (x, y + 130), (x + 170, y + 65)]
    draw.polygon(pts, fill=BG, outline=RED)
    draw.line((x, y, x, y + 130, x + 170, y + 65, x, y), fill=RED, width=3)
    text(draw, (x + 28, y - 28), name, "h2", RED)
    text(draw, (x + 42, y + 53), label, "small", TEXT)
    text(draw, (x + 20, y + 38), "+", "h2", TEXT)
    text(draw, (x + 20, y + 88), "-", "h2", TEXT)
    return (x, y + 50), (x, y + 95), (x + 170, y + 65)


def resistor(draw, x1, y1, x2, y2, label):
    draw.line((x1, y1, x1 + 20, y1), fill=WIRE, width=4)
    start = x1 + 20
    pts = [(start, y1)]
    step = (x2 - x1 - 40) / 6
    for i in range(6):
        pts.append((start + step * (i + 0.5), y1 - (13 if i % 2 == 0 else -13)))
        pts.append((start + step * (i + 1), y1))
    draw.line(pts, fill=RED, width=3, joint="curve")
    draw.line((x2 - 20, y2, x2, y2), fill=WIRE, width=4)
    text(draw, ((x1 + x2) // 2 - 42, y1 - 42), label, "small", RED)


def cap_to_ground(draw, x, y, label):
    draw.line((x, y, x, y + 32), fill=WIRE, width=3)
    draw.line((x - 22, y + 32, x + 22, y + 32), fill=RED, width=3)
    draw.line((x - 22, y + 50, x + 22, y + 50), fill=RED, width=3)
    draw.line((x, y + 50, x, y + 82), fill=WIRE, width=3)
    draw.line((x - 26, y + 82, x + 26, y + 82), fill=WIRE, width=3)
    text(draw, (x + 34, y + 28), label, "small", RED)
    text(draw, (x + 34, y + 76), "GND", "small", BLUE)


def node(draw, x, y):
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=RED)


def schematic_png():
    w, h = 2400, 1600
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    grid(d, w, h)

    text(d, (55, 40), "STM32G431CBT6 最小系统板频响仪转接板 - 原理图", "title")
    text(d, (55, 92), "照着这个在嘉立创 EDA 画即可：不是投板文件，是电路连接参考图。所有接口低压共地。", "txt", NOTE)

    # MCU connector
    connector(d, 60, 150, "H1 接 G431 最小系统板", ["3V3", "GND", "PA4_DAC", "PA0_ADC_VIN", "PA1_ADC_VOUT"], 310)
    d.line((370, 210, 510, 210), fill=WIRE, width=4)
    d.line((370, 250, 510, 250), fill=WIRE, width=4)
    d.line((370, 290, 990, 290), fill=WIRE, width=4)
    d.line((370, 330, 2140, 330, 2140, 1180), fill=WIRE, width=4)
    d.line((370, 370, 2220, 370, 2220, 1420), fill=WIRE, width=4)
    text(d, (400, 196), "3V3_MCU", "net", BLUE)
    text(d, (400, 236), "GND", "net", BLUE)
    text(d, (400, 276), "PA4_DAC", "net", BLUE)
    text(d, (400, 316), "PA0_ADC_VIN", "net", BLUE)
    text(d, (400, 356), "PA1_ADC_VOUT", "net", BLUE)

    # Power / VMID
    box(d, (500, 145), "电源与 1.65V 偏置", [
        "3V3_MCU -> FB1/10R -> 3V3_A",
        "C101 10uF + C102 100nF 去耦",
        "R101=10k, R102=10k 分压",
        "U1A 缓冲输出 VMID=1.65V",
    ], w=420, h=210)
    d.line((510, 210, 900, 210), fill=WIRE, width=4)
    resistor(d, 560, 210, 665, 210, "FB1/10R")
    cap_to_ground(d, 720, 210, "C101 10uF")
    cap_to_ground(d, 835, 210, "C102 100nF")
    d.line((900, 210, 900, 300), fill=WIRE, width=4)
    text(d, (820, 195), "3V3_A", "net", BLUE)
    d.line((625, 250, 900, 250), fill=WIRE, width=4)
    text(d, (825, 238), "GND", "net", BLUE)

    box(d, (500, 380), "U1 四运放推荐", [
        "TLV9064 / TLV9054 / MCP6004",
        "3.3V 单电源，轨到轨输入输出",
        "U1A: VMID, U1B: DAC",
        "U1C/U1D: 两路 ADC 缓冲",
    ], w=420, h=170)

    # DAC
    box(d, (990, 145), "PA4 DAC 激励输出", [
        "PA4_DAC -> R301 100R -> U1B 缓冲",
        "U1B 输出 -> R302 100R -> H2 OUT",
        "H2 接被测电路输入端",
        "软件输出带 1.65V 偏置的正弦",
    ], w=590, h=220)
    d.line((990, 290, 1090, 290), fill=WIRE, width=4)
    resistor(d, 1090, 290, 1210, 290, "R301 100R")
    _, _, out = opamp(d, 1240, 230, "U1B", "DAC缓冲")
    d.line((1210, 290, 1240, 280), fill=WIRE, width=4)
    d.line((1410, 295, 1480, 295), fill=WIRE, width=4)
    resistor(d, 1480, 295, 1600, 295, "R302 100R")
    connector(d, 1635, 235, "H2 激励 OUT", ["SIG", "GND"], 215)
    d.line((1600, 295, 1635, 295), fill=WIRE, width=4)
    d.line((1635, 335, 1545, 335, 1545, 410), fill=WIRE, width=4)
    text(d, (1520, 430), "GND", "net", BLUE)
    text(d, (1200, 392), "JP401/JP402：校准时 H2_OUT 经 1k 接 CH1/CH2 输入，正常断开", "small", NOTE)

    # helper to draw channel
    def channel(y, ch, input_name, adc_name, xoff=0):
        connector(d, 60, y, f"H{3 if ch == 1 else 4} {input_name} 输入", ["SIG", "GND"], 245)
        d.line((305, y + 60, 395, y + 60), fill=WIRE, width=4)
        resistor(d, 395, y + 60, 510, y + 60, f"R{1 if ch == 1 else 21} 1k")
        node(d, 520, y + 60)
        text(d, (475, y + 28), f"CH{ch}_RAW", "net", BLUE)
        box(d, (455, y + 105), f"D{1 if ch == 1 else 21} 输入保护", [
            "BAT54S 双肖特基",
            "钳位到 3V3_A / GND",
            "前面 1k 限流",
        ], w=260, h=120)
        d.line((520, y + 60, 520, y + 115), fill=WIRE, width=4)

        box(d, (780, y - 10), f"SW{ch} 量程选择", [
            "1x：CH_RAW -> CH_SCALE",
            "10x：90k / 10k 分压",
            "100x：990k / 10k 分压",
            "三选一，不能同时闭合",
        ], w=430, h=200)
        d.line((520, y + 60, 780, y + 60), fill=WIRE, width=4)
        d.line((1210, y + 80, 1330, y + 80), fill=WIRE, width=4)
        text(d, (1230, y + 52), f"CH{ch}_SCALE", "net", BLUE)
        text(d, (805, y + 142), f"10x: R{2 if ch == 1 else 22}=90k, R{3 if ch == 1 else 23}=10k", "small", RED)
        text(d, (805, y + 170), f"100x: 330k+330k+330k / R{7 if ch == 1 else 27}=10k", "small", RED)

        box(d, (1330, y - 10), f"SW{ch} AC/DC 耦合", [
            "DC：CH_SCALE 直通 U1 输入",
            f"AC：串 C{2 if ch == 1 else 22}=1uF",
            f"AC 时 R{8 if ch == 1 else 28}=100k 到 VMID",
            "测滤波器频响优先用 AC",
        ], w=425, h=200)
        d.line((1330, y + 80, 1755, y + 80), fill=WIRE, width=4)
        d.line((1535, y + 80, 1535, y + 165), fill=WIRE, width=3)
        text(d, (1555, y + 140), "VMID", "net", BLUE)

        nonlocal_adcy = 1180 if ch == 1 else 1420
        plus, minus, output = opamp(d, 1810, y + 10, f"U1{'C' if ch == 1 else 'D'}", f"CH{ch}缓冲")
        d.line((1755, y + 80, 1810, y + 60), fill=WIRE, width=4)
        d.line((1980, y + 75, 2050, y + 75), fill=WIRE, width=4)
        resistor(d, 2050, y + 75, 2170, y + 75, f"R{9 if ch == 1 else 29} 220R")
        d.line((2170, y + 75, 2140 if ch == 1 else 2220, nonlocal_adcy), fill=WIRE, width=4)
        cap_to_ground(d, 2195, y + 75, f"C{3 if ch == 1 else 23} 1nF")
        text(d, (2020, y + 22), f"到 {adc_name}", "net", BLUE)

    channel(650, 1, "Vin/参考", "PA0_ADC_VIN")
    channel(1070, 2, "Vout/响应", "PA1_ADC_VOUT")

    # safety note
    d.rounded_rectangle((55, 1510, 2345, 1560), radius=12, fill="#fff0df", outline="#d08a42", width=3)
    text(d, (75, 1522), "安全边界：只测低压共地电路。市电、高压、开关电源一次侧、浮地高共模信号不要直接接。", "txt", "#7a2e00")

    im.save(OUT / "G431CBT6_frontend_adapter_schematic_picture.png")


def pad(draw, x, y, r=16, label=None):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=PAD, outline="#7b5a00", width=3)
    if label:
        text(draw, (x + r + 8, y - 12), label, "small", SILK)


def chip(draw, x, y, w, h, name):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill="#171717", outline=SILK, width=3)
    text(draw, (x + 18, y + h // 2 - 13), name, "h2", SILK)
    for i in range(7):
        py = y + 18 + i * (h - 36) / 6
        draw.rectangle((x - 10, py - 7, x, py + 7), fill=PAD)
        draw.rectangle((x + w, py - 7, x + w + 10, py + 7), fill=PAD)


def trace(draw, pts, color, width=10):
    draw.line(pts, fill=color, width=width, joint="curve")


def component_box(draw, xy, wh, name, fill="#2f9659"):
    x, y = xy
    w, h = wh
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=fill, outline=SILK, width=3)
    text(draw, (x + 10, y + h / 2 - 12), name, "small", SILK)


def pcb_png():
    w, h = 1900, 1250
    im = Image.new("RGB", (w, h), "#f5efe1")
    d = ImageDraw.Draw(im)
    text(d, (55, 40), "STM32G431CBT6 频响仪转接板 - PCB 布局/走线参考图", "title")
    text(d, (55, 92), "不是可投板文件：用于照着在嘉立创 EDA 摆件、命名网络、走线。红色=顶层信号，蓝色=底层信号，绿色=铺地。", "txt", NOTE)

    bx, by, bw, bh = 110, 150, 1680, 1000
    d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=28, fill=PCB, outline="#0c3c22", width=8)
    d.rounded_rectangle((bx + 35, by + 35, bx + bw - 35, by + bh - 35), radius=18, outline="#55b878", width=2)
    text(d, (bx + 35, by + 25), "整板铺 GND 铜，外部接口地与 G431 GND 共地", "small", SILK)

    # mounting holes
    for x, y in [(170, 210), (1730, 210), (170, 1090), (1730, 1090)]:
        d.ellipse((x - 24, y - 24, x + 24, y + 24), fill="#f5efe1", outline=SILK, width=4)

    # H1 MCU connector
    component_box(d, (155, 410), (220, 285), "H1 接G431", "#2b8350")
    for i, lab in enumerate(["3V3", "GND", "PA4", "PA0", "PA1"]):
        pad(d, 205, 455 + i * 48, 14, lab)
    text(d, (145, 720), "H1 放左边，方便短线接最小系统板", "small", SILK)

    # external connectors
    for y, title in [(270, "H2 DAC OUT"), (610, "H3 Vin IN"), (930, "H4 Vout IN")]:
        d.rounded_rectangle((1540, y - 65, 1715, y + 65), radius=14, fill="#206f42", outline=SILK, width=3)
        text(d, (1560, y - 52), title, "h2", SILK)
        pad(d, 1585, y + 10, 20, "SIG")
        pad(d, 1660, y + 10, 20, "GND")

    # IC and passives
    chip(d, 850, 515, 220, 170, "U1 四运放")
    component_box(d, (700, 270), (250, 95), "VMID: R101/R102 + C101/C102")
    component_box(d, (650, 405), (145, 70), "FB1/10R")

    # DAC area
    component_box(d, (1040, 250), (170, 70), "R301")
    component_box(d, (1230, 250), (170, 70), "R302")
    component_box(d, (1240, 365), (250, 70), "JP401/402 校准")
    text(d, (1110, 200), "DAC 输出区靠近 H2", "small", SILK)

    # CH1 input front-end
    component_box(d, (1185, 535), (120, 65), "D1/R1")
    component_box(d, (1060, 610), (245, 95), "CH1 量程 1x/10x/100x")
    component_box(d, (1060, 730), (245, 95), "CH1 AC/DC + C2/R8")
    text(d, (1120, 500), "CH1 前端靠近 H3，保护先于长走线", "small", SILK)

    # CH2 input front-end
    component_box(d, (1185, 855), (120, 65), "D21/R21")
    component_box(d, (1060, 930), (245, 95), "CH2 量程 1x/10x/100x")
    component_box(d, (1060, 1045), (245, 70), "CH2 AC/DC + C22/R28")
    text(d, (1110, 835), "CH2 前端靠近 H4", "small", SILK)

    # ADC output RC near H1
    component_box(d, (520, 705), (210, 70), "R9/C3 靠近PA0")
    component_box(d, (520, 845), (210, 70), "R29/C23 靠近PA1")

    # traces
    # power
    trace(d, [(205, 455), (650, 455), (650, 405), (760, 405), (760, 365), (850, 365), (850, 515)], COPPER_TOP, 9)
    trace(d, [(205, 503), (350, 503), (350, 1110), (1715, 1110)], MASK, 12)
    # dac
    trace(d, [(205, 551), (510, 551), (510, 300), (1040, 300), (1210, 300), (1350, 300), (1585, 280)], COPPER_TOP, 10)
    # ch1
    trace(d, [(1585, 620), (1305, 620), (1305, 565), (1185, 565), (1185, 655), (1070, 655), (1070, 775), (960, 775), (960, 685)], COPPER_TOP, 10)
    trace(d, [(850, 610), (730, 740), (520, 740), (205, 599)], COPPER_BOT, 10)
    # ch2
    trace(d, [(1585, 940), (1305, 940), (1305, 890), (1185, 890), (1185, 980), (1070, 980), (1070, 1070), (980, 1070), (980, 685)], COPPER_TOP, 10)
    trace(d, [(850, 660), (720, 880), (520, 880), (205, 647)], COPPER_BOT, 10)
    # cal
    trace(d, [(1455, 400), (1490, 400), (1490, 620), (1305, 620)], "#f5b532", 7)
    trace(d, [(1455, 420), (1510, 420), (1510, 940), (1305, 940)], "#f5b532", 7)

    # labels/arrows
    text(d, (420, 430), "PA4 DAC 走线尽量短", "small", SILK)
    text(d, (390, 735), "PA0 ADC 输入端 RC 靠近 MCU", "small", SILK)
    text(d, (390, 875), "PA1 ADC 输入端 RC 靠近 MCU", "small", SILK)
    text(d, (1430, 725), "输入接口旁边先放保护和量程", "small", SILK)
    text(d, (650, 1120), "模拟地整面铺铜，输入走线和 DAC 输出尽量分开", "h2", SILK)

    legend_x, legend_y = 125, 1185
    d.rectangle((legend_x, legend_y, legend_x + 32, legend_y + 16), fill=COPPER_TOP)
    text(d, (legend_x + 44, legend_y - 6), "顶层信号", "small", TEXT)
    d.rectangle((legend_x + 180, legend_y, legend_x + 212, legend_y + 16), fill=COPPER_BOT)
    text(d, (legend_x + 224, legend_y - 6), "底层信号", "small", TEXT)
    d.rectangle((legend_x + 360, legend_y, legend_x + 392, legend_y + 16), fill=PAD)
    text(d, (legend_x + 404, legend_y - 6), "焊盘/跳帽", "small", TEXT)

    im.save(OUT / "G431CBT6_frontend_adapter_pcb_picture.png")


if __name__ == "__main__":
    schematic_png()
    pcb_png()
    print(OUT / "G431CBT6_frontend_adapter_schematic_picture.png")
    print(OUT / "G431CBT6_frontend_adapter_pcb_picture.png")
