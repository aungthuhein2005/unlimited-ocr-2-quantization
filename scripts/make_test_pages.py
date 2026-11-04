
# scripts/make_test_pages.py
# Renders synthetic test pages with exactly known text, so ground truth needs no transcription.
# Writes data/syn_NN_<kind>.png + data/syn_NN_<kind>.txt. Run locally (needs Windows fonts):
#   python scripts/make_test_pages.py
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path("C:/Windows/Fonts")
OUT_DIR = Path("data")
W, H = 1240, 1754  # A4 at 150 dpi
MARGIN = 110

# Each page: kind, font file, layout, blocks (first block is the title)
PAGES = [
    ("en_report", "times.ttf", "paragraphs", [
        "Annual Maintenance Report 2025",
        "The northern pumping station completed its scheduled maintenance in October. Engineers replaced "
        "two worn impellers, recalibrated the flow sensors and installed a new backup generator. The station "
        "was offline for nine days, which was two days shorter than planned.",
        "Water quality remained within all regulatory limits during the work. Temporary supply was provided "
        "from the eastern reservoir, and no customer reported an interruption of service.",
        "For the coming year the team recommends replacing the control software, which is no longer supported "
        "by its vendor, and adding remote monitoring to the three smaller stations in the valley.",
    ]),
    ("en_twocol", "georgia.ttf", "two_columns", [
        "Notes on Urban Gardens",
        "Community gardens have spread quickly through older neighbourhoods. Many began on empty lots that "
        "had been fenced off for years, and residents simply asked the owners for permission to plant.",
        "The first harvests were small, but the gardens soon became meeting places. Older residents taught "
        "younger ones how to prepare soil, save seeds and protect seedlings from late frost.",
        "City planners now include garden space in new housing projects. They argue that a shared garden "
        "costs little to maintain and gives people a reason to spend time outdoors with their neighbours.",
        "Critics point out that some gardens are used only by a few families. Successful projects usually "
        "have a small committee, clear rules for each plot and an open day for visitors every month.",
    ]),
    ("en_form", "arial.ttf", "lines", [
        "Library Membership Application",
        "Full name: Daniel Okafor",
        "Date of birth: 14 March 1991",
        "Address: 27 Harbour Road, Flat 4B",
        "City: Port Elizabeth",
        "Postal code: 6001",
        "Email: d.okafor@example.org",
        "Phone: +27 41 555 0183",
        "Membership type: Adult (annual)",
        "Preferred branch: Central Library",
        "Signature: D. Okafor",
        "Date: 2 November 2026",
    ]),
    ("en_numbers", "calibri.ttf", "lines", [
        "Invoice INV-2026-0417",
        "Issued: 03/11/2026  Due: 03/12/2026",
        "Customer ID: C-88213",
        "Item 1: Laser printer toner x 4 = 236.00",
        "Item 2: A4 paper, 80 g/m2, 10 boxes = 412.50",
        "Item 3: USB-C cable 2 m x 12 = 95.88",
        "Item 4: Service call (2.5 hours) = 187.50",
        "Subtotal: 931.88",
        "VAT 15%: 139.78",
        "Total due: 1,071.66",
        "Account: 4021 7730 0098 1256",
        "Reference: PO 55-1093-A",
    ]),
    ("en_degraded", "times.ttf", "paragraphs", [
        "Field Survey Summary",
        "The survey team visited fourteen sites along the river between May and July. At each site they "
        "recorded water depth, bank erosion and the number of nesting birds.",
        "Erosion was strongest below the old bridge, where the bank has retreated by almost two metres "
        "since the previous survey. The team recommends planting willows to stabilise the soil.",
        "Bird numbers were similar to last year, although two species were seen for the first time.",
    ]),
    ("en_lowres", "verdana.ttf", "paragraphs", [
        "Customer Notice",
        "Our office will be closed on Friday for staff training. Orders placed on Thursday evening will be "
        "processed on Monday morning.",
        "If you need urgent help, please call the support line, which remains open from eight in the "
        "morning until six in the evening. We apologise for any inconvenience.",
    ]),
    ("en_code", "cour.ttf", "lines", [
        "Server Setup Checklist",
        "1. Update packages: apt update && apt upgrade -y",
        "2. Create user: adduser deploy --disabled-password",
        "3. Copy key to /home/deploy/.ssh/authorized_keys",
        "4. Set port 2222 in /etc/ssh/sshd_config",
        "5. Enable firewall: ufw allow 2222/tcp",
        "6. Restart service: systemctl restart sshd",
        "7. Check status: systemctl status sshd --no-pager",
        "8. Log result in ops/log_2026-11-04.md",
    ]),
    ("de", "arial.ttf", "paragraphs", [
        "Bericht über die Stadtbibliothek",
        "Die Stadtbibliothek hat im vergangenen Jahr mehr als zwölftausend neue Leser gewonnen. Besonders "
        "beliebt waren die Lesestunden für Kinder und die Kurse für ältere Menschen, die den Umgang mit "
        "Tablets lernen möchten.",
        "Für das nächste Jahr plant die Bibliothek längere Öffnungszeiten am Samstag. Außerdem sollen "
        "größere Arbeitsplätze für Schüler eingerichtet werden, die dort ruhig lernen möchten.",
    ]),
    ("fr", "times.ttf", "paragraphs", [
        "Compte rendu de la réunion",
        "La réunion s'est tenue mardi à la mairie. Les habitants ont exprimé leur inquiétude au sujet de la "
        "circulation près de l'école, où les voitures roulent souvent trop vite.",
        "Le maire a promis d'installer deux ralentisseurs avant l'été et d'étudier la création d'une piste "
        "cyclable. Une nouvelle réunion aura lieu en février pour présenter le projet détaillé.",
    ]),
    ("es", "georgia.ttf", "paragraphs", [
        "Informe sobre el huerto escolar",
        "¿Qué aprendieron los alumnos este año? Además de cultivar tomates y pimientos, los niños midieron "
        "el crecimiento de las plantas y anotaron los resultados en un cuaderno compartido.",
        "¡La cosecha fue excelente! El próximo año la escuela quiere añadir un pequeño invernadero y "
        "organizar una feria donde las familias puedan probar lo que los alumnos han cultivado.",
    ]),
    ("ru", "arial.ttf", "paragraphs", [
        "Отчёт о работе музея",
        "В прошлом году музей посетили более сорока тысяч человек. Самой популярной стала выставка старых "
        "фотографий города, которую подготовили вместе с местными жителями.",
        "В следующем году музей планирует открыть новый зал для детей и проводить экскурсии по выходным. "
        "Кроме того, часть коллекции будет доступна в интернете.",
    ]),
    ("zh", "msyh.ttc", "paragraphs_cjk", [
        "社区图书馆年度报告",
        "去年社区图书馆共接待读者三万多人次。儿童阅读活动和老年人电脑课程最受欢迎，许多家庭每周都会来参加。",
        "明年图书馆计划延长周末的开放时间，并增加安静的学习区域。我们也希望更多志愿者加入，帮助整理图书和组织活动。",
    ]),
    ("ja", "msgothic.ttc", "paragraphs_cjk", [
        "市民農園のお知らせ",
        "今年の市民農園では、四十の区画すべてが利用されました。参加者はトマトやナスを育て、収穫祭には多くの家族が集まりました。",
        "来年は水道設備を新しくし、子ども向けの体験教室を毎月開く予定です。申し込みは十二月から市役所で受け付けます。",
    ]),
    ("ko", "malgun.ttf", "paragraphs", [
        "마을 도서관 운영 보고",
        "작년에 마을 도서관을 찾은 사람은 이만 명이 넘었습니다. 어린이 독서 모임과 어르신 컴퓨터 교실이 특히 인기가 많았습니다.",
        "내년에는 주말 운영 시간을 늘리고 조용한 공부 공간을 더 만들 계획입니다. 자원봉사자 모집도 곧 시작합니다.",
    ]),
]


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)


def wrap(text, fnt, max_width, cjk):
    """Split text into lines that fit max_width; CJK wraps per character, others per word."""
    tokens = list(text) if cjk else text.split(" ")
    lines, current = [], ""
    for token in tokens:
        candidate = current + token if cjk or not current else current + " " + token
        if fnt.getlength(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = token
    lines.append(current)
    return lines


def draw_column(draw, blocks, x, y, width, body, cjk, wrap_blocks=True):
    line_height = int(body.size * 1.5)
    for block in blocks:
        lines = wrap(block, body, width, cjk) if wrap_blocks else [block]
        for line in lines:
            draw.text((x, y), line, font=body, fill="black")
            y += line_height
        y += int(body.size * 0.8) if wrap_blocks else 0
    return y


def render(kind, font_name, layout, blocks):
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    title_font, body = font(font_name, 46), font(font_name, 30)
    cjk = layout == "paragraphs_cjk"

    draw.text((MARGIN, MARGIN), blocks[0], font=title_font, fill="black")
    y = MARGIN + 100
    text_width = W - 2 * MARGIN

    if layout == "two_columns":
        gap = 60
        col_width = (text_width - gap) // 2
        half = len(blocks[1:]) // 2
        y_end = max(
            draw_column(draw, blocks[1:1 + half], MARGIN, y, col_width, body, cjk),
            draw_column(draw, blocks[1 + half:], MARGIN + col_width + gap, y, col_width, body, cjk),
        )
    elif layout == "lines":
        y_end = draw_column(draw, blocks[1:], MARGIN, y, text_width, body, cjk, wrap_blocks=False)
    else:
        y_end = draw_column(draw, blocks[1:], MARGIN, y, text_width, body, cjk)

    assert y_end < H - MARGIN, f"{kind}: text overflows the page"
    return degrade(kind, page)


def degrade(kind, page):
    if kind == "en_degraded":
        # Slight blur, 1.5 degree tilt and grey noise, like a poor photocopy
        page = page.filter(ImageFilter.GaussianBlur(1.2))
        page = page.rotate(1.5, expand=True, fillcolor="white")
        noise = Image.effect_noise(page.size, 40).convert("RGB")
        page = Image.blend(page, noise, 0.12)
    elif kind == "en_lowres":
        page = page.resize((W // 3, H // 3), Image.LANCZOS)
    return page


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for number, (kind, font_name, layout, blocks) in enumerate(PAGES, start=1):
        stem = OUT_DIR / f"syn_{number:02d}_{kind}"
        render(kind, font_name, layout, blocks).save(stem.with_suffix(".png"), optimize=True)
        # Ground truth: one block per line, two-column pages in reading order (left column first)
        stem.with_suffix(".txt").write_text("\n".join(blocks) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {stem}.png/.txt")


if __name__ == "__main__":
    main()
