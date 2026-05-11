# Phase: 13
# Purpose: Generate "Honest But Lethal" PPTX for Bilim Senligi
# Inputs: Verified project metrics from artifacts/
# Outputs: artifacts/bilim_senligi_presentation.pptx
# Limitations: Requires python-pptx (pip install python-pptx)
#
# Usage:
#   pip install python-pptx
#   python scripts/generate_presentation.py
#
# All metrics cited are from actual project artifacts:
#   - TNR 98.3%, M<2.0 Recall 58.6% → artifacts/evaluation_metrics.json
#   - PhaseNet cascade 1.52s median MAE → artifacts/phase7_demo_sprint_log.md
#   - RAG: 31 TBDY-2018 provisions indexed → llm/index/tbdy_chunks.json
#   - Station identity leakage 84.8% → documented in phase analyses

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "artifacts" / "bilim_senligi_presentation.pptx"

# --- Color Palette: Dokuz Eylul Academic ---
NAVY = RGBColor(0, 40, 85)
DEEP_NAVY = RGBColor(0, 25, 60)
GOLD = RGBColor(195, 155, 45)
LIGHT_GOLD = RGBColor(255, 235, 170)
WHITE = RGBColor(255, 255, 255)
OFF_WHITE = RGBColor(248, 250, 252)
DARK_TEXT = RGBColor(35, 35, 45)
GRAY_TEXT = RGBColor(90, 95, 105)
LIGHT_GRAY = RGBColor(200, 205, 215)
ACCENT_RED = RGBColor(180, 50, 50)
ACCENT_GREEN = RGBColor(40, 140, 80)
ACCENT_BLUE = RGBColor(50, 120, 200)
SUBTLE_BG = RGBColor(235, 240, 248)

FONT_TITLE = "Calibri Light"
FONT_BODY = "Calibri"
FONT_MONO = "Consolas"


def set_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape_rect(slide, left, top, width, height, fill_color, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape


def add_rounded_box(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, left, top, width, height, font_size=Pt(18),
             bold=False, italic=False, color=DARK_TEXT, align=PP_ALIGN.LEFT,
             font_name=FONT_BODY, vertical_anchor=MSO_ANCHOR.TOP):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = vertical_anchor
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = font_size
    p.font.bold = bold
    p.font.italic = italic
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = align
    return tf


def add_bullets(slide, items, left, top, width, height, font_size=Pt(16),
                color=DARK_TEXT, spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        if isinstance(item, dict):
            p.text = item.get("text", "")
            p.font.size = item.get("size", font_size)
            p.font.bold = item.get("bold", False)
            p.font.italic = item.get("italic", False)
            p.font.color.rgb = item.get("color", color)
            p.font.name = item.get("font", FONT_BODY)
            p.level = item.get("level", 0)
            p.space_before = item.get("space_before", Pt(0))
        elif item == "":
            p.text = ""
            p.font.size = Pt(8)
        else:
            p.text = item
            p.font.size = font_size
            p.font.color.rgb = color
            p.font.name = FONT_BODY

        p.space_after = spacing
    return tf


def slide_header(slide, title, subtitle=None):
    add_shape_rect(slide, Inches(0), Inches(0), Inches(10), Inches(1.3), NAVY)
    add_shape_rect(slide, Inches(0), Inches(1.3), Inches(10), Inches(0.06), GOLD)

    add_text(slide, title,
             Inches(0.6), Inches(0.15), Inches(9), Inches(0.8),
             font_size=Pt(26), bold=True, color=WHITE, font_name=FONT_TITLE)

    if subtitle:
        add_text(slide, subtitle,
                 Inches(0.6), Inches(0.8), Inches(9), Inches(0.4),
                 font_size=Pt(13), color=RGBColor(180, 195, 215), italic=True)


def slide_footer(slide, text):
    add_text(slide, text,
             Inches(0.5), Inches(7.0), Inches(9), Inches(0.3),
             font_size=Pt(9), italic=True, color=GRAY_TEXT)


def metric_box(slide, left, top, value, label, value_color=GOLD):
    box = add_rounded_box(slide, left, top, Inches(2.6), Inches(1.4),
                          RGBColor(15, 30, 55), GOLD)
    tf = box.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.15)
    p = tf.paragraphs[0]
    p.text = value
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = value_color
    p.font.name = FONT_TITLE
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = label
    p2.font.size = Pt(11)
    p2.font.color.rgb = LIGHT_GRAY
    p2.font.name = FONT_BODY
    p2.alignment = PP_ALIGN.CENTER


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ==================================================================
    # SLIDE 1: TITLE
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, DEEP_NAVY)

    # Gold accent line at top
    add_shape_rect(s, Inches(0), Inches(0), Inches(10), Inches(0.08), GOLD)

    # Main title block
    add_text(s, "Derin Öğrenme ve Yerel LLM Entegrasyonu ile",
             Inches(0.8), Inches(1.6), Inches(8.4), Inches(0.6),
             font_size=Pt(20), color=RGBColor(170, 190, 215), font_name=FONT_TITLE,
             align=PP_ALIGN.CENTER)

    add_text(s, "Mikro-Sismik Karar Destek Sistemi",
             Inches(0.8), Inches(2.2), Inches(8.4), Inches(0.8),
             font_size=Pt(34), bold=True, color=WHITE, font_name=FONT_TITLE,
             align=PP_ALIGN.CENTER)

    add_text(s, "Veri Kıtlığı Koşullarında Hibrit Tespit Mimarisi ve Hava Boşluklu Yönetmelik Yorumlama",
             Inches(0.8), Inches(3.0), Inches(8.4), Inches(0.5),
             font_size=Pt(14), italic=True, color=RGBColor(150, 170, 200),
             font_name=FONT_BODY, align=PP_ALIGN.CENTER)

    # Divider
    add_shape_rect(s, Inches(3.5), Inches(3.7), Inches(3), Inches(0.02), GOLD)

    # Authors
    add_text(s, "Bertuğ TAŞ  •  Kadir Emir YÜCEL  •  Melih TAKYACİ  •  Emre ÖZDEMİR",
             Inches(0.8), Inches(4.2), Inches(8.4), Inches(0.5),
             font_size=Pt(15), color=RGBColor(200, 210, 225), align=PP_ALIGN.CENTER)

    add_text(s, "Danışman: Efendi NASIBOĞLU",
             Inches(0.8), Inches(4.7), Inches(8.4), Inches(0.4),
             font_size=Pt(13), color=RGBColor(170, 185, 205), align=PP_ALIGN.CENTER)

    add_text(s, "Dokuz Eylül Üniversitesi • Fen Fakültesi • Bilgisayar Bilimleri Bölümü",
             Inches(0.8), Inches(5.3), Inches(8.4), Inches(0.4),
             font_size=Pt(12), color=RGBColor(130, 150, 180), align=PP_ALIGN.CENTER)

    # Event badge
    add_text(s, "III. Ulusal Temel Bilimler Gençlik Sempozyumu ve Bilim Şenliği • Mayıs 2026",
             Inches(0.8), Inches(6.5), Inches(8.4), Inches(0.4),
             font_size=Pt(10), color=RGBColor(110, 130, 160), align=PP_ALIGN.CENTER)

    # ==================================================================
    # SLIDE 2: THE PROBLEM
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, OFF_WHITE)
    slide_header(s, "Problem: Sismik Veri Boşluğu",
                 "Türkiye'nin M<2.0 mikro-sismik olayları neden görünmez?")

    add_bullets(s, [
        {"text": "Katalog Tamlık Eşiği", "bold": True, "size": Pt(18), "color": NAVY},
        "Türkiye ulusal sismik katalogları: Mc ≈ 2.7 (Tan, 2021)",
        "M < 2.0 olaylar rutin kayıt altyapısına yansımıyor",
        "",
        {"text": "Bizim Gerçekliğimiz: Veri Kıtlığı", "bold": True, "size": Pt(18), "color": ACCENT_RED},
        "Sadece ~80 bağımsız deprem olayı (2 istasyon)",
        "Klasik derin öğrenme (binlerce etiketli örnek gerektirir) burada çöker",
        "",
        {"text": "Keşfettiğimiz Ek Sorun: İstasyon Kimlik Sızıntısı", "bold": True, "size": Pt(18), "color": ACCENT_RED},
        "GPD modeli hangi istasyonun kaydettiğini %84.8 doğrulukla tahmin ediyor",
        "Aynı deprem, iki farklı istasyonda %30.7 farklı sınıflandırma alıyor",
        "Bu, SeisBench makalelerinde belgelenmemiş bir gerçek-dünya sorunu",
    ], Inches(0.6), Inches(1.5), Inches(8.8), Inches(5.5), font_size=Pt(15))

    slide_footer(s, "Bilimsel değer: Fay geometrisi haritalama. NOT: Deprem öngörüsüyle doğrudan ilişki YOKTUR.")

    # ==================================================================
    # SLIDE 3: THE PIVOT
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, OFF_WHITE)
    slide_header(s, "Mühendislik Dönüşümü",
                 "Naif plan çöktü → Daha güçlü bir mimari doğdu")

    # Left: What failed
    box_l = add_rounded_box(s, Inches(0.4), Inches(1.6), Inches(4.4), Inches(5.2),
                            RGBColor(255, 240, 240), ACCENT_RED)
    tf = box_l.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.15)

    items_left = [
        ("BAŞARISIZ OLAN PLAN", True, Pt(14), ACCENT_RED),
        ("", False, Pt(6), DARK_TEXT),
        ("\"3 modeli ince-ayar yap, karşılaştır\"", False, Pt(14), DARK_TEXT),
        ("", False, Pt(6), DARK_TEXT),
        ("Neden çöktü:", True, Pt(13), DARK_TEXT),
        ("• 80 olay ile ince-ayar imkansız", False, Pt(13), GRAY_TEXT),
        ("• EQTransformer binlerce etiket gerektirir", False, Pt(13), GRAY_TEXT),
        ("• İstasyon bias tüm modelleri bozuyor", False, Pt(13), GRAY_TEXT),
        ("• GPD faz belirleyici değil, pencere", False, Pt(13), GRAY_TEXT),
        ("  sınıflandırıcısı (mimari kısıt)", False, Pt(13), GRAY_TEXT),
    ]
    for i, (text, bold, size, color) in enumerate(items_left):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = size
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = FONT_BODY

    # Right: What we built
    box_r = add_rounded_box(s, Inches(5.2), Inches(1.6), Inches(4.4), Inches(5.2),
                            RGBColor(235, 250, 240), ACCENT_GREEN)
    tf2 = box_r.text_frame
    tf2.word_wrap = True
    tf2.margin_left = Inches(0.2)
    tf2.margin_top = Inches(0.15)

    items_right = [
        ("MÜHENDİSLİK ÇÖZÜMÜ", True, Pt(14), ACCENT_GREEN),
        ("", False, Pt(6), DARK_TEXT),
        ("Hibrit Mimari (GPD + SVM)", False, Pt(14), DARK_TEXT),
        ("", False, Pt(6), DARK_TEXT),
        ("Neden işe yaradı:", True, Pt(13), DARK_TEXT),
        ("• GPD'nin öğrenilmiş özelliklerini çıkar", False, Pt(13), GRAY_TEXT),
        ("• İstasyon bazlı Z-skoru normalizasyonu", False, Pt(13), GRAY_TEXT),
        ("  ile bias'ı istatistiksel olarak sıyır", False, Pt(13), GRAY_TEXT),
        ("• Az veriyle çalışan SVM sınıflandırıcı", False, Pt(13), GRAY_TEXT),
        ("• Kaskad PhaseNet ile faz belirleme", False, Pt(13), GRAY_TEXT),
    ]
    for i, (text, bold, size, color) in enumerate(items_right):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = text
        p.font.size = size
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = FONT_BODY

    # Arrow
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                   Inches(4.5), Inches(3.9), Inches(0.8), Inches(0.5)) if False else \
        s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                           Inches(4.5), Inches(3.9), Inches(0.8), Inches(0.5))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = NAVY
    arrow.line.fill.background()

    slide_footer(s, "\"Başarısızlıklar makaledir\" — problemler çözülmüş olarak belgelenmiştir.")

    # ==================================================================
    # SLIDE 4: LAYER 1 — DETECTION
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, OFF_WHITE)
    slide_header(s, "Katman 1: Hibrit Tespit Mimarisi",
                 "GPD özellikleri → Z-skoru normalizasyonu → PCA → SVM (RBF)")

    # Pipeline diagram as text
    add_bullets(s, [
        {"text": "Pipeline Akışı:", "bold": True, "size": Pt(16), "color": NAVY},
        "Ham Dalga Formu (3-bileşen, 100 Hz, 60s pencere)",
        "  → GPD Dondurulmuş Katmanlar → 200-boyutlu latent temsil",
        "  → İstasyon Bazlı Z-Skoru Normalizasyonu (bias eliminasyonu)",
        "  → PCA (75 bileşen, %95.6 varyans açıklanır)",
        "  → SVM (RBF çekirdeği, C=20)",
        "  → Olay / Gürültü kararı",
        "",
        {"text": "Neden bu mimari?", "bold": True, "size": Pt(16), "color": NAVY},
        "• GPD'nin konvolüsyon katmanları zaten sismik paternleri öğrenmiş",
        "• Ama üst katmanlar istasyon kimliğini de kodluyor (%84.8)",
        "• Z-skoru bu bias'ı istatistiksel olarak temizliyor",
        "• SVM sadece 80 örnekle bile etkili (kernel trick)",
    ], Inches(0.6), Inches(1.5), Inches(5.8), Inches(5.5), font_size=Pt(14))

    # Metric boxes on the right
    metric_box(s, Inches(7.0), Inches(1.8), "98.3%", "Gürültü Reddi (TNR)")
    metric_box(s, Inches(7.0), Inches(3.5), "58.6%", "M<2.0 Recall")
    metric_box(s, Inches(7.0), Inches(5.2), "~80", "Eğitim Olayı")

    slide_footer(s, "Metrikler: artifacts/evaluation_metrics.json — bağımsız doğrulama seti üzerinde")

    # ==================================================================
    # SLIDE 5: PHASE PICKING — CASCADE
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, OFF_WHITE)
    slide_header(s, "Katman 1: Kaskad Faz Belirleme",
                 "GPD+SVM tetikler → PhaseNet U-Net hassas varış zamanı belirler")

    add_bullets(s, [
        {"text": "Mimari Kısıt (Keşfedilen Sorun):", "bold": True, "size": Pt(16), "color": ACCENT_RED},
        "GPD bir pencere sınıflandırıcısıdır (4s pencere → \"deprem var/yok\")",
        "P-dalgasının hangi örnekte başladığını söyleyemez",
        "GPD argmax ile faz belirleme: 16.03s MAE (anlamsız)",
        "",
        {"text": "Çözüm: İki Aşamalı Kaskad", "bold": True, "size": Pt(16), "color": ACCENT_GREEN},
        "Aşama 1: GPD+SVM → Olayı tespit et (ucuz, hızlı)",
        "Aşama 2: PhaseNet U-Net → Yalnızca onaylanan olaylarda",
        "         örnek düzeyinde P/S varış zamanı belirle",
        "",
        {"text": "İyileşme:", "bold": True, "size": Pt(16), "color": NAVY},
        "GPD tek başına:     16.03s MAE  →  mimari olarak anlamsız",
        "PhaseNet kaskad:     1.52s ortanca AE  →  10× iyileşme",
    ], Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.5), font_size=Pt(14))

    # Metric boxes
    metric_box(s, Inches(7.0), Inches(2.0), "1.52s", "Ortanca P-Pick AE")
    metric_box(s, Inches(7.0), Inches(3.7), "10×", "İyileşme Faktörü")
    metric_box(s, Inches(7.0), Inches(5.4), "0.1s", "Gelecek Hedef", value_color=LIGHT_GRAY)

    slide_footer(s, "0.1s hedefi: PhaseNet'in Türkiye verisiyle ince-ayarı gerektirir (net gelecek çalışma yolu)")

    # ==================================================================
    # SLIDE 6: LAYER 2 — AIR-GAPPED LLM
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, OFF_WHITE)
    slide_header(s, "Katman 2: Hava Boşluklu Karar Destek Sistemi",
                 "Tamamen çevrimdışı — internet bağlantısı SIFIR")

    add_bullets(s, [
        {"text": "Mimari Tasarım Kararı:", "bold": True, "size": Pt(16), "color": NAVY},
        "Kritik altyapı verisi internete çıkmamalıdır",
        "Deprem sonrası internet altyapısı çökebilir",
        "→ Çözüm: Tamamen yerel, hava boşluklu (air-gapped) LLM",
        "",
        {"text": "Teknoloji Yığını:", "bold": True, "size": Pt(16), "color": NAVY},
        "• Model: DeepSeek-R1 (1.5B parametre) — Ollama üzerinde",
        "• İndeksleme: FAISS vektör veritabanı",
        "• Bilgi tabanı: 31 TBDY-2018 maddesi + 15 proje artifaktı",
        "• Arayüz: Streamlit (3 sekmeli UI)",
        "",
        {"text": "Güvenlik Önlemleri:", "bold": True, "size": Pt(16), "color": NAVY},
        "• RAG: Model yalnızca indekslenmiş belgelerden yanıt üretir",
        "• Prompt kuralları: 6 zorunlu kural (kaynak atfı, kapsam sınırı)",
        "• Sorumluluk reddi: Her raporda zorunlu uyarı",
    ], Inches(0.6), Inches(1.5), Inches(9.0), Inches(5.5), font_size=Pt(14))

    slide_footer(s, "Küçük model avantajı: 1.5B parametre → daha az halüsinasyon, prompt'a daha sıkı uyum")

    # ==================================================================
    # SLIDE 7: THE REGULATORY BRAIN
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, OFF_WHITE)
    slide_header(s, "Yönetmelik Beyni: TBDY-2018 Temelli Raporlama",
                 "Sıfır halüsinasyon mimarisi — her iddia kaynağa bağlı")

    # Example report box
    report_box = add_rounded_box(s, Inches(0.5), Inches(1.6), Inches(5.5), Inches(4.0),
                                 RGBColor(245, 248, 252), ACCENT_BLUE)
    tf = report_box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.15)

    report_lines = [
        ("ÖRNEK ÇIKTI:", True, Pt(11), ACCENT_BLUE),
        ("", False, Pt(4), DARK_TEXT),
        ("\"37.8°N, 27.2°E konumunda M1.8 olay tespit", False, Pt(12), DARK_TEXT),
        ("edilmiştir. TBDY-2018 Madde 2.1'e göre bu", False, Pt(12), DARK_TEXT),
        ("bölge 1. derece deprem bölgesindedir.", False, Pt(12), DARK_TEXT),
        ("", False, Pt(4), DARK_TEXT),
        ("Zemin sınıfı: ZC (Vs30 ≈ 280 m/s).", False, Pt(12), DARK_TEXT),
        ("", False, Pt(4), DARK_TEXT),
        ("Referans: TBDY-2018 Tablo 2.1, Madde 16.5\"", False, Pt(11), ACCENT_BLUE),
        ("", False, Pt(4), DARK_TEXT),
        ("⚠ Bu olasılıksal model tahminlerine", False, Pt(10), ACCENT_RED),
        ("dayanmaktadır. Nihai karar yetkili", False, Pt(10), ACCENT_RED),
        ("mühendise aittir.", False, Pt(10), ACCENT_RED),
    ]
    for i, (text, bold, size, color) in enumerate(report_lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = size
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = FONT_MONO if i > 1 else FONT_BODY

    # Right side: architecture notes
    add_bullets(s, [
        {"text": "RAG Mimarisi:", "bold": True, "size": Pt(15), "color": NAVY},
        "Belge → chunk → embedding",
        "→ FAISS indeksi",
        "→ Sorgu ile en yakın k=5 parça",
        "→ LLM'e bağlam olarak verilir",
        "",
        {"text": "Sonuç:", "bold": True, "size": Pt(15), "color": NAVY},
        "Kaynağı olmayan bilgi üretilemez",
        "Her iddia belgeye bağlı",
        "Kapsam dışı sorular reddedilir",
    ], Inches(6.2), Inches(1.6), Inches(3.5), Inches(4.5), font_size=Pt(13))

    slide_footer(s, "31 TBDY-2018 maddesi + 15 proje artifaktı = 46 doğrulanmış bilgi kaynağı")

    # ==================================================================
    # SLIDE 8: SYSTEM DEMO — REPLAY BUFFER
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, OFF_WHITE)
    slide_header(s, "Canlı Demo: Replay Buffer ve İzleme Arayüzü",
                 "Güvenli simülasyon — gerçek KOERI verileri, kontrollü oynatım")

    add_bullets(s, [
        {"text": "Akış Mimarisi:", "bold": True, "size": Pt(16), "color": NAVY},
        "Replay Buffer: Arşivlenmiş KOERI verileri → kayan pencere (60s)",
        "Ring Buffer: Son N pencereyi bellekte tutar",
        "Her pencere: GPD+SVM tetikleme → PhaseNet (eğer olay) → RAG raporu",
        "",
        {"text": "Demo UI (Streamlit — 3 sekme):", "bold": True, "size": Pt(16), "color": NAVY},
        "Sekme 1: İnteraktif olay haritası (Marmara bölgesi)",
        "Sekme 2: Dalga formu görüntüleyici + P/S pick çizgileri",
        "Sekme 3: LLM soru-cevap (yalnızca proje artifaktları üzerinden)",
        "",
        {"text": "Neden Replay Buffer (canlı veri değil)?", "bold": True, "size": Pt(16), "color": NAVY},
        "• FDSN web servisi ~30-60s gecikme → gerçek zamanlı değil",
        "• Demo güvenilirliği: internet kesintisinden bağımsız",
        "• Bilimsel dürüstlük: \"gerçek zamanlı\" iddiası yapmıyoruz",
    ], Inches(0.6), Inches(1.5), Inches(9.0), Inches(5.5), font_size=Pt(14))

    slide_footer(s, "Gelecek: SeedLink protokolü ile gerçek zamanlı veri akışı (sub-second gecikme)")

    # ==================================================================
    # SLIDE 9: FUTURE ROADMAP
    # ==================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, DEEP_NAVY)

    # Gold accent
    add_shape_rect(s, Inches(0), Inches(0), Inches(10), Inches(0.06), GOLD)

    add_text(s, "Gelecek Yol Haritası",
             Inches(0.6), Inches(0.4), Inches(9), Inches(0.7),
             font_size=Pt(28), bold=True, color=WHITE, font_name=FONT_TITLE)

    add_text(s, "\"Yangın alarmı inşa ettik. Sırada: itfaiye entegrasyonu.\"",
             Inches(0.6), Inches(1.0), Inches(9), Inches(0.5),
             font_size=Pt(14), italic=True, color=GOLD)

    # Three columns for future work
    # Column 1
    col1 = add_rounded_box(s, Inches(0.3), Inches(1.8), Inches(3.0), Inches(4.5),
                           RGBColor(10, 35, 70), ACCENT_BLUE)
    tf = col1.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_top = Inches(0.15)
    lines = [
        ("0.1s MAE HEDEFİ", True, Pt(11), ACCENT_BLUE),
        ("", False, Pt(6), WHITE),
        ("PhaseNet ince-ayar:", True, Pt(12), WHITE),
        ("• 5000+ olay (ISC Bulletin)", False, Pt(11), LIGHT_GRAY),
        ("• Analist onaylı P/S picks", False, Pt(11), LIGHT_GRAY),
        ("• 20+ istasyon", False, Pt(11), LIGHT_GRAY),
        ("• Domain adaptation", False, Pt(11), LIGHT_GRAY),
        ("", False, Pt(4), WHITE),
        ("Beklenen: 0.1-0.3s MAE", False, Pt(11), GOLD),
    ]
    for i, (text, bold, size, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = size
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = FONT_BODY

    # Column 2
    col2 = add_rounded_box(s, Inches(3.5), Inches(1.8), Inches(3.0), Inches(4.5),
                           RGBColor(10, 35, 70), GOLD)
    tf = col2.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_top = Inches(0.15)
    lines = [
        ("EEW SİSTEMİ", True, Pt(11), GOLD),
        ("", False, Pt(6), WHITE),
        ("SeedLink entegrasyonu:", True, Pt(12), WHITE),
        ("• Gerçek zamanlı telemetri", False, Pt(11), LIGHT_GRAY),
        ("• P-dalga → M tahmini (3s)", False, Pt(11), LIGHT_GRAY),
        ("• GMPE → PGA tahmini", False, Pt(11), LIGHT_GRAY),
        ("• S-dalga varış hesabı", False, Pt(11), LIGHT_GRAY),
        ("", False, Pt(4), WHITE),
        ("Hedef: 5-30s uyarı süresi", False, Pt(11), GOLD),
    ]
    for i, (text, bold, size, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = size
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = FONT_BODY

    # Column 3
    col3 = add_rounded_box(s, Inches(6.7), Inches(1.8), Inches(3.0), Inches(4.5),
                           RGBColor(10, 35, 70), ACCENT_GREEN)
    tf = col3.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_top = Inches(0.15)
    lines = [
        ("SHM ENTEGRASYONU", True, Pt(11), ACCENT_GREEN),
        ("", False, Pt(6), WHITE),
        ("IoT sensör verileri:", True, Pt(12), WHITE),
        ("• Bina ivmeölçerleri (MEMS)", False, Pt(11), LIGHT_GRAY),
        ("• Katlar-arası ötelenme", False, Pt(11), LIGHT_GRAY),
        ("• Doğal frekans kayması", False, Pt(11), LIGHT_GRAY),
        ("• Hasar göstergeleri", False, Pt(11), LIGHT_GRAY),
        ("", False, Pt(4), WHITE),
        ("TBDY-2018 Bölüm 15 ile", False, Pt(11), GOLD),
        ("otomatik eşleştirme", False, Pt(11), GOLD),
    ]
    for i, (text, bold, size, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = size
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = FONT_BODY

    # Bottom statement
    add_text(s, "Şu an: Tespit zekası (yangın alarmı) inşa edildi ve Türk istasyonlarında genelleştiği kanıtlandı.",
             Inches(0.5), Inches(6.5), Inches(9), Inches(0.4),
             font_size=Pt(12), color=RGBColor(160, 175, 200), align=PP_ALIGN.CENTER)
    add_text(s, "Sonraki adım: Bu çekirdeği gerçek zamanlı erken uyarı altyapısına entegre etmek.",
             Inches(0.5), Inches(6.9), Inches(9), Inches(0.4),
             font_size=Pt(12), color=GOLD, align=PP_ALIGN.CENTER, bold=True)

    # ==================================================================
    # SAVE
    # ==================================================================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PATH))
    print(f"Presentation saved: {OUTPUT_PATH}")
    print(f"Total slides: {len(prs.slides)}")
    print("Slide deck:")
    titles = [
        "1. Title",
        "2. Problem: Sismik Veri Boşluğu",
        "3. Mühendislik Dönüşümü (Pivot)",
        "4. Katman 1: Hibrit Tespit (GPD+SVM)",
        "5. Katman 1: Kaskad Faz Belirleme (PhaseNet)",
        "6. Katman 2: Hava Boşluklu LLM (DeepSeek+RAG)",
        "7. Yönetmelik Beyni (TBDY-2018)",
        "8. Canlı Demo (Replay Buffer)",
        "9. Gelecek Yol Haritası (EEW + SHM)",
    ]
    for t in titles:
        print(f"  {t}")


if __name__ == "__main__":
    build_presentation()
