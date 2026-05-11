# Phase: 12-13
# Purpose: Generate academic PPTX presentation for Bilim Senligi
# Inputs: Content from bertugBilimSenligiYukle.docx (proposal text only)
# Outputs: artifacts/bilim_senligi_presentation.pptx
# Limitations: Requires python-pptx (pip install python-pptx)
#
# Usage:
#   pip install python-pptx
#   python scripts/generate_presentation.py

"""
Generates an academic presentation for:
"Derin Ogrenme ve Buyuk Dil Modelleri Entegrasyonu ile Mikro-Sismik Oruntu Tespiti:
 Insaat Sektorune Yonelik Erken Uyari ve Destek Sistemi"

All content is derived STRICTLY from the original proposal document.
No fabricated metrics. All targets stated in future/targeted tense.
"""

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "artifacts" / "bilim_senligi_presentation.pptx"

# --- Color Scheme (Academic / Dokuz Eylul University) ---
NAVY = RGBColor(0, 51, 102)
DARK_GRAY = RGBColor(51, 51, 51)
MEDIUM_GRAY = RGBColor(102, 102, 102)
LIGHT_BG = RGBColor(240, 244, 248)
WHITE = RGBColor(255, 255, 255)
ACCENT_BLUE = RGBColor(0, 102, 178)
ACCENT_GOLD = RGBColor(180, 140, 50)

FONT_TITLE = "Calibri"
FONT_BODY = "Calibri"


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title_bar(slide, text, top=Inches(0), height=Inches(1.2)):
    """Add a colored title bar at the top of the slide."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        left=Inches(0), top=top,
        width=Inches(10), height=height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY
    shape.line.fill.background()

    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = FONT_TITLE
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Inches(0.5)


def add_body_text(slide, bullets, top=Inches(1.5), left=Inches(0.5),
                  width=Inches(9), height=Inches(5), font_size=Pt(18)):
    """Add bulleted body text."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        if isinstance(bullet, tuple):
            text, level = bullet
        else:
            text, level = bullet, 0

        p.text = text
        p.level = level
        p.font.size = font_size
        p.font.name = FONT_BODY
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)

        if level == 0:
            p.font.size = font_size
        else:
            p.font.size = Pt(font_size.pt - 2)
            p.font.color.rgb = MEDIUM_GRAY


def add_footnote(slide, text, top=Inches(6.8)):
    """Add a small footnote at the bottom."""
    txBox = slide.shapes.add_textbox(Inches(0.5), top, Inches(9), Inches(0.4))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = MEDIUM_GRAY
    p.font.name = FONT_BODY


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # ================================================================
    # SLIDE 1: Title
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, NAVY)

    # Main title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(8.4), Inches(2.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Derin Öğrenme ve Büyük Dil Modelleri Entegrasyonu ile"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = FONT_TITLE
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "Mikro-Sismik Örüntü Tespiti"
    p2.font.size = Pt(32)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.font.name = FONT_TITLE
    p2.alignment = PP_ALIGN.CENTER

    p3 = tf.add_paragraph()
    p3.text = "İnşaat Sektörüne Yönelik Erken Uyarı ve Destek Sistemi"
    p3.font.size = Pt(20)
    p3.font.color.rgb = RGBColor(180, 200, 220)
    p3.font.name = FONT_TITLE
    p3.alignment = PP_ALIGN.CENTER

    # Authors
    txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(4.5), Inches(8.4), Inches(1.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p4 = tf2.paragraphs[0]
    p4.text = "Bertuğ TAŞ, Kadir Emir YÜCEL, Melih TAKYACİ, Emre ÖZDEMİR"
    p4.font.size = Pt(16)
    p4.font.color.rgb = RGBColor(200, 210, 225)
    p4.font.name = FONT_BODY
    p4.alignment = PP_ALIGN.CENTER

    p5 = tf2.add_paragraph()
    p5.text = "Danışman: Efendi Nasıboğlu"
    p5.font.size = Pt(14)
    p5.font.color.rgb = RGBColor(200, 210, 225)
    p5.font.name = FONT_BODY
    p5.alignment = PP_ALIGN.CENTER

    p6 = tf2.add_paragraph()
    p6.text = "Dokuz Eylül Üniversitesi, Fen Fakültesi, Bilgisayar Bilimleri Bölümü"
    p6.font.size = Pt(13)
    p6.font.color.rgb = RGBColor(160, 175, 195)
    p6.font.name = FONT_BODY
    p6.alignment = PP_ALIGN.CENTER
    p6.space_before = Pt(6)

    # Event
    txBox3 = slide.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(8.4), Inches(0.6))
    tf3 = txBox3.text_frame
    p7 = tf3.paragraphs[0]
    p7.text = "III. Ulusal Temel Bilimler Gençlik Sempozyumu ve Bilim Şenliği • 12–13 Mayıs 2026"
    p7.font.size = Pt(11)
    p7.font.color.rgb = RGBColor(140, 160, 180)
    p7.font.name = FONT_BODY
    p7.alignment = PP_ALIGN.CENTER

    # ================================================================
    # SLIDE 2: Problem Definition
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Problem: Sismik Katalog Boşluğu")

    add_body_text(slide, [
        "Türkiye ulusal sismik kataloglarının tamlık eşiği:",
        ("Mc ≈ 2.7  (Tan, 2021)", 1),
        "",
        "Bu eşiğin altındaki mikro-sismik olaylar mevcut rutin",
        "kayıt altyapısına büyük ölçüde yansımamaktadır.",
        "",
        "Bilimsel değer:",
        ("Fay geometrisinin yüksek çözünürlüklü haritalanması", 1),
        ("Bölgesel stres dağılımının belirlenmesi", 1),
        "",
        "Önemli not: Bu olayların deprem öngörüsüyle",
        "doğrudan bir ilişkisi YOKTUR.",
    ])

    add_footnote(slide, "Kaynak: Tan, O. (2021). Natural Hazards and Earth System Sciences, 21(7), 2059–2073.")

    # ================================================================
    # SLIDE 3: Two-Layer Architecture
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Önerilen Sistem Mimarisi: İki Katmanlı Yapı")

    # Layer 1 box
    shape1 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(1.8), Inches(4.3), Inches(4.5)
    )
    shape1.fill.solid()
    shape1.fill.fore_color.rgb = RGBColor(230, 240, 250)
    shape1.line.color.rgb = ACCENT_BLUE
    shape1.line.width = Pt(1.5)

    tf1 = shape1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.2)
    tf1.margin_top = Inches(0.2)
    p = tf1.paragraphs[0]
    p.text = "KATMAN 1"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.font.name = FONT_TITLE

    p = tf1.add_paragraph()
    p.text = "Derin Öğrenme Tespit Katmanı"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.font.name = FONT_TITLE
    p.space_after = Pt(12)

    for line in [
        "KOERI dalga formu verileri",
        "SeisBench çerçevesi",
        "PhaseNet • EQTransformer • GPD",
        "Türkiye verisine ince-ayar",
        "M<2.0 recall + faz belirleme",
    ]:
        p = tf1.add_paragraph()
        p.text = "• " + line
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.font.name = FONT_BODY
        p.space_after = Pt(4)

    # Layer 2 box
    shape2 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(5.2), Inches(1.8), Inches(4.3), Inches(4.5)
    )
    shape2.fill.solid()
    shape2.fill.fore_color.rgb = RGBColor(250, 245, 230)
    shape2.line.color.rgb = ACCENT_GOLD
    shape2.line.width = Pt(1.5)

    tf2 = shape2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = Inches(0.2)
    tf2.margin_top = Inches(0.2)
    p = tf2.paragraphs[0]
    p.text = "KATMAN 2"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD
    p.font.name = FONT_TITLE

    p = tf2.add_paragraph()
    p.text = "LLM Karar Destek Katmanı"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.font.name = FONT_TITLE
    p.space_after = Pt(12)

    for line in [
        "Olay meta-verileri (konum, M, derinlik)",
        "TBDY-2018 hükümleri",
        "Zemin sınıfı bilgileri",
        "RAG mimarisi",
        "Yönetmelik-duyarlı raporlama",
    ]:
        p = tf2.add_paragraph()
        p.text = "• " + line
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.font.name = FONT_BODY
        p.space_after = Pt(4)

    # Arrow between layers
    arrow = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW,
        Inches(4.55), Inches(3.8), Inches(0.8), Inches(0.5)
    )
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = MEDIUM_GRAY
    arrow.line.fill.background()

    add_footnote(slide, "Özgün katkı: Bu iki katmanı tek pipeline içinde birleştiren ilk entegre mimari önerisi (Türkiye bağlamında)")

    # ================================================================
    # SLIDE 4: Layer 1 — Deep Learning Details
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Katman 1: Derin Öğrenme Modelleri")

    add_body_text(slide, [
        "Kullanılacak modeller (SeisBench üzerinden):",
        "",
        ("GPD (Ross vd., 2018)", 1),
        ("Genelleştirilmiş sismik faz tespiti", 2),
        ("PhaseNet (Zhu & Beroza, 2019)", 1),
        ("P ve S fazlarının varış zamanlarının otomatik belirlenmesi", 2),
        ("EQTransformer (Mousavi vd., 2020)", 1),
        ("Eş zamanlı deprem tespiti ve faz belirleme (dikkat mekanizmalı)", 2),
        "",
        "Yöntem:",
        ("Her üç model Türkiye verisine özgü ince-ayar ile eğitilecektir", 1),
        ("Performansları aynı değerlendirme çerçevesinde karşılaştırılacaktır", 1),
    ])

    add_footnote(slide, "Framework: Woollam vd. (2022). SeisBench—A toolbox for ML in seismology. Seis. Res. Lett., 93(3)")

    # ================================================================
    # SLIDE 5: Dataset
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Veri Seti: 2023 Kahramanmaraş Artçı Sarsıntı Dizisi")

    add_body_text(slide, [
        "Veri kaynağı: KOERI sismik ağı (Cambaz vd., 2021)",
        "",
        "Neden bu veri seti?",
        ("Veri zenginliği: Yoğun artçı sarsıntı aktivitesi", 1),
        ("Güncellik: 2023 yılı verileri", 1),
        ("Geniş magnıtüd aralığı: M<1.0 — M>5.0", 1),
        "",
        "Erişim yöntemi:",
        ("FDSN web servisleri (EIDA/KOERI düğümü)", 1),
        ("Ham dalga formu verileri (miniSEED formatı)", 1),
        ("3 bileşenli (BHZ, BHN, BHE) sismik kayıtlar", 1),
    ])

    add_footnote(slide, "Kaynak: Ding vd. (2023). Earthquake Science, 36. doi:10.1016/j.eqs.2023.06.002")

    # ================================================================
    # SLIDE 6: Targeted Metrics
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Hedeflenen Başarı Ölçütleri")

    add_body_text(slide, [
        "Temel değerlendirme eksenleri:",
        "",
        "1. Olay Tespiti (Detection)",
        ("M < 2.0 bandında recall performansı", 1),
        ("Düşük yanlış pozitif oranı (gürültü pencereleri üzerinde)", 1),
        "",
        "2. Faz Belirleme (Phase Picking)",
        ("Ortalama mutlak hata (MAE) hedefi: 0.1 saniye", 1),
        ("P ve S faz varış zamanları ayrı ayrı değerlendirilecektir", 1),
        "",
        "3. Karşılaştırma",
        ("Üç model aynı metriklerle değerlendirilecektir", 1),
        ("Magnıtüd bandlarına göre stratifiye analiz", 1),
    ], font_size=Pt(17))

    # Emphasis box
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(1), Inches(6.0), Inches(8), Inches(0.8)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(255, 248, 220)
    shape.line.color.rgb = ACCENT_GOLD
    shape.line.width = Pt(1)
    tf = shape.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = "⚠ 0.1s MAE bir hedeftir, tamamlanmış bir sonuç değildir."
    p.font.size = Pt(13)
    p.font.italic = True
    p.font.color.rgb = RGBColor(120, 100, 30)
    p.font.name = FONT_BODY
    p.alignment = PP_ALIGN.CENTER

    # ================================================================
    # SLIDE 7: Layer 2 — LLM & RAG
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Katman 2: LLM ve RAG Entegrasyonu")

    add_body_text(slide, [
        "Giriş verileri:",
        ("Tespit edilen olayların meta-verileri (konum, büyüklük, derinlik)", 1),
        "",
        "Bilgi tabanı (indekslenmiş belgeler):",
        ("TBDY-2018 yönetmelik hükümleri", 1),
        ("Zemin sınıfı bilgileri (Vs30 haritaları)", 1),
        "",
        "Mimari: Retrieval-Augmented Generation (RAG)",
        ("Dil modeli yalnızca indekslenmiş belgelerden yanıt üretir", 1),
        ("Her yanıtın kaynak belgesi izlenebilir", 1),
        ("Halüsinas yon riski minimize edilir", 1),
        "",
        "Amaç:",
        ("'Bu bölgede şu mikro-sismik aktivite tespit edilmiştir;", 1),
        (" TBDY-2018 kapsamında şu değerlendirmeler geçerlidir.'", 1),
    ], font_size=Pt(16))

    add_footnote(slide, "RAG: Wang vd. (2025). J. Building Engineering, 103, 112189.")

    # ================================================================
    # SLIDE 8: Original Contribution
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Özgün Katkı")

    add_body_text(slide, [
        "Projenin temel özgün katkısı:",
        "",
        "Derin öğrenme tabanlı mikro-sismik tespit mekanizması ile",
        "LLM tabanlı yönetmelik-duyarlı karar destek modülünü",
        "tek bir pipeline içinde birleştiren ve bunu doğrudan",
        "Türkiye sismik verisi üzerinde uygulayan",
        "",
        "     ilk entegre mimari önerisi.",
        "",
        "",
        "Bu iki bileşeni — otomatik sismik tespit ve yönetmelik-duyarlı",
        "yapay zeka raporlama — tek bir entegre sistemde birleştiren",
        "bir çalışma Türkiye bağlamında daha önce önerilmemiştir.",
    ], font_size=Pt(17))

    # ================================================================
    # SLIDE 9: Future Work & Conclusion
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Gelecek Çalışma ve Sonuç")

    add_body_text(slide, [
        "Gelecek çalışma:",
        ("Yapısal sağlık izleme (SHM) sensör verilerinin sisteme entegrasyonu", 1),
        ("Bina montajlı ivmeölçerlerden hasar göstergeleri", 1),
        ("TBDY-2018 Bölüm 15 (mevcut bina değerlendirmesi) ile eşleştirme", 1),
        "",
        "Sonuç:",
        ("Türkiye'nin sismik kataloğundaki M<2.7 altı boşluğu", 1),
        ("derin öğrenme ile doldurmayı ve bu bilgiyi", 1),
        ("yönetmelik-duyarlı bir yapay zeka aracıyla", 1),
        ("inşaat sektörüne sunmayı amaçlayan", 1),
        ("entegre bir mimari önermekteyiz.", 1),
    ])

    # Divider
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(1), Inches(6.2), Inches(8), Inches(0.02)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = MEDIUM_GRAY
    shape.line.fill.background()

    add_footnote(slide, "SHM entegrasyonu: bildiri metninde açıkça 'gelecek çalışma' olarak belirtilmiştir.", top=Inches(6.4))

    # ================================================================
    # SLIDE 10: References
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Kaynaklar")

    refs = [
        "[1] Tan, O. (2021). Nat. Hazards Earth Syst. Sci., 21(7), 2059–2073.",
        "[2] Cambaz, M. D. vd. (2021). Seis. Res. Lett., 92(3), 1571–1580.",
        "[3] Woollam, J. vd. (2022). Seis. Res. Lett., 93(3), 1695–1709.",
        "[4] Zhu, W. & Beroza, G. C. (2019). Geophys. J. Int., 216(1), 261–273.",
        "[5] Mousavi, S. M. vd. (2020). Nature Comm., 11, 3952.",
        "[6] Ross, Z. E. vd. (2018). BSSA, 108(5A), 2894–2901.",
        "[7] Ding, M. vd. (2023). Earthquake Science, 36.",
        "[8] Wang, Z. vd. (2025). J. Building Eng., 103, 112189.",
        "[9] Mousavi, S. M. vd. (2025). Geophys. J. Int., 240(2), 1281–1294.",
    ]

    add_body_text(slide, refs, font_size=Pt(13), top=Inches(1.5))

    # ================================================================
    # SLIDE 11: Thank You / Q&A
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, NAVY)

    txBox = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(8), Inches(2))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Teşekkürler"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = FONT_TITLE
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "Sorularınızı memnuniyetle yanıtlarız."
    p2.font.size = Pt(20)
    p2.font.color.rgb = RGBColor(180, 200, 220)
    p2.font.name = FONT_BODY
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(20)

    txBox2 = slide.shapes.add_textbox(Inches(1), Inches(5.5), Inches(8), Inches(1))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p3 = tf2.paragraphs[0]
    p3.text = "bertugtaas@gmail.com"
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(160, 175, 195)
    p3.font.name = FONT_BODY
    p3.alignment = PP_ALIGN.CENTER

    p4 = tf2.add_paragraph()
    p4.text = "Dokuz Eylül Üniversitesi • Bilgisayar Bilimleri Bölümü • 2026"
    p4.font.size = Pt(12)
    p4.font.color.rgb = RGBColor(140, 160, 180)
    p4.font.name = FONT_BODY
    p4.alignment = PP_ALIGN.CENTER

    # ================================================================
    # Save
    # ================================================================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PATH))
    print(f"Presentation saved: {OUTPUT_PATH}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    build_presentation()
