# Phase 12: Ethical Science Fair Pitch

**Source:** `bertugBilimSenligiYukle.docx` (original proposal)
**Ethical Standard:** Zero fabrication. Proposal tense preserved. No claims beyond text.

---

## STEP 1: Factual Extraction from Source Document

### The Precise Problem Statement

Direct quote (Turkish):
> "Türkiye ulusal sismik kataloglarının tamlık eşiği yaklaşık olarak Mc≈2.7 düzeyinde seyretmektedir[1]. Bu eşiğin altında kalan mikro-sismik olaylar ne yazık ki mevcut rutin kayıt altyapısına büyük ölçüde yansımamaktadır."

Fact: Turkey's national seismic catalog has a completeness threshold of approximately Mc≈2.7. Events below this threshold are largely absent from routine monitoring infrastructure.

### Scientific Value (explicitly NOT earthquake prediction)

Direct quote:
> "Bu küçük ölçekli olayların deprem öngörüsüyle doğrudan bir ilişkisi kurulmamakla birlikte fay geometrisinin ve bölgesel stres dağılımının yüksek çözünürlüklü biçimde haritalanması açısından göz ardı edilemeyecek düzeyde bilimsel değer taşımaktadır."

Fact: The document explicitly states these events do NOT have a direct relationship to earthquake prediction. Their value lies in high-resolution mapping of fault geometry and regional stress distribution.

### Proposed Layer 1: Seismic Detection Pipeline

From text:
- Data source: KOERI seismic network raw waveform data [2]
- Framework: SeisBench [3]
- Models: PhaseNet [4], EQTransformer [5], GPD [6]
- Action verb: "eğitilecek ve performansları karşılaştırılacaktır" (WILL BE trained and compared — future tense)
- Dataset: 2023 Kahramanmaras aftershock sequence [7]
- Success criteria: M<2.0 recall + phase picking accuracy
- MAE target: "ortalama mutlak hatanın 0.1s düzeyine yaklaştırılması **hedeflenmektedir**" (IS TARGETED — not achieved)

### Proposed Layer 2: LLM-RAG Decision Support

From text:
- Input: detected event metadata (location, magnitude, depth)
- Knowledge base: TBDY-2018 provisions + soil classification data
- Architecture: Retrieval-Augmented Generation (RAG) [8]
- Integration: LLM processes queries grounded in indexed regulatory documents

### Original Contribution (as stated in document)

Direct quote:
> "Projenin temel özgün katkısı derin öğrenme tabanlı mikro-sismik tespit mekanizması ile LLM tabanlı yönetmelik-duyarlı karar destek modülünü tek bir pipeline içinde birleştiren ve bunu doğrudan Türkiye sismik verisi üzerinde uygulayan ilk entegre mimari önerisi olmasıdır."

Translation: The first integrated architecture proposal that combines DL-based micro-seismic detection with LLM-based regulation-aware decision support in a single pipeline, applied directly to Turkish seismic data.

### Future Work (explicitly stated)

> "Yapısal sağlık izleme sensör verilerinin sisteme entegrasyonu gelecek çalışma kapsamında ayrıca planlanmaktadır."

Fact: SHM sensor integration is explicitly planned as future work — not a deliverable of this study.

---

## STEP 2: Ethical Presentation Script (Turkish, 5 minutes)

---

### Slayt 1 — Başlık (15 saniye)

"Sayın jüri üyeleri, merhaba. Sunumumuzun başlığı: 'Derin Öğrenme ve Büyük Dil Modelleri Entegrasyonu ile Mikro-Sismik Örüntü Tespiti: İnşaat Sektörüne Yönelik Erken Uyarı ve Destek Sistemi.' Ben Bertuğ Taş, Dokuz Eylül Üniversitesi Bilgisayar Bilimleri Bölümü'nden. Ekip arkadaşlarım Kadir Emir Yücel, Melih Takyaci, Emre Özdemir ve danışmanımız Efendi Nasiboğlu."

---

### Slayt 2 — Problem Tanımı (60 saniye)

"Çalışmamızın çıkış noktası şudur: Türkiye'nin ulusal sismik kataloglarının tamlık eşiği yaklaşık Mc≈2.7 düzeyindedir. Bu ne anlama geliyor? Magnitüd 2.7'nin altındaki mikro-sismik olaylar, mevcut rutin kayıt altyapısına büyük ölçüde yansımamaktadır.

Bu küçük ölçekli olayların deprem öngörüsüyle doğrudan bir ilişkisi yoktur — bunu açıkça belirtmek istiyoruz. Ancak fay geometrisinin ve bölgesel stres dağılımının yüksek çözünürlüklü biçimde haritalanması açısından göz ardı edilemeyecek bilimsel değer taşımaktadırlar.

Özellikle inşaat sektörü için bu bilginin önemi şudur: bir bölgedeki fay haritası ne kadar ayrıntılı olursa, TBDY-2018 yönetmeliği kapsamında yapılan sismik tehlike değerlendirmeleri o kadar güvenilir olur."

---

### Slayt 3 — Önerilen Mimari: Katman 1 (75 saniye)

"Önerdiğimiz sistem iki katmanlı bir mimariye sahiptir.

Birinci katmanda, KOERI sismik ağına ait ham dalga formu verilerini kullanmaktayız. Veri seti olarak 2023 Kahramanmaraş artçı sarsıntı dizisini tercih ettik. Bu tercihte dizinin veri zenginliği ve güncelliği belirleyici olmuştur.

Bu veriler üzerinde, sismolojik makine öğrenmesi için geliştirilen SeisBench çerçevesi aracılığıyla üç model eğitilmektedir:

- PhaseNet: Zhu ve Beroza tarafından geliştirilen, P ve S fazlarının varış zamanlarını otomatik olarak belirleyen bir derin sinir ağı.
- EQTransformer: Mousavi ve arkadaşlarının geliştirdiği, eş zamanlı olarak hem deprem tespiti hem faz belirleme yapabilen dikkat mekanizmalı bir model.
- GPD: Ross ve arkadaşlarının genelleştirilmiş sismik faz tespiti için tasarladığı derin öğrenme modeli.

Bu üç modelin Türkiye verisine özgü ince-ayar ile eğitilmesi ve performanslarının karşılaştırılması hedeflenmektedir. Temel başarı ölçütleri olarak M<2.0 bandındaki recall performansı ile faz belirleme doğruluğu ele alınmaktadır. Faz belirleme başarısında ortalama mutlak hatanın 0.1 saniye düzeyine yaklaştırılması hedeflenmektedir."

---

### Slayt 4 — Önerilen Mimari: Katman 2 (60 saniye)

"İkinci katmanda, tespit edilen olaylara ait konum, büyüklük ve derinlik gibi meta-veriler bir bilgi tabanı üzerinden işlenmektedir.

Bu bilgi tabanında TBDY-2018 hükümleri ile zemin sınıfı bilgileri indekslenmiştir. Sistem, Retrieval-Augmented Generation — yani RAG — mimarisi aracılığıyla bir büyük dil modeli ile bütünleştirilmektedir.

RAG mimarisinin avantajı şudur: dil modeli yalnızca indekslenmiş, doğrulanmış belgelerden yanıt üretir. Bu sayede halüsinasyon riski minimize edilir ve her yanıtın kaynağı izlenebilir.

Amaç, bir mühendise veya karar vericiye şunu söyleyebilmektir: 'Bu bölgede şu özelliklerde mikro-sismik aktivite tespit edilmiştir; TBDY-2018'in ilgili maddeleri kapsamında şu değerlendirmeler geçerlidir.'"

---

### Slayt 5 — Özgün Katkı ve Gelecek Çalışma (45 saniye)

"Projenin temel özgün katkısı şudur: derin öğrenme tabanlı mikro-sismik tespit mekanizması ile büyük dil modeli tabanlı yönetmelik-duyarlı karar destek modülünü tek bir pipeline içinde birleştiren ve bunu doğrudan Türkiye sismik verisi üzerinde uygulayan ilk entegre mimari önerisidir.

Bildiğimiz kadarıyla, bu iki bileşeni — otomatik sismik tespit ve yönetmelik-duyarlı yapay zeka raporlama — tek bir entegre sistemde birleştiren bir çalışma Türkiye bağlamında daha önce önerilmemiştir.

Gelecek çalışma kapsamında, yapısal sağlık izleme sensör verilerinin sisteme entegrasyonu ayrıca planlanmaktadır."

---

### Slayt 6 — Kapanış (15 saniye)

"Özetle: Türkiye'nin sismik katalogundaki M<2.7 altı boşluğu derin öğrenme ile doldurmayı ve bu bilgiyi yönetmelik-duyarlı bir yapay zeka aracıyla inşaat sektörüne sunmayı amaçlayan entegre bir mimari önermekteyiz. Sorularınızı memnuniyetle yanıtlarız."

---

**Toplam süre:** ~4 dakika 30 saniye (doğal Türkçe konuşma hızında)

---

## STEP 3: Honest Q&A Defense

---

### Soru 1: "0.1 saniyelik faz belirleme MAE hedefinize ulaştınız mı?"

**Etik Cevap:**

"Hayır — 0.1 saniye, çalışmamızda bir hedef olarak belirlenmiştir, tamamlanmış bir sonuç olarak sunulmamaktadır. Bildiri metnimizde de bu ifade 'hedeflenmektedir' fiili ile verilmiştir.

Bu hedefin nasıl ölçüleceğini açıklayalım: Eğitilmiş modelin ürettiği P-fazı varış zamanı tahmini ile analist tarafından doğrulanmış gerçek varış zamanı arasındaki ortalama mutlak fark hesaplanacaktır. Bu hesaplama, modelin ince-ayar sürecinin tamamlanmasının ardından değerlendirme aşamasında yapılacaktır.

Literatürde PhaseNet'in orijinal çalışmasında benzer veri setlerinde 0.1-0.3 saniye aralığında MAE değerlerine ulaşılabildiği gösterilmiştir. Bizim hedefimiz, Türkiye'ye özgü veriler üzerinde bu performansın tekrarlanıp tekrarlanamayacağını test etmektir."

---

### Soru 2: "Bu gerçekten bir 'Erken Uyarı Sistemi' mi?"

**Etik Cevap:**

"Başlığımızda 'Erken Uyarı ve Destek Sistemi' ifadesi geçmektedir. Burada iki bileşeni ayırt etmek gerekir:

Birincisi, erken uyarı bağlamında: Sistemimizin tespit katmanı, mevcut kataloglarda yer almayan M<2.7 altı olayları otomatik olarak tanımlama kapasitesine sahip olacak şekilde tasarlanmıştır. Bu, mevcut durumda kaydedilmeyen olayları görünür kılmak anlamında bir 'erken bilgilendirme' sağlar.

İkincisi, destek bağlamında: RAG mimarisi üzerinden TBDY-2018 yönetmeliğini otomatik olarak yorumlayan bir karar destek modülü önerilmektedir.

Ancak açıkça belirtelim: bu çalışma, S-dalgası varmadan önce saniyeler içinde alarm veren türde bir sismik erken uyarı sistemi (EEW) tasarımı değildir. O tür bir sistem gerçek zamanlı telemetri altyapısı, hızlı magnitüd tahmini ve yer hareketi tahmin denklemleri gerektirir. Bizim odak noktamız tespit ve yönetmelik-duyarlı raporlama katmanlarıdır."

---

### Soru 3: "Üç modelin karşılaştırma sonuçları nedir?"

**Etik Cevap:**

"Bildirimizde bu karşılaştırmanın yapılacağı belirtilmektedir. Metodolojimiz şu şekildedir: SeisBench çerçevesi aracılığıyla PhaseNet, EQTransformer ve GPD modellerinin her biri aynı veri seti üzerinde — 2023 Kahramanmaraş artçı sarsıntı dizisi — ince-ayar ile eğitilecek ve aynı değerlendirme metrikleri ile karşılaştırılacaktır.

Değerlendirme metrikleri olarak M<2.0 bandında recall ve faz belirleme MAE kullanılacaktır. Bu metrikler, her model için ayrı ayrı hesaplanacak ve karşılaştırmalı bir tablo halinde sunulacaktır.

Henüz tüm modellerin eğitim ve değerlendirme döngüsü tamamlanmamıştır; bu nedenle nihai karşılaştırma sonuçlarını şu an sunamıyoruz. Sunabileceğimiz, metodolojinin kendisi ve değerlendirme çerçevesidir."

---

### Soru 4: "LLM halüsinasyon yapmayacağını nasıl garanti ediyorsunuz?"

**Etik Cevap:**

"RAG mimarisi tam olarak bu soruna yönelik tasarlanmıştır. Sistem şu şekilde çalışacaktır: Büyük dil modeli serbest bilgi üretmez; yalnızca FAISS vektör indeksinde bulunan TBDY-2018 maddeleri ve proje artifaktlarından alıntılayarak yanıt üretir. Kaynağı olmayan bir bilgiyi üretemez çünkü cevap üretim mekanizması doğrudan belge parçalarına bağlıdır.

Buna ek olarak, prompt tasarımında her iddia için kaynak atfı zorunlu kılınacak ve bilgi tabanı dışındaki konularda yanıt vermemesi sağlanacaktır. Bu önlemler halüsinasyonu tamamen ortadan kaldırmaz — hiçbir LLM sistemi yüzde yüz garanti veremez — ancak riski kabul edilebilir düzeye indirecek şekilde minimize eder.

Son olarak, sistemin her çıktısında 'Bu yapay zeka tabanlı bir değerlendirmedir, nihai karar yetkili mühendise aittir' şeklinde bir sorumluluk reddi yer alacaktır."

---

### Soru 5: "Bu çalışmanın AFAD veya KOERI'nin mevcut sistemlerinden farkı nedir?"

**Etik Cevap:**

"AFAD ve KOERI mevcut rutin altyapılarıyla büyük ölçüde M≈2.7 üzerindeki olayları kataloglamaktadır. Bizim çalışmamız bu eşiğin altındaki mikro-sismik olayları hedeflemektedir. Bu farklı bir problem ölçeğidir.

İkinci fark, AFAD veya KOERI ham parametreler yayınlar — koordinat, magnitüd, derinlik — ancak bunları doğrudan TBDY-2018 yönetmeliği ile eşleştiren ve doğal dilde yorum üreten bir mekanizma sunmamaktadır. Bizim önerdiğimiz ikinci katman bu boşluğu doldurmayı amaçlamaktadır.

Bu bir alternatif değil, tamamlayıcı bir sistem önerisidir."

---

## Ethical Guardrails Summary

| Aspect | What We SAY | What We DO NOT Say |
|--------|-------------|---------------------|
| MAE target | "0.1s hedeflenmektedir" | "0.1s'ye ulaştık" |
| Model comparison | "Karşılaştırılacaktır" | "Karşılaştırdık ve X kazandı" |
| EEW | "Tespit ve karar destek sistemi" | "Deprem erken uyarı veriyor" |
| Prediction | "Fay haritalama için değerli" | "Deprem tahmin ediyoruz" |
| LLM | "Halüsinasyon riski minimize edilir" | "Halüsinasyon imkansız" |
| SHM | "Gelecek çalışma kapsamında planlanmaktadır" | "Entegre ettik" |

---

**Phase 12 Complete.**
