[TASK-301] MARL (PPO/DQN) Ortam, Aksiyon ve Ödül Fonksiyonu Geliştirme

Çok ajanlı pekiştirmeli öğrenme (MARL) mimarisi kurulacak. Durum uzayı (kuyruk uzunluğu, bekleme süresi, faz bilgisi), aksiyon uzayı (yeşil ışık süresini uzatma/kısaltma, faz geçişi) ve ödül fonksiyonu (R=−(α⋅Kuyruk Gecikmesi+β⋅Dur-Kalk Sayısı)R=−(α⋅Kuyruk Gecikmesi+β⋅Dur-Kalk Sayısı)) matematiksel olarak modellenip PyTorch ortamında eğitilecek.

Kabul
- Çok ajanlı pekiştirmeli öğrenme (MARL) mimarisi kurulacak. Durum uzayı (kuyruk uzunluğu, bekleme süresi, faz bilgisi), aksiyon uzayı (yeşil ışık süresini uzatma/kısaltma, faz geçişi) ve ödül fonksiyonu (R=−(α⋅Kuyruk Gecikmesi+β⋅Dur-Kalk Sayısı)R=−(α⋅Kuyruk Gecikmesi+β⋅Dur-Kalk Sayısı)) matematiksel olarak modellenip PyTorch ortamında eğitilecek.
- Eğitim sürecindeki kayıp (loss) ve kümülatif ödül grafikleri TensorBoard üzerinden belgelenmeli.


[TASK-302] Proaktif Acil Durum Yeşil Koridor Algoritması

Acil müdahale aracı (ambulans/itfaiye) sisteme sinyal gönderdiğinde, sadece bulunduğu kavşağı değil, güzergah üzerindeki sonraki 2 kavşağı proaktif olarak boşaltan (ön tahliye / fermuar düzeni sağlayan) yeşil dalga algoritması kodlanacak.

Kabul
- 3 km'lik pilot koridorda acil araç geçiş süresi baz çizgisine göre ≥%50 oranında düşürülmeli.
- Acil durum aracı geçtikten sonra kavşakların MARL kontrolüne güvenli ve sarsıntısız dönüşü sağlanmalı.


[TASK-303] Trafik İstatistikleri ve Yoğunluk Kestirimi için Analiz / LLM Modülü

Geçmiş İZUM ve simülasyon verilerinden yararlanarak arterlerdeki darboğazları önceden tahmin eden zaman serisi / istatistiksel analiz modülü geliştirilecek. Bu verileri yorumlayarak operatöre doğal dilde trafik brifingi ve tıkanıklık uyarısı veren hafif bir LLM tabanlı karar destek ajanı entegre edilecek.

Kabul
- Önümüzdeki 15 dakikalık trafik yoğunluğunu kestiren istatistiksel tahmin mekanizması doğrulanmalı.
- LLM servisinin sistem loglarından anomali ve kaza raporu özetlerini başarıyla üretmesi sağlanmalı.

