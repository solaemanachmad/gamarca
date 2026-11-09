# GAMA-RCA: Ide Pengembangan Riset

*Laporan deep research, 21 Oktober 2026. Sumber: kode repo gama-rca (README, riwayat git), draf paper EAI, tiga PDF di project, dan 22 sumber web. Dari 25 klaim yang diverifikasi silang (3 voter), 24 lolos dan 1 gugur.*

## 1. Diagnosis singkat

Hasil dev-30 terbaik: **EL 54,4 · FI 20,0 · RP 34,8 · Final 38,2**. Entitas tepat di 13/30 kasus, tetapi tipe fault yang benar hanya 3 dari 13 itu. Ada tiga hal yang membuat hasil terasa "tidak memuaskan":

1. **Pemetaan bukti → label adalah masalah utama.** Semua varian sejauh ini (P1, TYPE, TYPELEAN, CMP, EVSC, KP) tetap meminta LLM 7B memilih bebas di antara 28 label. Literatur menunjukkan cara ini lemah. Tanpa contoh kasus dan struktur, klasifikasi kategori nyaris nol (RCACopilot: Micro-F1 0,026 zero-shot vs 0,766 dengan exemplar).
2. **n = 30 terlalu kecil untuk melihat efek varian prompt.** Satu kasus setara 3,3 poin FI. Selisih antar-arm Anda (16,7 → 20,0) hanya sekitar 1 kasus. Wajar bila semua selisih berpasangan mencakup nol: ini masalah power statistik, belum tentu idenya buruk.
3. **Backbone 7B.** Di studi lain, mengganti backbone memberi efek sekitar 5× lebih besar daripada mengganti framework (Lu et al. 2026: +10,8 pp vs sekitar 2 pp).

## 2. Ide prioritas

| # | Ide | Dampak yang diharapkan | Biaya | Bukti |
|---|---|---|---|---|
| 1 | **Lapisan fingerprint fault-type + klasifikasi hierarkis + eliminasi hipotesis** | Tinggi pada FI | Sedang | RCACopilot, distribusi label RCA100, GoS |
| 2 | **Verifier bergaya DiagGuard** (Grounder + Verifier, aturan tetap) | Sedang–tinggi pada EL/FI | Rendah–sedang | +10,5 pp median Acc@1, 12/12 konfigurasi RCA100 |
| 3 | **Ablation backbone lokal** (Qwen3-8B/14B, Qwen3-32B terkuantisasi, thinking on/off) | Tinggi, belum pasti | Rendah | Qwen3 report, Lu et al., Riddell et al. |
| 4 | **Angka eksplisit di rantai kausal** demi checkpoint RP | Sedang pada RP | Rendah | 99,5% checkpoint bersyarat numerik |
| 5 | **Pembobotan modalitas untuk typing:** log/event/alert diperkuat, trace dibatasi | Sedang pada FI | Rendah | Riddell et al. 2026 |
| 6 | **Interaktivitas terbatas** (tool query dengan skema tetap, ≤ 3–5 langkah) | Tidak pasti pada 7B | Tinggi | Hanya terbukti di model besar |
| 7 | **Protokol evaluasi:** aturan dibekukan, 73 kasus held-out, uji berpasangan | Wajib untuk publikasi | Rendah | — |

### Ide 1: Fingerprint per tipe fault → klasifikasi bertingkat (prioritas utama)

Ubah "LLM memilih 1 dari 28" menjadi tiga langkah.

- **Tahap A, level dan grup (deterministik + LLM).** Tentukan dulu levelnya (node / pod / service). RCA100: apm.svc 83, k8s.node 17, k8s.pod 3. Lalu tentukan satu dari 6 grup semantik: Application logic 38, JVM 16, Cloud resource 14, Middleware/DB 13, K8s lifecycle 12, Resource/perf 10.
- **Tahap B, fingerprint.** Untuk tiap dari 28 tipe, tulis aturan bukti yang wajib ada dan yang menyangkal. Aturan ini diturunkan dari *definisi* tipe (taxonomy.py, katalog PoS yang berlisensi MIT, dokumentasi chaos tool), **bukan** dari label RCA100. Contoh:
  - `nodeCpuHigh` → node CPU naik dan beberapa pod di node yang sama melambat, tanpa event restart.
  - `memoryPressure` vs OOM → event OOMKilled/restart membedakan keduanya.
  - `rateLimiting` vs `httpError5xx` → kode 429 / template log "rate limit" vs 5xx.
- **Tahap C, eliminasi dan output terkendala.** Fingerprint menggugurkan tipe yang tidak didukung bukti. LLM hanya memilih di antara ≤ 3 kandidat yang tersisa, dengan output yang dibatasi ke ID label yang valid (guided decoding / logit mask). Ini versi deterministik dari ide state machine GoS/PoS yang bisa dijalankan model 7B.

Perbedaan dengan arm CMP yang sudah ada: kandidatnya dipilih oleh **aturan**, bukan oleh LLM, dan tipe tanpa bukti wajib tidak pernah sampai ke LLM.

Catatan:

- Tipe ekor yang hanya muncul 1 kasus (codeDefect, nodeDown, cacheBreakdown, diskIOHigh) tidak akan banyak terbantu.
- Bank exemplar boleh dibuat dari injeksi fault sintetis Anda sendiri di OpenTelemetry Demo, atau dari dataset lain (AIOps2025). Exemplar tidak boleh diambil dari RCA100.

### Ide 2: Verifier tanpa training

Lu et al. (2026; preprint dari kelompok pembuat RCA100) menguji DiagGuard pada subset propagasi RCA100 (57 kasus, 6 framework). Hasilnya: Acc@1 naik di **12/12 konfigurasi**, median +10,5 pp (rentang 3,5–35,1).

Padanan di GAMA-RCA adalah satu panggilan "soundness check" setelah Coordinator, yang memeriksa:

- (a) apakah entitas penyebab ada di rantai;
- (b) apakah tipe fault cocok dengan fingerprint entitas itu;
- (c) apakah setiap langkah rantai mengutip `signal=value` yang benar-benar ada di tabel.

Jika gagal, kembali ke kandidat berikutnya.

Batasan bukti: metrik yang diukur adalah lokalisasi (bukan FI), single-run, dan backbone API besar.

### Ide 3: Upgrade backbone (eksperimen termurah dengan potensi terbesar)

- Menurut Qwen3 Technical Report: Qwen3-8B ≈ Qwen2.5-14B, dan Qwen3-14B ≈ Qwen2.5-32B (base model, benchmark umum).
- Di studi 48.000 skenario dengan model terbuka 3B–104B, Qwen3-32B konsisten terbaik.
- Repo Anda sudah mendukung `enable_thinking` (commit 9aac85f), jadi tinggal menjalankan arm yang sama.
- Grid yang disarankan:
  - Qwen2.5-7B (sekarang)
  - Qwen3-8B (thinking off / on)
  - Qwen3-14B
  - Qwen3-32B AWQ/4-bit, yang muat di GPU Kaggle 2×T4 dengan konteks dibatasi
- Hasil ini sendiri sudah bisa menjadi kontribusi: "studi skala backbone lokal yang tunduk lisensi".

### Ide 4: Angka eksplisit untuk RP

Dari 661 checkpoint, 99,5% berbentuk ⟨comparator, value, unit⟩, dengan mayoritas request count (204), avg latency (178), dan error count (97). Paksa setiap langkah rantai menyebut nilai dan satuan, misalnya `error_count=412 (baseline 3)`, `avg_latency=2.8s`. Flag `CHAIN_ABSOLUTE_SIGNALS` sudah mengarah ke sini. Verifier (Ide 2) bisa memastikan angka tersebut benar-benar ada.

Peringatan: RP lokal Anda masih aproksimasi. Algoritma pencocokan resmi (toleransi, parsing) tidak dipublikasikan.

### Ide 5: Pembobotan modalitas

Riddell et al. (2026) menemukan:

- menghapus log menurunkan akurasi tipe hingga −0,13;
- menghapus metrik menurunkan lokalisasi;
- menghapus trace justru bisa *menaikkan* akurasi.

Untuk langkah typing, tampilkan template Drain3, event K8s, dan alert lebih dulu, dan ringkas tabel TWIST/trace menjadi 1–2 baris.

### Ide 6: Agen interaktif? Jangan dulu

- Di model besar, agen adaptif jauh lebih baik (RCABench: sekitar 78–80% vs 42–53% Acc@1 untuk pipeline tetap). Namun agen-agen itu memakai sekitar 1,5 juta token per kasus.
- Di model ≤ 20B, ReAct, Plan-and-Execute, dan code-act sering gagal: Qwen2.5-14B ReAct di OpenRCA hanya 0,75%, dan Qwen3-4B gagal di 87% run Plan-and-Execute.
- Komponen terpenting OpsAgent (14B) justru pemroses data deterministik (Correct 16,5 → 2,3 bila dihapus).

**Kesimpulan:** pertahankan pipeline. Interaktivitas baru layak dicoba setelah backbone ≥ 14–32B, dan bentuknya sebaiknya tool tetap dengan sedikit langkah.

## 3. Metodologi dan framing paper

**Protokol evaluasi.** Bekukan aturan, prompt, dan arm di dev-30, lalu jalankan **sekali** pada 73 kasus held-out. Laporkan:

- bootstrap CI (sudah ada di `summarize_arms.py`) dan McNemar / permutation test berpasangan per kasus untuk FI biner;
- skor per grup semantik dan per level (node vs service);
- matriks kebingungan antar-tipe (sangat informatif untuk bab analisis).

**Baseline yang dijanjikan draf.** Direct-LLM, Standard RAG, GraphRAG-only, dan Multi-Agent-only belum ada di tabel hasil. Keempatnya wajib dijalankan, atau klaimnya dihapus.

**Kontribusi yang jujur dan kuat:**

1. pipeline RCA lokal 7–32B yang tunduk lisensi CC BY-NC-SA (tanpa API, tanpa training);
2. analisis kesenjangan EL vs FI, ditambah lapisan fingerprint/hierarki/verifier sebagai solusinya, dengan ablation pada 103 kasus;
3. studi skala backbone lokal.

Hindari klaim SOTA terhadap PoS: PoS memakai model API besar, dan metriknya *joint accuracy*, bukan Final.

**Ketidaksesuaian draf vs kode yang harus diperbaiki:**

- Draf menyebut Ollama, 4 backbone, temperature 0.2, 103 kasus, dan "6 rounds". Kode memakai transformers, Qwen2.5-7B, temperature 0, 30 kasus dev, dan pipeline satu lintasan.
- Nama framework campur antara GAMA-RCA dan GraphRAG-RCA.
- Klaim "tidak ada RCA berbasis retrieval vektor" rawan dibantah, karena RCACopilot memakai retrieval insiden historis. Perhalus menjadi "tidak ada yang menggabungkan graph gate + vektor sebelum agen per modalitas di RCA100".
- Kata kunci masih placeholder, dan Tabel 1 rusak (kolom duplikat).

## 4. Rencana eksperimen (urutan)

1. **Minggu 1:** jalankan grid backbone (Ide 3) pada arm utama yang sudah ada, di dev-30. Tidak perlu kode baru.
2. **Minggu 1–2:** tulis fingerprint 28 tipe dari definisi, lalu implementasikan tahap A–C (Ide 1) dan output terkendala.
3. **Minggu 2:** verifier soundness (Ide 2) dan angka eksplisit (Ide 4).
4. **Minggu 3:** 4 baseline draf, lalu bekukan semuanya dan jalankan pada 73 kasus held-out dengan uji statistik.
5. **Minggu 4:** tulis ulang Methods/Results berdasarkan hasil nyata.

## 5. Keterbatasan riset ini

- Tidak ditemukan solusi pemenang Tianchi 2025 / CCF AIOps 2025 yang terverifikasi, dan angka PoS tidak diverifikasi ulang.
- Sebagian besar bukti berasal dari preprint 2026 (Lu et al., GoS, OpsAgent), single-run, dengan model API besar. Transfer ke 7B lokal belum diuji.
- Metrik di literatur (Acc@1, Micro-F1, LLM-judge) berbeda dari EL/FI/RP. Belum ada sumber yang melaporkan kenaikan FI 28 kelas secara langsung.
- Satu klaim gugur (0–3): OpsAgent 14B *tidak* terbukti mengungguli RCA-Agent Claude. Jangan mengutipnya.

## Sumber

- Cai et al. 2026, *A Multi-Dataset Benchmark for Evaluating LLM Agents in Microservice Failure Diagnosis* (PDF di project); data: https://www.aiops.cn/gitlab/aiops-live-benchmark/agenticopseval
- Lu et al. 2026, *Beyond Fault Localization: Trajectory-Level Study of LLM Agents for Microservice RCA* (PDF di project): RCABench, DiagGuard
- RCACopilot (EuroSys 2024): https://arxiv.org/abs/2305.15778
- Riddell et al. 2026: https://arxiv.org/abs/2601.22208
- OpsAgent: https://arxiv.org/abs/2510.24145
- GoS: https://arxiv.org/abs/2603.21250
- OpenRCA (ICLR 2025): https://openreview.net/forum?id=M4qNIzQYpd · https://github.com/microsoft/OpenRCA
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- DiagFusion: https://arxiv.org/abs/2302.10512 · Eadro: https://arxiv.org/abs/2302.05092
- PoS: https://github.com/luoyu100/PoS
