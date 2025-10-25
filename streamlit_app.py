import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Üstü BES'te Kalsın — Hızlı Sunum", layout="wide")

# ---------- yardımcılar ----------
def next_multiple(x: int, base: int) -> int:
    k = x // base
    return (k + 1) * base

def contribution(amount: float, base: int) -> float:
    amt_int = int(round(amount))
    return float(max(0, next_multiple(amt_int, base) - amt_int))

def tl(x):
    return f"{x:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------- sabitler ----------
CATEGORIES = [
    ("Market", 24), ("Kafe", 12), ("Restoran", 14), ("Ulaşım", 10),
    ("Eczane", 6), ("Giyim", 8), ("Elektronik", 5),
    ("Online Alışveriş", 12), ("Fatura/Servis", 9),
]
INCOME_PROFILES = {
    "Düşük Gelir":   {"lognorm_mean": 3.0,  "lognorm_sd": 0.45, "spend_mult": 1.00},
    "Orta Gelir":    {"lognorm_mean": 3.6,  "lognorm_sd": 0.50, "spend_mult": 1.15},
    "Üst-Orta Gelir":{"lognorm_mean": 4.0,  "lognorm_sd": 0.55, "spend_mult": 1.25},
    "Yüksek Gelir":  {"lognorm_mean": 4.35, "lognorm_sd": 0.60, "spend_mult": 1.40},
}
CATEGORY_SCALE = {
    "Market":1.0, "Kafe":0.6, "Restoran":1.4, "Ulaşım":0.4, "Eczane":0.9,
    "Giyim":1.8, "Elektronik":2.4, "Online Alışveriş":1.6, "Fatura/Servis":2.2
}
PACKAGE_BASES = {"Mini (5)": 5, "Midi (10)": 10, "Maxi (20)": 20}
ALL_PACKAGES = list(PACKAGE_BASES.keys())

# sabit arka plan parametreleri (kullanıcı değiştirmiyor)
TRIALS = 3000
SEED = 123

# ---------- simülasyon (Poisson günlük işlem sayısı destekli) ----------
def simulate_month_poisson(mean_tx_per_day: float,
                           days_in_month: int = 30,
                           trials: int = 3000,
                           seed: int | None = 123,
                           profile_name: str = "Orta Gelir",
                           package_label: str = "Midi (10)") -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed + 1)
    prof = INCOME_PROFILES[profile_name]
    base = PACKAGE_BASES[package_label]
    cats, probs = zip(*CATEGORIES)
    probs = np.array(probs) / np.sum(probs)

    rows = []
    for t in range(trials):
        total = 0.0
        tx_count = 0
        # her gün için poisson ile işlem adedi üret
        for _ in range(days_in_month):
            n_tx = np.random.poisson(lam=mean_tx_per_day)
            if n_tx <= 0:
                continue
            # her işlem için kategori seç ve tutar üret
            chosen_idx = np.random.choice(len(cats), size=n_tx, p=probs)
            for idx in chosen_idx:
                cat = cats[idx]
                amount = float(np.random.lognormal(mean=prof["lognorm_mean"], sigma=prof["lognorm_sd"]))
                amount *= CATEGORY_SCALE.get(cat, 1.0) * prof["spend_mult"]
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
            tx_count += n_tx
        rows.append({"trial": t+1, "Toplam_Katki_TL": round(total, 2), "Toplam_Islem": tx_count})
    return pd.DataFrame(rows)

# ---------- SAYFA: Başlık + minimal kontroller ----------
st.title("📈 Ay Sonu Dağılımı — Hızlı Sunum")
st.caption("Minimum seçimle: Profil, Paket, Gün sayısı ve Günlük işlem (λ). Diğer ayarlar arka planda sabit.")

col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    PROFILE = st.selectbox("Profil", list(INCOME_PROFILES.keys()), index=1)
with col2:
    DAYS = st.slider("Gün sayısı (ay)", 7, 62, 30)
with col3:
    package_label = st.selectbox("Paket (grafik için)", ALL_PACKAGES, index=1)

# Günlük işlem adedi: float, 0.5 adımlı
mean_tx = st.slider("Günlük işlem adedi (ortalama, λ)", min_value=0.5, max_value=8.0, value=2.0, step=0.5,
                    help="Örn. 2.5 seçersen her gün Poisson(2.5) ile işlem sayısı üretilir — daha gerçekçi ve kesirli değer destekli.")

st.markdown("---")

# simülasyonu çalıştır (kullanıcı ayarlarına göre, trials ve seed sabit)
df_month_one = simulate_month_poisson(
    mean_tx_per_day=float(mean_tx),
    days_in_month=DAYS,
    trials=TRIALS,
    seed=SEED,
    profile_name=PROFILE,
    package_label=package_label
)

# ---------- Basit KPI'lar (az ve çarpıcı) ----------
median_v = float(df_month_one["Toplam_Katki_TL"].median())
mean_v = float(df_month_one["Toplam_Katki_TL"].mean())

k1, k2 = st.columns([1,1])
k1.metric("Medyan (ay sonu)", tl(median_v))
k2.metric("Ortalama (ay sonu)", tl(mean_v))

# kısa vurucu cümle
if median_v > 0:
    st.markdown(f"**Hızlı özet:** Seçilen ay ve parametrelere göre tipik kullanıcı (~medyan) ay sonunda yaklaşık **{tl(median_v)}** yuvarlama katkısı biriktirir.")
else:
    st.markdown("Seçimler sonucu medyan katkı 0 görünüyor — günlük işlem adedini veya gün sayısını artırmayı deneyin.")

st.markdown("---")

# ---------- Grafik: histogram + yoğunluk + medyan/ortalama çizgisi ----------
base = alt.Chart(df_month_one)

hist = base.mark_bar(opacity=0.65).encode(
    x=alt.X("Toplam_Katki_TL:Q", bin=alt.Bin(maxbins=40), title="Ay Sonu Toplam Katkı (TL)"),
    y=alt.Y("count():Q", title="Deneme sayısı"),
    tooltip=[alt.Tooltip("count():Q", title="Deneme sayısı")]
).properties(height=300)

density = base.transform_density(
    "Toplam_Katki_TL",
    as_=["Toplam_Katki_TL","Yoğunluk"]
).mark_line(strokeWidth=2).encode(
    x="Toplam_Katki_TL:Q",
    y="Yoğunluk:Q"
)

rule_median = alt.Chart(pd.DataFrame({"x":[median_v]})).mark_rule(color="#1f77b4", strokeWidth=2).encode(x="x:Q")
rule_mean = alt.Chart(pd.DataFrame({"x":[mean_v]})).mark_rule(color="#ff7f0e", strokeWidth=2, strokeDash=[6,4]).encode(x="x:Q")

txt_median = alt.Chart(pd.DataFrame({"x":[median_v], "label":[f"Medyan: {tl(median_v)}"]})).mark_text(
    align="left", dx=5, dy=-10, color="#1f77b4"
).encode(x="x:Q", text="label:N")

txt_mean = alt.Chart(pd.DataFrame({"x":[mean_v], "label":[f"Ortalama: {tl(mean_v)}"]})).mark_text(
    align="left", dx=5, dy=10, color="#ff7f0e"
).encode(x="x:Q", text="label:N")

chart = (hist + density + rule_median + rule_mean + txt_median + txt_mean).properties(
    title=f"{package_label} • Dağılım (n={TRIALS})"
)
st.altair_chart(chart, use_container_width=True)

# ---------- BES Projeksiyonu (ayarları gizli tutulan basit versiyon) ----------
with st.expander("🔒 Basit BES Projeksiyonu (isteğe bağlı detaylar) — aç/kapa"):
    st.write("Simülasyon medyanını (tipik aylık katkı) kullanarak çok basit bir BES birikim projeksiyonu yapar.")
    colA, colB = st.columns([1,1])
    with colA:
        years_to_retire = st.number_input("Kalan süre (yıl)", min_value=1, value=30, step=1)
    with colB:
        annual_return = st.slider("Beklenen yıllık brüt getiri (%)", min_value=0.0, max_value=20.0, value=12.0) / 100.0

    monthly_contrib = max(0.0, median_v)
    st.write(f"Varsayılan aylık katkı (simülasyon medyanı): **{tl(monthly_contrib)}**")

    # hesaplama (basit, masraf/vergiyi dahil etmiyoruz burada)
    months = int(years_to_retire * 12)
    monthly_rate = (annual_return) / 12.0
    if abs(monthly_rate) < 1e-12:
        fv = monthly_contrib * months
    else:
        fv = monthly_contrib * (( (1 + monthly_rate) ** months - 1) / monthly_rate)
    fv = float(fv)

    st.metric("Proj. BES Bakiye (emeklilikte, nominal)", tl(fv))
    st.markdown(f"Kısa not: Bu hesaplama **basit bir projeksiyon** — masraf, vergi, enflasyon ve dinamik katkı değişiklikleri bu modelde yok.")

st.markdown(f"<div style='color: #6b7280; font-size:12px'>Simülasyon oluşturuldu: {datetime.utcnow().date().isoformat()}</div>", unsafe_allow_html=True)
