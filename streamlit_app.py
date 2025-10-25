import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

# sayfa ayarı
st.set_page_config(page_title="Üstü BES'te Kalsın — Ay Sonu Dağılımı", layout="wide")

# ---------- küçük CSS (sunum için temiz görünüm) ----------
st.markdown(
    """
    <style>
    /* sayfa arka planı ve kart benzeri hissiyat */
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
    }
    .big-title {
        font-size:42px;
        font-weight:800;
        color:#0b3d91;
        margin-bottom:6px;
    }
    .subtitle {
        font-size:14px;
        color:#334155;
        margin-top:-8px;
    }
    /* gizle Streamlit menü ve footer (daha temiz sunum) */
    #MainMenu, footer, header {visibility: hidden;}
    /* KPI kutuları */
    .kpi {
        background: linear-gradient(180deg,#ffffff,#f3f7ff);
        border-radius:12px;
        padding:12px;
        box-shadow: 0 6px 18px rgba(11,61,145,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- yardımcılar ----------
def next_multiple(x: int, base: int) -> int:
    k = x // base
    return (k + 1) * base

def contribution(amount: float, base: int) -> float:
    amt_int = int(round(amount))
    return float(max(0, next_multiple(amt_int, base) - amt_int))

def tl(x):
    return f"{x:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------- sabitler (kullandığınız halden aynen) ----------
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

# ---------- simülasyon (mevcut fonksiyonunuzdan küçük değişiklik) ----------
def simulate_month_total_one(mean_tx_per_day: float,
                             days_in_month: int = 30,
                             trials: int = 5000,
                             seed: int | None = 123,
                             profile_name: str = "Orta Gelir",
                             package_label: str = "Midi (10)") -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    prof = INCOME_PROFILES[profile_name]
    base = PACKAGE_BASES[package_label]
    cats, probs = zip(*CATEGORIES)

    rows = []
    for t in range(trials):
        total = 0.0
        tx_count = 0
        for _ in range(days_in_month):
            n_tx = int(mean_tx_per_day)
            if n_tx <= 0:
                continue
            for _ in range(n_tx):
                cat = random.choices(cats, weights=probs, k=1)[0]
                amount = float(np.random.lognormal(mean=prof["lognorm_mean"], sigma=prof["lognorm_sd"]))
                amount *= CATEGORY_SCALE.get(cat, 1.0) * prof["spend_mult"]
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
            tx_count += n_tx
        rows.append({"trial": t+1, "Toplam_Katki_TL": round(total, 2), "Toplam_Islem": tx_count})
    return pd.DataFrame(rows)

# ---------- SAYFA: başlık ve seçimler ----------
st.markdown('<div class="big-title">📈 Ay Sonu Dağılımı — Sunum Versiyonu</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hangi paketi seçerse kullanıcı, ay sonunda ortalama ne kadar "yuvarlama katkısı" biriktirir? Hızlı, çarpıcı özetler ve dağılım gösterimi.</div>', unsafe_allow_html=True)
st.markdown("---")

# kontrol paneli (üslup: minimal)
col1, col2, col3, col4 = st.columns([1.4,1,1,1])
with col1:
    PROFILE = st.selectbox("Profil", list(INCOME_PROFILES.keys()), index=1)
with col2:
    DAYS = st.slider("Gün sayısı (ay)", 7, 62, 30)
with col3:
    TRIALS = st.selectbox("Deneme adedi", [1000, 2000, 5000, 10000], index=2)
with col4:
    SEED = st.number_input("Seed", min_value=0, value=123, step=1)

c1, c2 = st.columns([1.3, 1])
with c1:
    package_label = st.selectbox("Paket (grafik için)", ["Mini (5)", "Midi (10)", "Maxi (20)"], index=1)
with c2:
    mean_tx = st.select_slider("Günlük işlem adedi", options=[1,2,3,4,5], value=2)

st.markdown("")  # küçük boşluk

# simülasyonu çalıştır
df_month_one = simulate_month_total_one(
    mean_tx_per_day=float(mean_tx),
    days_in_month=DAYS,
    trials=TRIALS,
    seed=SEED,
    profile_name=PROFILE,
    package_label=package_label
)

# ---------- ÇARPICI İSTATİSTİKLER (KPI'lar) ----------
median_v = float(df_month_one["Toplam_Katki_TL"].median())
mean_v = float(df_month_one["Toplam_Katki_TL"].mean())
p5 = float(df_month_one["Toplam_Katki_TL"].quantile(0.05))
p95 = float(df_month_one["Toplam_Katki_TL"].quantile(0.95))
max_v = float(df_month_one["Toplam_Katki_TL"].max())

# olasılık eşiği kontrolü — kullanıcıyı etkileyecek bir gösterge:
thresholds = [10, 25, 50, 100]  # TL
probs = {t: (df_month_one["Toplam_Katki_TL"] >= t).mean() for t in thresholds}

k1, k2, k3, k4 = st.columns([1,1,1,1])
with k1:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric(label="Medyan (ay sonu)", value=tl(median_v), delta=f"P5–P95: {tl(p5)} — {tl(p95)}")
    st.markdown('</div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric(label="Ortalama (ay sonu)", value=tl(mean_v), delta=f"Maks: {tl(max_v)}")
    st.markdown('</div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric(label=f">%≥{thresholds[2]} TL olasılığı", value=f"{probs[thresholds[2]]:.1%}")
    st.markdown('</div>'
