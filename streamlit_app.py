import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import altair as alt

st.set_page_config(page_title="Üstü BES'te Kalsın — Ay Sonu Dağılımı", layout="wide")

# ---------- yardımcılar ----------
def next_multiple(x: int, base: int) -> int:
    k = x // base
    return (k + 1) * base  # tam kat olsa da bir sonrakine

def contribution(amount: float, base: int) -> float:
    # önce TUTARI TAM SAYI TL'ye yuvarla (0.5 ve üzeri yukarı)
    amt_int = int(round(amount))
    return float(max(0, next_multiple(amt_int, base) - amt_int))

def tl(x): 
    return f"{x:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------- kategoriler / profiller ----------
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

# ---------- tek paket için aylık toplam simülasyonu (TÜM KATEGORİLER) ----------
def simulate_month_total_one(mean_tx_per_day: float,
                             days_in_month: int = 30,
                             trials: int = 5000,   # 5000 deneme
                             seed: int | None = 123,
                             profile_name: str = "Orta Gelir",
                             package_label: str = "Midi (10)") -> pd.DataFrame:
    """
    Tüm kategorilerden, CATEGORIES ağırlıklarına göre işlem üretir.
    Seçilen paket için ay sonu toplam katkı dağılımını döner.
    """
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

# ---------- SAYFA: Ay Sonu Dağılımı ----------
st.title("📈 Ay Sonu Dağılımı (Tüm Kategoriler)")

# seçimler
c1, c2 = st.columns([1,1])
with c1:
    package_label = st.selectbox("Paket", ["Mini (5)", "Midi (10)", "Maxi (20)"], index=1)
with c2:
    mean_tx = st.select_slider(
        "Günlük işlem adedi",
        options=[1, 2, 3, 4, 5],
        value=2,
        help="Günlük ortalama toplam işlem adedi (λ)."
    )

# Tanıtım varsayılanları 
DAYS = 30
TRIALS = 5000
SEED = 123
PROFILE = "Orta Gelir"

df_month_one = simulate_month_total_one(
    mean_tx_per_day=float(mean_tx),
    days_in_month=DAYS,
    trials=TRIALS,
    seed=SEED,
    profile_name=PROFILE,
    package_label=package_label
)

st.markdown(f"**Profil:** {PROFILE} • **Ay:** {DAYS} gün • **Deneme:** {TRIALS}")

# Histogram + yoğunluk 
BINS = 50
base = alt.Chart(df_month_one)

hist = base.mark_bar(opacity=0.55).encode(
    x=alt.X("Toplam_Katki_TL:Q", bin=alt.Bin(maxbins=BINS), title="Ay Sonu Toplam Katkı (TL)"),
    y=alt.Y("count():Q", title="Adet"),
    tooltip=["count():Q"]
).properties(height=280)

density = base.transform_density(
    "Toplam_Katki_TL",
    as_=["Toplam_Katki_TL","Yoğunluk"]
).mark_line().encode(
    x="Toplam_Katki_TL:Q",
    y="Yoğunluk:Q",
    tooltip=["Toplam_Katki_TL:Q","Yoğunluk:Q"]
)

st.altair_chart(hist + density, use_container_width=True)

# Özet metrikler
st.subheader("📌 Özet (Ay Sonu)")
s = df_month_one["Toplam_Katki_TL"].describe()
p5  = float(df_month_one["Toplam_Katki_TL"].quantile(0.05))
p95 = float(df_month_one["Toplam_Katki_TL"].quantile(0.95))

m1, m2, m3 = st.columns(3)
m1.metric("Medyan", tl(float(df_month_one["Toplam_Katki_TL"].median())))
m2.metric("Ortalama", tl(float(s["mean"])))
m3.metric("P5–P95", f"{tl(p5)} — {tl(p95)}")

st.markdown("**İşlem Sayısı (ay)**")
st.table(df_month_one["Toplam_Islem"].describe().to_frame(name="Toplam İşlem").T.round(2))

st.divider()
st.download_button(
    "Ay Sonu Toplam Katkılar (CSV)",
    data=df_month_one.to_csv(index=False).encode("utf-8"),
    file_name=f"ay_sonu_toplam_{package_label.replace(' ','').replace('(','').replace(')','')}_tum_kategoriler.csv",
    mime="text/csv"
)

st.caption("Not: Tutar önce tam TL'ye yuvarlanır; paket bazları: Mini=5 TL, Midi=10 TL, Maxi=20 TL.")
