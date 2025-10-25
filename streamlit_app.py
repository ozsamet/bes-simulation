import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt

st.set_page_config(page_title="Üstü BES'te Kalsın — Ay Sonu Dağılımı", layout="wide")

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
    ("Online Alışveriş", 12), ("Fatura/Servis", 9)
]
CATEGORY_SCALE = {
    "Market":1.0, "Kafe":0.6, "Restoran":1.4, "Ulaşım":0.4, "Eczane":0.9,
    "Giyim":1.8, "Elektronik":2.4, "Online Alışveriş":1.6, "Fatura/Servis":2.2
}
PACKAGE_BASES = {"5'lik Yuvarla": 5, "10'luk Yuvarla": 10, "20'lik Yuvarla": 20}

# ---------- simülasyon ----------
@st.cache_data(show_spinner=False)
def simulate_month_total(base: int, mean_tx_per_day: float,
                         days_in_month: int = 30, trials: int = 5000) -> pd.DataFrame:
    np.random.seed(123)
    random.seed(123)

    rows = []
    cats, probs = zip(*CATEGORIES)
    for t in range(trials):
        total = 0.0
        for _ in range(days_in_month):
            for _ in range(int(mean_tx_per_day)):
                cat = random.choices(cats, weights=probs, k=1)[0]
                amount = float(np.random.lognormal(mean=3.6, sigma=0.5))
                amount *= CATEGORY_SCALE.get(cat, 1.0) * 1.15
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
        rows.append(total)
    return pd.DataFrame({"Toplam_Katki_TL": rows})

# ---------- sayfa ----------
st.title("📈 Ay Sonu Dağılımı")

col1, col2 = st.columns([1,1])
with col1:
    package_label = st.selectbox("Yuvarlama Paketi", list(PACKAGE_BASES.keys()), index=1)
with col2:
    mean_tx = st.slider("Günlük Ortalama İşlem", 1.0, 5.0, 2.0, 0.5)

base = PACKAGE_BASES[package_label]
df_month = simulate_month_total(base, mean_tx)

# ---------- görselleştirme ----------
BINS = 50
base_chart = alt.Chart(df_month)
hist = base_chart.mark_bar(opacity=0.55).encode(
    x=alt.X("Toplam_Katki_TL:Q", bin=alt.Bin(maxbins=BINS), title="Ay Sonu Toplam Katkı (TL)"),
    y=alt.Y("count():Q", title="Adet")
)
density = base_chart.transform_density(
    "Toplam_Katki_TL", as_=["Toplam_Katki_TL","Yoğunluk"]
).mark_line().encode(x="Toplam_Katki_TL:Q", y="Yoğunluk:Q")

st.altair_chart(hist + density, use_container_width=True)

# ---------- özet ----------
s = df_month["Toplam_Katki_TL"].describe()
p5, p95 = df_month["Toplam_Katki_TL"].quantile(0.05), df_month["Toplam_Katki_TL"].quantile(0.95)
m1, m2, m3 = st.columns(3)
m1.metric("Medyan", tl(float(df_month["Toplam_Katki_TL"].median())))
m2.metric("Ortalama", tl(float(s["mean"])))
m3.metric("P5–P95", f"{tl(p5)} — {tl(p95)}")

st.divider()

# ---------- BES projeksiyonu ----------
st.subheader("💰 BES Getiri Projeksiyonu")

yil = st.slider("Sistemde Kalınan Süre (yıl)", 5, 30, 20, 1)
getiri = st.slider("Yıllık Ortalama Getiri (%)", 0.0, 15.0, 6.0, 0.5)

aylik_ort = df_month["Toplam_Katki_TL"].mean()
yillik = aylik_ort * 12
gelecek_deger = yillik * (((1 + getiri/100) ** yil - 1) / (getiri/100))

st.metric("Tahmini Birikim", tl(gelecek_deger))

st.markdown(
    f"Ortalama aylık {tl(aylik_ort)} katkı ile **{yil} yıl** boyunca ve "
    f"yıllık %{getiri} getiriyle tahmini toplam birikim **{tl(gelecek_deger)}** olur."
)

st.markdown("---")
st.markdown("🩵 Küçük yuvarlamalar, büyük birikimlere dönüşür.")
