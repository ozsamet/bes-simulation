import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Üstü BES'te Kalsın — Ay Sonu Dağılımı", layout="wide")

# ---------- yardımcılar ----------
def next_multiple(x: int, base: int) -> int:
    k = x // base
    return (k + 1) * base

def contribution(amount: float, base: int) -> float:
    amt_int = int(round(amount))
    return float(max(0, next_multiple(amt_int, base) - amt_int))

def tl(x: float) -> str:
    return f"{x:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------- sabitler ----------
CATEGORIES = [
    ("Market", 24), ("Kafe", 12), ("Restoran", 14), ("Ulaşım", 10),
    ("Eczane", 6), ("Giyim", 8), ("Elektronik", 5),
    ("Online Alışveriş", 12), ("Fatura/Servis", 9),
]
CATEGORY_SCALE = {
    "Market":1.0, "Kafe":0.6, "Restoran":1.4, "Ulaşım":0.4, "Eczane":0.9,
    "Giyim":1.8, "Elektronik":2.4, "Online Alışveriş":1.6, "Fatura/Servis":2.2
}
PACKAGE_BASES = {"5'lik Yuvarla": 5, "10'luk Yuvarla": 10, "20'lik Yuvarla": 20}
TRIALS = 3000
SEED = 123
DAYS = 30  # sabit

# ---------- simülasyon (cache) ----------
@st.cache_data(show_spinner=False)
def simulate_month_poisson(base: int, mean_tx_per_day: float,
                           days_in_month: int = DAYS,
                           trials: int = TRIALS,
                           seed: int = SEED) -> pd.DataFrame:
    np.random.seed(seed); random.seed(seed + 1)
    cats, probs = zip(*CATEGORIES)
    probs = np.array(probs) / np.sum(probs)

    rows = []
    for _ in range(trials):
        total = 0.0
        for _ in range(days_in_month):
            n_tx = np.random.poisson(lam=mean_tx_per_day)
            if n_tx <= 0:
                continue
            idxs = np.random.choice(len(cats), size=n_tx, p=probs)
            for idx in idxs:
                cat = cats[idx]
                amount = float(np.random.lognormal(mean=3.6, sigma=0.5))
                amount *= CATEGORY_SCALE.get(cat, 1.0) * 1.15
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
        rows.append(total)
    return pd.DataFrame({"Toplam_Katki_TL": rows})

# ---------- finansal yardımcılar ----------
def fv_of_monthly(monthly_amount: float, annual_return_pct: float, years: int) -> float:
    r_m = (annual_return_pct / 100.0) / 12.0
    n = years * 12
    if abs(r_m) < 1e-12:
        return monthly_amount * n
    return monthly_amount * (((1 + r_m) ** n - 1) / r_m)

def level_annuity_from_lump(lump: float, annual_rate_pct: float, years: int) -> float:
    r_m = (annual_rate_pct / 100.0) / 12.0
    n = years * 12
    if n <= 0:
        return 0.0
    if abs(r_m) < 1e-12:
        return lump / n
    return lump * r_m / (1 - (1 + r_m) ** (-n))

# ---------- UI ----------
st.title("📈 Ay Sonu Dağılımı — Üstü BES’te Kalsın")

col1, col2 = st.columns([1,1])
with col1:
    package_label = st.selectbox("Yuvarlama Paketi", list(PACKAGE_BASES.keys()), index=1)
with col2:
    mean_tx = st.slider("Günlük Ortalama İşlem (λ)", 0.5, 8.0, 2.0, 0.5)

st.markdown("---")

# Simülasyon ve dağılım özetleri
base_val = PACKAGE_BASES[package_label]
df = simulate_month_poisson(base=base_val, mean_tx_per_day=float(mean_tx))

median_v = float(df["Toplam_Katki_TL"].median())
mean_v   = float(df["Toplam_Katki_TL"].mean())
p5, p95  = df["Toplam_Katki_TL"].quantile(0.05), df["Toplam_Katki_TL"].quantile(0.95)

k1, k2, k3 = st.columns(3)
k1.metric("Tipik Aylık Katkı (Medyan)", tl(median_v))
k2.metric("Aylık Ortalama Katkı", tl(mean_v))
k3.metric("Dağılım Bandı (P5–P95)", f"{tl(float(p5))} — {tl(float(p95))}")

# Grafik (histogram + yoğunluk + işaretçiler)
base_chart = alt.Chart(df)
hist = base_chart.mark_bar(opacity=0.6).encode(
    x=alt.X("Toplam_Katki_TL:Q", bin=alt.Bin(maxbins=40), title="Ay Sonu Toplam Katkı (TL)"),
    y=alt.Y("count():Q", title="Deneme sayısı")
)
density = base_chart.transform_density(
    "Toplam_Katki_TL", as_=["Toplam_Katki_TL","Yoğunluk"]
).mark_line(strokeWidth=2).encode(x="Toplam_Katki_TL:Q", y="Yoğunluk:Q")
rule_med  = alt.Chart(pd.DataFrame({"x":[median_v]})).mark_rule().encode(x="x:Q")
rule_mean = alt.Chart(pd.DataFrame({"x":[mean_v]})).mark_rule(strokeDash=[6,4]).encode(x="x:Q")
st.altair_chart(hist + density + rule_med + rule_mean, use_container_width=True)

st.markdown("---")

# ---------- BES PROJEKSİYONU (3 alternatif aynı grafikte) ----------
st.subheader("💰 BES Projeksiyonu")

colA, colB = st.columns([1,1])
with colA:
    years_in_system = st.slider("Sistemde Kalınacak Süre (Yıl)", 5, 40, 20, 1)
with colB:
    expected_return = st.slider("Beklenen Yıllık Getiri (%)", 0.0, 20.0, 8.0, 0.5)

monthly_typical = median_v  # medyan daha tutucu
balance_fv      = fv_of_monthly(monthly_typical, expected_return, years_in_system)

# Alternatif ödeme süreleri (15/20/25 yıl) — emeklilik dönemi getiri varsayımı %4
ret_rate_post = 4.0
alts = []
for yrs in (15, 20, 25):
    alts.append({
        "Ödeme Süresi (Yıl)": yrs,
        "Aylık Ödeme (TL)": level_annuity_from_lump(balance_fv, ret_rate_post, yrs)
    })
alt_df = pd.DataFrame(alts)

# Mini kartlar
c1, c2, c3 = st.columns(3)
c1.metric("Tipik Aylık Katkı", tl(monthly_typical))
c2.metric("Projeksiyon Bakiyesi", tl(balance_fv))
c3.metric("Emeklilik Dönemi Getirisi", f"%{ret_rate_post:.1f}")

# Tek grafikte üç alternatif (sütun grafik + değer etiketleri)
bars = alt.Chart(alt_df).mark_bar().encode(
    x=alt.X("Ödeme Süresi (Yıl):O", title="Ödeme Süresi"),
    y=alt.Y("Aylık Ödeme (TL):Q", title="Aylık Ödeme (TL)"),
    tooltip=[alt.Tooltip("Ödeme Süresi (Yıl):O"), alt.Tooltip("Aylık Ödeme (TL):Q", format=".2f")]
).properties(height=280, title="Eşit Aylık Ödeme — 15 / 20 / 25 Yıl")

labels = alt.Chart(alt_df).mark_text(dy=-5).encode(
    x="Ödeme Süresi (Yıl):O",
    y="Aylık Ödeme (TL):Q",
    text=alt.Text("Aylık Ödeme (TL):Q", format=".0f")
)

st.altair_chart(bars + labels, use_container_width=True)

# Kısa kurumsal özet
st.markdown(
    f"**Özet:** {years_in_system} yıl sistemde kalıp aylık ~{tl(monthly_typical)} katkı ve yıllık %{expected_return:.1f} getiride "
    f"emeklilik başlangıcında ~{tl(balance_fv)} birikim; bu tutar 15/20/25 yılda sırasıyla "
    f"{tl(alt_df.loc[alt_df['Ödeme Süresi (Yıl)']==15, 'Aylık Ödeme (TL)'].iloc[0])} / "
    f"{tl(alt_df.loc[alt_df['Ödeme Süresi (Yıl)']==20, 'Aylık Ödeme (TL)'].iloc[0])} / "
    f"{tl(alt_df.loc[alt_df['Ödeme Süresi (Yıl)']==25, 'Aylık Ödeme (TL)'].iloc[0])} aylık ödemeye karşılık gelir."
)

st.markdown(f"<div style='color:#6b7280;font-size:12px'>Oluşturulma: {datetime.utcnow().date().isoformat()}</div>", unsafe_allow_html=True)
