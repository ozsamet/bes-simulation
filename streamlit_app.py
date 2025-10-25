import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Üstü BES'te Kalsın — Mobil Sunum", layout="wide")

# ----------------- Mobil CSS (light; responsive) -----------------
st.markdown("""
<style>
/* Menü/başlık sadeleştirme (kiosk hissi) */
#MainMenu, footer {visibility: hidden;}
/* Genel metin ve spacing */
.stApp { background: #ffffff; }
.block-container { padding-top: 0.5rem; padding-bottom: 1rem; }
/* Kart metrikleri mobilde tek sütun */
@media (max-width: 640px) {
  .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
  h1, h2, h3 { line-height: 1.15; }
  [data-testid="stMetric"] { margin-bottom: 6px; }
  div[data-testid="stVerticalBlock"] > div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
  .stSlider { padding-top: 0.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ----------------- yardımcılar -----------------
def next_multiple(x: int, base: int) -> int:
    k = x // base
    return (k + 1) * base

def contribution(amount: float, base: int) -> float:
    amt_int = int(round(amount))
    return float(max(0, next_multiple(amt_int, base) - amt_int))

def tl(x: float) -> str:
    return f"{x:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")

# ----------------- sabitler -----------------
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
TRIALS, SEED, DAYS = 3000, 123, 30

# ----------------- simülasyon (cache) -----------------
@st.cache_data(show_spinner=False)
def simulate_month_poisson(base: int, mean_tx_per_day: float,
                           days_in_month: int = DAYS,
                           trials: int = TRIALS,
                           seed: int = SEED) -> pd.DataFrame:
    np.random.seed(seed); random.seed(seed + 1)
    cats, probs = zip(*CATEGORIES); probs = np.array(probs) / np.sum(probs)
    rows = []
    for _ in range(trials):
        total = 0.0
        for _ in range(days_in_month):
            n_tx = np.random.poisson(lam=mean_tx_per_day)
            if n_tx <= 0: continue
            idxs = np.random.choice(len(cats), size=n_tx, p=probs)
            for idx in idxs:
                cat = cats[idx]
                amount = float(np.random.lognormal(mean=3.6, sigma=0.5))
                amount *= CATEGORY_SCALE.get(cat, 1.0) * 1.15
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
        rows.append(total)
    return pd.DataFrame({"Toplam_Katki_TL": rows})

# ----------------- finansal yardımcılar -----------------
def fv_of_monthly(monthly_amount: float, annual_return_pct: float, years: int) -> float:
    r_m = (annual_return_pct / 100.0) / 12.0
    n = years * 12
    if abs(r_m) < 1e-12:
        return monthly_amount * n
    return monthly_amount * (((1 + r_m) ** n - 1) / r_m)

def level_annuity_from_lump(lump: float, annual_rate_pct: float, years: int) -> float:
    r_m = (annual_rate_pct / 100.0) / 12.0
    n = years * 12
    if n <= 0: return 0.0
    if abs(r_m) < 1e-12: return lump / n
    return lump * r_m / (1 - (1 + r_m) ** (-n))

# ----------------- UI -----------------
st.title("📈 Ay Sonu Dağılımı — Üstü BES’te Kalsın")

c1, c2 = st.columns(2)
with c1:
    package_label = st.selectbox("Yuvarlama Paketi", list(PACKAGE_BASES.keys()), index=1)
with c2:
    mean_tx = st.slider("Günlük Ortalama İşlem (λ)", 0.5, 8.0, 2.0, 0.5)

st.divider()

# --- dağılım simülasyonu
base_val = PACKAGE_BASES[package_label]
df = simulate_month_poisson(base=base_val, mean_tx_per_day=float(mean_tx))
median_v = float(df["Toplam_Katki_TL"].median())
mean_v   = float(df["Toplam_Katki_TL"].mean())
p5, p95  = df["Toplam_Katki_TL"].quantile(0.05), df["Toplam_Katki_TL"].quantile(0.95)

m1, m2, m3 = st.columns(3)
m1.metric("Tipik Aylık Katkı (Medyan)", tl(median_v))
m2.metric("Aylık Ortalama Katkı", tl(mean_v))
m3.metric("Band (P5–P95)", f"{tl(float(p5))} — {tl(float(p95))}")

# --- dağılım grafiği
base_chart = alt.Chart(df)
hist = base_chart.mark_bar(opacity=0.6).encode(
    x=alt.X("Toplam_Katki_TL:Q", bin=alt.Bin(maxbins=40), title="Ay Sonu Toplam Katkı (TL)"),
    y=alt.Y("count():Q", title="Deneme sayısı")
).properties(height=300)
density = base_chart.transform_density("Toplam_Katki_TL", as_=["Toplam_Katki_TL","Yoğunluk"]).mark_line(strokeWidth=2).encode(
    x="Toplam_Katki_TL:Q", y="Yoğunluk:Q"
)
rule_med  = alt.Chart(pd.DataFrame({"x":[median_v]})).mark_rule().encode(x="x:Q")
rule_mean = alt.Chart(pd.DataFrame({"x":[mean_v]})).mark_rule(strokeDash=[6,4]).encode(x="x:Q")
st.altair_chart(hist + density + rule_med + rule_mean, use_container_width=True)

st.divider()

# ----------------- BES PROJEKSİYONU -----------------
st.subheader("💰 BES Projeksiyonu (Mobil Düzen)")

colA, colB = st.columns(2)
with colA:
    years_in_system = st.slider("Sistemde Kalınacak Süre (Yıl)", 5, 40, 20, 1)
with colB:
    expected_return = st.slider("Beklenen Yıllık Getiri (%)", 0.0, 20.0, 8.0, 0.5)

monthly_typical = median_v  # tutucu
balance_fv = fv_of_monthly(monthly_typical, expected_return, years_in_system)

# yıllara göre bakiye (çizgi grafik) — önceki line graph
balances = []
r_m = (expected_return/100.0)/12.0
bal = 0.0
for y in range(1, years_in_system+1):
    annual_c = monthly_typical * 12
    if abs(r_m) < 1e-12:
        bal = bal + annual_c
    else:
        bal = bal * (1 + r_m) ** 12 + annual_c * (((1 + r_m) ** 12 - 1) / r_m)
    balances.append({"Yıl": y, "Bakiye": round(bal, 2)})
bal_df = pd.DataFrame(balances)

line_bal = alt.Chart(bal_df).mark_line(point=True).encode(
    x=alt.X("Yıl:O", title="Yıl"),
    y=alt.Y("Bakiye:Q", title="Bakiye (TL)"),
    tooltip=[alt.Tooltip("Yıl:O"), alt.Tooltip("Bakiye:Q", format=".2f")]
).properties(height=260, title="Projeksiyon: Yıllara Göre BES Bakiyesi")

st.altair_chart(line_bal, use_container_width=True)

# 15/20/25 yıl aylık ödeme — tek grafikte çizgi + nokta
ret_rate_post = 4.0
alt_years = [15, 20, 25]
alt_pay = [level_annuity_from_lump(balance_fv, ret_rate_post, y) for y in alt_years]
alt_df = pd.DataFrame({"Ödeme Süresi (Yıl)": alt_years, "Aylık Ödeme (TL)": alt_pay})

pay_line = alt.Chart(alt_df).mark_line(point=True).encode(
    x=alt.X("Ödeme Süresi (Yıl):O", title="Ödeme Süresi"),
    y=alt.Y("Aylık Ödeme (TL):Q", title="Aylık Ödeme (TL)"),
    tooltip=[alt.Tooltip("Ödeme Süresi (Yıl):O"), alt.Tooltip("Aylık Ödeme (TL):Q", format=".2f")]
).properties(height=240, title="Eşit Aylık Ödeme — 15 / 20 / 25 Yıl")

st.altair_chart(pay_line, use_container_width=True)

# mini bilgi kartları
c1, c2, c3 = st.columns(3)
c1.metric("Tipik Aylık Katkı", tl(monthly_typical))
c2.metric("Projeksiyon Bakiyesi", tl(balance_fv))
c3.metric("Emeklilik Getirisi Varsayımı", f"%{ret_rate_post:.1f}")

d1, d2, d3 = st.columns(3)
d1.metric("Aylık Ödeme — 15Y", tl(alt_pay[0]))
d2.metric("Aylık Ödeme — 20Y", tl(alt_pay[1]))
d3.metric("Aylık Ödeme — 25Y", tl(alt_pay[2]))

st.markdown(
    f"**Özet:** {years_in_system} yıl sistemde kalıp aylık ~{tl(monthly_typical)} katkı ve yıllık %{expected_return:.1f} getiriyle "
    f"emeklilik başlangıcında ~{tl(balance_fv)} birikim hedeflenir. 15/20/25 yıl eşit ödeme: "
    f"{tl(alt_pay[0])} / {tl(alt_pay[1])} / {tl(alt_pay[2])} /ay."
)

st.markdown(f"<div style='color:#6b7280;font-size:12px'>Oluşturulma: {datetime.utcnow().date().isoformat()}</div>", unsafe_allow_html=True)
