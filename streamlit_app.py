import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Üstü BES'te Kalsın — Hızlı Sunum", layout="wide")

# ---------- yardımcı fonksiyonlar ----------
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
CATEGORY_SCALE = {
    "Market":1.0, "Kafe":0.6, "Restoran":1.4, "Ulaşım":0.4, "Eczane":0.9,
    "Giyim":1.8, "Elektronik":2.4, "Online Alışveriş":1.6, "Fatura/Servis":2.2
}

PACKAGE_BASES = {"5'lik Yuvarla": 5, "10'luk Yuvarla": 10, "20'lik Yuvarla": 20}
ALL_PACKAGES = list(PACKAGE_BASES.keys())

TRIALS = 3000
SEED = 123
DAYS = 30  # sabit: 1 ay = 30 gün

# ---------- simülasyon ----------
def simulate_month_poisson(mean_tx_per_day: float,
                           days_in_month: int = DAYS,
                           trials: int = TRIALS,
                           seed: int | None = SEED,
                           package_label: str = "10'luk Yuvarla") -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed + 1)

    base = PACKAGE_BASES[package_label]
    cats, probs = zip(*CATEGORIES)
    probs = np.array(probs) / np.sum(probs)

    rows = []
    for t in range(trials):
        total = 0.0
        tx_count = 0
        for _ in range(days_in_month):
            n_tx = np.random.poisson(lam=mean_tx_per_day)
            if n_tx <= 0:
                continue
            chosen_idx = np.random.choice(len(cats), size=n_tx, p=probs)
            for idx in chosen_idx:
                cat = cats[idx]
                amount = float(np.random.lognormal(mean=3.6, sigma=0.5))
                amount *= CATEGORY_SCALE.get(cat, 1.0)
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
            tx_count += n_tx
        rows.append({"trial": t+1, "Toplam_Katki_TL": round(total, 2), "Toplam_Islem": tx_count})
    return pd.DataFrame(rows)

# ---------- ARAYÜZ ----------
st.title("📈 Ay Sonu Dağılımı — Üstü BES’te Kalsın")
st.caption("Ay = 30 gün (sabit).")

col1, col2 = st.columns([1,1])
with col1:
    package_label = st.selectbox("Yuvarlama Paketi", ALL_PACKAGES, index=1)
with col2:
    mean_tx = st.slider("Günlük ortalama işlem sayısı (λ)", 0.5, 8.0, 2.0, 0.5)

st.markdown("---")

df_month_one = simulate_month_poisson(
    mean_tx_per_day=float(mean_tx),
    days_in_month=DAYS,
    trials=TRIALS,
    seed=SEED,
    package_label=package_label
)

median_v = float(df_month_one["Toplam_Katki_TL"].median())
mean_v = float(df_month_one["Toplam_Katki_TL"].mean())

k1, k2 = st.columns([1,1])
k1.metric("Medyan (ay sonu)", tl(median_v))
k2.metric("Ortalama (ay sonu)", tl(mean_v))

st.markdown("---")

# ---------- Grafik ----------
base = alt.Chart(df_month_one)
hist = base.mark_bar(opacity=0.65).encode(
    x=alt.X("Toplam_Katki_TL:Q", bin=alt.Bin(maxbins=40), title="Ay Sonu Toplam Katkı (TL)"),
    y=alt.Y("count():Q", title="Deneme sayısı"),
)
density = base.transform_density(
    "Toplam_Katki_TL", as_=["Toplam_Katki_TL","Yoğunluk"]
).mark_line(strokeWidth=2).encode(
    x="Toplam_Katki_TL:Q", y="Yoğunluk:Q"
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
    height=300, title=f"{package_label} • Dağılım (n={TRIALS}, 30 gün)"
)
st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# ---------- Yüzdelik Tablosu ----------
pct_df = pd.DataFrame({
    "Ölçüt": ["Min", "P5", "P10", "Medyan", "P75", "P90", "P95", "Maks"],
    "Değer (TL)": [
        df_month_one["Toplam_Katki_TL"].min(),
        df_month_one["Toplam_Katki_TL"].quantile(0.05),
        df_month_one["Toplam_Katki_TL"].quantile(0.10),
        df_month_one["Toplam_Katki_TL"].quantile(0.50),
        df_month_one["Toplam_Katki_TL"].quantile(0.75),
        df_month_one["Toplam_Katki_TL"].quantile(0.90),
        df_month_one["Toplam_Katki_TL"].quantile(0.95),
        df_month_one["Toplam_Katki_TL"].max()
    ]
})
pct_df["Değer (TL)"] = pct_df["Değer (TL)"].apply(lambda x: tl(float(x)))
st.table(pct_df)

st.markdown("---")

# ---------- BES Projeksiyonu ----------
st.subheader("🔒 Basit BES Projeksiyonu")

colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    years_to_retire = st.number_input("Kalan süre (yıl)", min_value=1, value=30, step=1)
with colB:
    annual_return = st.slider("Yıllık getiri (%)", 0.0, 20.0, 12.0) / 100.0
with colC:
    annual_fee = st.slider("Yıllık kesinti (%)", 0.0, 5.0, 1.0) / 100.0
with colD:
    payout_years = st.number_input("Ödeme süresi (yıl)", min_value=5, value=20, step=1)

monthly_contrib = median_v
months = int(years_to_retire * 12)
net_annual = annual_return - annual_fee
monthly_rate = net_annual / 12.0

if abs(monthly_rate) < 1e-12:
    fv = monthly_contrib * months
else:
    fv = monthly_contrib * (((1 + monthly_rate) ** months - 1) / monthly_rate)

months_payout = int(payout_years * 12)
payout_monthly = 0.04 / 12.0
if months_payout > 0:
    annuity_monthly = fv * (payout_monthly) / (1 - (1 + payout_monthly) ** (-months_payout))
else:
    annuity_monthly = fv / max(1, months_payout)

col1, col2 = st.columns([1,1])
col1.metric("Proj. BES Bakiye (emeklilikte)", tl(fv))
col2.metric(f"Beklenen aylık gelir (~{payout_years} yıl)", tl(annuity_monthly))

balances = []
balance = 0.0
for y in range(1, years_to_retire + 1):
    annual_contrib = monthly_contrib * 12
    if abs(monthly_rate) < 1e-12:
        balance += annual_contrib
    else:
        balance = balance * (1 + monthly_rate) ** 12 + annual_contrib * (((1 + monthly_rate) ** 12 - 1) / monthly_rate)
    balances.append({"Yıl": y, "Bakiye": round(balance, 2)})

bal_df = pd.DataFrame(balances)
line = alt.Chart(bal_df).mark_line(point=True).encode(
    x="Yıl:O", y="Bakiye:Q"
).properties(height=240, title="Projeksiyon: BES Bakiyesi")

st.altair_chart(line, use_container_width=True)

st.markdown(f"<div style='color: #6b7280; font-size:12px'>Simülasyon oluşturuldu: {datetime.utcnow().date().isoformat()}</div>", unsafe_allow_html=True)
