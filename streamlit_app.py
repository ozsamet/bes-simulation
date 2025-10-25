import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Üstü BES'te Kalsın Simülasyonu", layout="wide")

# ---------- yardımcılar ----------
def next_multiple(x: int, base: int) -> int:
    k = x // base
    return (k + 1) * base

def contribution(amount: float, base: int) -> float:
    amt_int = int(round(amount))
    return float(max(0, next_multiple(amt_int, base) - amt_int))

def tl(x: float) -> str:
    return f"{x:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x: float) -> str:
    return f"%{x:.1f}"

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
FEE_PCT_ANNUAL = 2.5  # FEE: Yıllık fon işletim gideri

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
    """Aylık düzenli katkının gelecekteki değeri (FV).
    annual_return_pct: YILLIK net getiri yüzdesi (FİG düşülmüş olmalı)."""
    r_m = (annual_return_pct / 100.0) / 12.0
    n = years * 12
    if abs(r_m) < 1e-12:
        return monthly_amount * n
    return monthly_amount * (((1 + r_m) ** n - 1) / r_m)

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

# ---------- BES PROJEKSİYONU ----------
st.subheader("💰 BES Projeksiyonu")

colA, colB, colC = st.columns([1,1,1])
with colA:
    years_in_system = st.slider("Sistemde Kalınacak Süre (Yıl)", 5, 40, 20, 1)
with colB:
    expected_return = st.slider("Reel Beklenen Yıllık Getiri (Brüt, %)", 0.0, 10.0, 4.0, 0.5)
with colC:
    fixed_monthly = st.number_input("Aylık Fix Katkı Payın (TL)", min_value=0.0, value=1750.0, step=50.0)

# NEW: Net getiri = brüt beklenen getiri - yıllık FİG
net_return = expected_return - FEE_PCT_ANNUAL  # FEE
# Not: Net getiri negatif olabilir; formül bunu destekler.

# Yuvarlamadan gelen tipik aylık katkı
monthly_typical = median_v

# FV hesapları (NET getiri ile)
balance_fv_roundup  = fv_of_monthly(monthly_typical, net_return, years_in_system)
balance_fv_fixed    = fv_of_monthly(fixed_monthly, net_return, years_in_system)
balance_fv_both     = fv_of_monthly(fixed_monthly + monthly_typical, net_return, years_in_system)

# Yıllık bazda çizim (Fix + Yuvarlama, NET r ile)
balances = []
r_m = (net_return/100.0)/12.0  # NET aylık oran
bal = 0.0
monthly_total = fixed_monthly + monthly_typical
for y in range(1, years_in_system+1):
    annual_c = monthly_total * 12
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
).properties(height=260, title="Projeksiyon: Yıllara Göre BES Bakiyesi (Fix + Yuvarlama, NET)")
st.altair_chart(line_bal, use_container_width=True)

# Yuvarlama özel metrikler (NET)
total_principal_round = monthly_typical * 12 * years_in_system
gain_component_round  = max(0.0, balance_fv_roundup - total_principal_round)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tipik Aylık Yuvarlama Katkısı", tl(monthly_typical))
c2.metric("Yuvarlamadan Toplam Ana Para", tl(total_principal_round))
c3.metric("Yuvarlamadan Getiri (NET)", tl(gain_component_round))
c4.metric("Net Beklenen Getiri", pct(net_return))  # NEW: Net getiri görünür

# Fix vs Yuvarlama karşılaştırmalı metrikler (NET)
uplift_monthly_pct = (monthly_typical / fixed_monthly) * 100.0 if fixed_monthly > 0 else None
uplift_balance_pct = ((balance_fv_both - balance_fv_fixed) / balance_fv_fixed) * 100.0 if balance_fv_fixed > 1e-9 else None

d1, d2, d3 = st.columns(3)
d1.metric("Fix Aylık Katkı", tl(fixed_monthly))
d2.metric("Aylık Ekstra (Yuvarlama / Fix)", "—" if uplift_monthly_pct is None else pct(uplift_monthly_pct))
d3.metric("Projeksiyon Uplift (Bakiye, NET)", "—" if uplift_balance_pct is None else pct(uplift_balance_pct))

# Özet (NET)
st.markdown(
    f"**Özet:** {years_in_system} yıl boyunca aylık fix katkı **{tl(fixed_monthly)}** ve seçilen paketten gelen tipik yuvarlama **{tl(monthly_typical)}** ile, "
    f"**yıllık brüt %{expected_return:.1f}** getiri ve **yıllık FİG %{FEE_PCT_ANNUAL:.1f}** sonrası **net %{net_return:.1f}** varsayımıyla "
    f"emeklilik başlangıcında yaklaşık **{tl(balance_fv_both)}** birikim oluşur. "
    f"Sadece fix katkı olsaydı **{tl(balance_fv_fixed)}** olurdu; yuvarlama eklemesi bakiyeyi NET bazda "
    f"{'—' if uplift_balance_pct is None else pct(uplift_balance_pct)} artırır."
)
