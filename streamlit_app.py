import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Üstü BES'te Kalsın Simülasyonu", layout="wide")

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
TRIALS = 2000
SEED = 123
DAYS = 31

# ------------------  VEKTÖRİZE SİMÜLASYON ------------------
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def simulate_month_poisson(base: int,
                           mean_tx_per_day: float,
                           days_in_month: int = DAYS,
                           trials: int = TRIALS,
                           seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats, probs = zip(*CATEGORIES)
    probs = np.array(probs, dtype=np.float64)
    probs /= probs.sum()

    # 1) Her deneme için aylık toplam işlem sayısı ~ Poisson(λ * DAYS)
    lam_month = float(mean_tx_per_day) * float(days_in_month)
    n_per_trial = rng.poisson(lam=lam_month, size=trials).astype(np.int64)

    total_tx = int(n_per_trial.sum())
    if total_tx == 0:
        return pd.DataFrame({"Toplam_Katki_TL": np.zeros(trials, dtype=np.float64)})

    # 2) Tüm işlemler için kategori ve tutarları topluca üret
    cat_idx = rng.choice(len(cats), size=total_tx, p=probs, shuffle=True)
    scale_vec = np.array([CATEGORY_SCALE[c] for c in cats], dtype=np.float64)

    # 3) İşlem tutarı: lognormal * kategori ölçeği * 1.15, alt sınır 5 TL
    amounts = rng.lognormal(mean=3.6, sigma=0.5, size=total_tx)
    amounts = np.maximum(5.0, amounts * scale_vec[cat_idx] * 1.15)

    # 4) Yuvarlama katkısı (tamamen vektörize)
    amt_int = np.rint(amounts).astype(np.int64)
    contrib = (base - (amt_int % base)) % base
    contrib = contrib.astype(np.float64)

    # 5) İşlemleri denemelere dağıt ve topla
    trial_ids = np.repeat(np.arange(trials, dtype=np.int64), n_per_trial)
    totals = np.zeros(trials, dtype=np.float64)
    np.add.at(totals, trial_ids, contrib)

    return pd.DataFrame({"Toplam_Katki_TL": totals})

def fv_of_monthly(monthly_amount: float, annual_return_pct: float, years: int) -> float:
    """Aylık eşit katkıların gelecek değeri (standart annüite formülü)."""
    r_m = (annual_return_pct / 100.0) / 12.0
    n = years * 12
    if abs(r_m) < 1e-12:
        return monthly_amount * n
    return monthly_amount * (((1 + r_m) ** n - 1) / r_m)

st.title("📈 Ay Sonu Dağılımı — Üstü BES’te Kalsın")

col1, col2 = st.columns([1,1])
with col1:
    package_label = st.selectbox("Yuvarlama Paketi", list(PACKAGE_BASES.keys()), index=1)
with col2:
    mean_tx = st.slider("Günlük Ortalama İşlem (λ)", 0.5, 8.0, 2.0, 0.5)

st.markdown("---")

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

st.subheader("💰 BES Projeksiyonu")

colA, colB, colC = st.columns([1,1,1])
with colA:
    years_in_system = st.slider("Sistemde Kalınacak Süre (Yıl)", 5, 40, 20, 1)
with colB:
    expected_return = st.slider("Reel Beklenen Yıllık Getiri (%)", 0.0, 10.0, 4.0, 1.0)
with colC:
    fixed_monthly: int = st.number_input("Aylık Fix Katkı Payın (TL)", min_value=0, value=1750, step=50, format="%d")

# --- Aylık tipik yuvarlama + fix katkı
monthly_typical = median_v
monthly_total   = fixed_monthly + monthly_typical

# --- Gelecek değerler (metrikler)
balance_fv_roundup  = fv_of_monthly(monthly_typical, expected_return, years_in_system)
balance_fv_fixed    = fv_of_monthly(fixed_monthly, expected_return, years_in_system)
balance_fv_both     = fv_of_monthly(monthly_total, expected_return, years_in_system)


balances = []
for y in range(1, years_in_system+1):
    bal_y = fv_of_monthly(monthly_total, expected_return, y)  # her yılın sonundaki FV
    balances.append({"Yıl": y, "Bakiye": round(bal_y, 2)})

bal_df = pd.DataFrame(balances)
line_bal = alt.Chart(bal_df).mark_line(point=True).encode(
    x=alt.X("Yıl:O", title="Yıl"),
    y=alt.Y("Bakiye:Q", title="Bakiye (TL)"),
    tooltip=[alt.Tooltip("Yıl:O"), alt.Tooltip("Bakiye:Q", format=".2f")]
).properties(height=260, title="Projeksiyon: Yıllara Göre BES Bakiyesi (Fix + Yuvarlama)")
st.altair_chart(line_bal, use_container_width=True)


total_principal_round = monthly_typical * 12 * years_in_system
gain_component_round  = max(0.0, balance_fv_roundup - total_principal_round)

c1, c2, c3 = st.columns(3)
c1.metric("Tipik Aylık Yuvarlama Katkısı", tl(monthly_typical))
c2.metric("Yuvarlamadan Toplam Ana Para", tl(total_principal_round))
c3.metric("Yuvarlamadan Getiri Kazancı", tl(gain_component_round))

uplift_monthly_pct = (monthly_typical / fixed_monthly) * 100.0 if fixed_monthly > 0 else None
uplift_balance_pct = ((balance_fv_both - balance_fv_fixed) / balance_fv_fixed) * 100.0 if balance_fv_fixed > 1e-9 else None

d1, d2, d3 = st.columns(3)
d1.metric("Fix Aylık Katkı", tl(float(fixed_monthly)))
d2.metric("Aylık Ekstra (Yuvarlama / Fix)", "—" if uplift_monthly_pct is None else pct(uplift_monthly_pct))
d3.metric("Projeksiyon Uplift (Bakiye)", "—" if uplift_balance_pct is None else pct(uplift_balance_pct))

st.markdown(
    f"**Özet:** {years_in_system} yıl boyunca aylık fix katkı **{tl(float(fixed_monthly))}** ve seçilen paketten gelen tipik yuvarlama **{tl(monthly_typical)}** ile, "
    f"yıllık %{expected_return:.1f} reel getiri varsayımında emeklilik başlangıcında yaklaşık **{tl(balance_fv_both)}** birikim oluşur. "
    f"Sadece fix katkı olsaydı **{tl(balance_fv_fixed)}** olurdu; yuvarlama, bakiyeyi yaklaşık "
    f"{'—' if uplift_balance_pct is None else pct(uplift_balance_pct)} oranında artırır."
)
