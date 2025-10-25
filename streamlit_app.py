import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Üstü BES'te Kalsın — Dashboard", layout="wide")

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
TRIALS, SEED, DAYS = 3000, 123, 30  # 1 ay = 30 gün sabit

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
    if n <= 0:
        return 0.0
    if abs(r_m) < 1e-12:
        return lump / n
    return lump * r_m / (1 - (1 + r_m) ** (-n))

# ----------------- query params (paylaşım desteği) -----------------
def get_params():
    try:
        # Yeni API
        qp = st.query_params
        return {
            "pkg": qp.get("pkg", None),
            "lam": float(qp.get("lam", 0)),
            "yrs": int(qp.get("yrs", 0)),
            "ret": float(qp.get("ret", 0)),
        }
    except:
        # Eski API fallback
        qp = st.experimental_get_query_params()
        return {
            "pkg": qp.get("pkg", [None])[0],
            "lam": float(qp.get("lam", [0])[0]),
            "yrs": int(qp.get("yrs", [0])[0]),
            "ret": float(qp.get("ret", [0])[0]),
        }

def set_params(pkg, lam, yrs, ret):
    try:
        st.query_params["pkg"] = pkg
        st.query_params["lam"] = lam
        st.query_params["yrs"] = yrs
        st.query_params["ret"] = ret
    except:
        st.experimental_set_query_params(pkg=pkg, lam=lam, yrs=yrs, ret=ret)

# ----------------- UI: üst başlık ve kontroller -----------------
st.title("📊 Üstü BES’te Kalsın — Ana Dashboard")

# Senaryo kısayolları
sc_col1, sc_col2, sc_col3 = st.columns([1.2, 1, 1])
with sc_col1:
    package_label = st.selectbox("Yuvarlama Paketi", list(PACKAGE_BASES.keys()), index=1)
with sc_col2:
    scenario = st.radio("Senaryo", ["Az", "Orta", "Yoğun", "Özel"], horizontal=True)
with sc_col3:
    # yıllık projeksiyon girdileri
    years_in_system = st.slider("Sistemde Kalınacak Süre (Yıl)", 5, 40, 20, 1)

# λ seçimi (0.5 adım). Senaryolar hızlı seçim sağlar.
default_lambda = {"Az": 1.5, "Orta": 2.5, "Yoğun": 4.0}.get(scenario, 2.0)
if scenario == "Özel":
    mean_tx = st.slider("Günlük Ortalama İşlem (λ)", 0.5, 8.0, 2.0, 0.5)
else:
    mean_tx = st.slider("Günlük Ortalama İşlem (λ)", 0.5, 8.0, default_lambda, 0.5)

expected_return = st.slider("Beklenen Yıllık Getiri (%)", 0.0, 20.0, 8.0, 0.5)

st.divider()

# ----------------- Simülasyon ve dağılım -----------------
base_val = PACKAGE_BASES[package_label]
df = simulate_month_poisson(base=base_val, mean_tx_per_day=float(mean_tx))
median_v = float(df["Toplam_Katki_TL"].median())
mean_v   = float(df["Toplam_Katki_TL"].mean())
p5, p95  = df["Toplam_Katki_TL"].quantile(0.05), df["Toplam_Katki_TL"].quantile(0.95)

# Dinamik manşet
st.subheader(f"{package_label}: Tipik kullanıcı ayda ~{tl(median_v)} biriktirir (λ={mean_tx}). Yıllık ≈ {tl(median_v*12)}.")

# KPI kartları
k1, k2, k3 = st.columns(3)
k1.metric("Tipik Aylık Katkı (Medyan)", tl(median_v))
k2.metric("Aylık Ortalama Katkı", tl(mean_v))
k3.metric("Dağılım Bandı (P5–P95)", f"{tl(float(p5))} — {tl(float(p95))}")

# Dağılım grafiği
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
st.subheader("💰 BES Projeksiyonu")

monthly_typical = median_v  # daha tutucu
balance_fv = fv_of_monthly(monthly_typical, expected_return, years_in_system)

# Yıllara göre bakiye: line chart
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

# 15/20/25 yıl tek grafikte monthly annuity (line+point)
ret_rate_post = 4.0
alt_years = [15, 20, 25]
alt_pay = [level_annuity_from_lump(balance_fv, ret_rate_post, y) for y in alt_years]
alt_df = pd.DataFrame({"Ödeme Süresi (Yıl)": alt_years, "Aylık Ödeme (TL)": alt_pay})

pay_line = alt.Chart(alt_df).mark_line(point=True).encode(
    x=alt.X("Ödeme Süresi (Yıl):O", title="Ödeme Süresi"),
    y=alt.Y("Aylık Ödeme (TL):Q", title="Aylık Ödeme (TL)"),
    tooltip=[alt.Tooltip("Ödeme Süresi (Yıl):O"), alt.Tooltip("Aylık Ödeme (TL):Q", format=".2f")]
).properties(height=240, title="Eşit Aylık Ödeme — 15 / 20 / 25 Yıl (Vars.: %4)")
st.altair_chart(pay_line, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Projeksiyon Bakiyesi", tl(balance_fv))
c2.metric("Aylık Ödeme — 15Y", tl(alt_pay[0]))
c3.metric("Aylık Ödeme — 20Y / 25Y", f"{tl(alt_pay[1])} / {tl(alt_pay[2])}")

st.markdown(
    f"**Özet:** {years_in_system} yıl, aylık ~{tl(monthly_typical)} katkı ve yıllık %{expected_return:.1f} ile "
    f"emeklilik başlangıcında ~{tl(balance_fv)}. 15/20/25 yıl eşit ödeme: "
    f"{tl(alt_pay[0])} / {tl(alt_pay[1])} / {tl(alt_pay[2])} /ay."
)

st.divider()

# ----------------- Paket Karşılaştırma (opsiyonel ama etkili) -----------------
with st.expander("🔎 Aynı λ için 5/10/20 Karşılaştırma"):
    def pkg_median(pkg_label: str) -> float:
        dfp = simulate_month_poisson(base=PACKAGE_BASES[pkg_label], mean_tx_per_day=float(mean_tx))
        return float(dfp["Toplam_Katki_TL"].median())
    rows = [{"Paket": k, "Medyan Aylık Katkı (TL)": pkg_median(k)} for k in PACKAGE_BASES.keys()]
    comp_df = pd.DataFrame(rows)
    st.dataframe(
        comp_df.assign(**{"Medyan (TL)": comp_df["Medyan Aylık Katkı (TL)"].apply(tl)})[["Paket","Medyan (TL)"]],
        use_container_width=True
    )
    bar = alt.Chart(comp_df).mark_bar().encode(
        x=alt.X("Paket:N", title="Paket"),
        y=alt.Y("Medyan Aylık Katkı (TL):Q", title="Medyan (TL)"),
        tooltip=["Paket","Medyan Aylık Katkı (TL):Q"]
    ).properties(height=220)
    st.altair_chart(bar, use_container_width=True)

# ----------------- Paylaşılabilir Link (QR için) -----------------
colshare1, colshare2 = st.columns([1,1])
with colshare1:
    if st.button("🔗 Paylaşılabilir Bağlantı Oluştur"):
        set_params(pkg=package_label, lam=mean_tx, yrs=years_in_system, ret=expected_return)
        st.success("Bağlantı adres çubuğuna yazıldı — QR için hazır.")
with colshare2:
    prms = get_params()
    if prms["pkg"]:
        st.caption(f"Yüklü Senaryo: pkg={prms['pkg']}, λ={prms['lam']}, yıl={prms['yrs']}, getiri={prms['ret']}")

st.markdown(f"<div style='color:#6b7280;font-size:12px'>Oluşturulma: {datetime.utcnow().date().isoformat()}</div>", unsafe_allow_html=True)
