import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Üstü BES'te Kalsın — Ay Sonu Dağılımı & BES Projeksiyon", layout="wide")

# ---------- yardımcılar ----------
def next_multiple(x: int, base: int) -> int:
    k = x // base
    return (k + 1) * base

def contribution(amount: float, base: int) -> float:
    amt_int = int(round(amount))
    return float(max(0, next_multiple(amt_int, base) - amt_int))

def tl(x):
    return f"{x:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------- sabitler (orijinal) ----------
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

# ---------- simülasyon fonksiyonu ----------
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

# ---------- SAYFA: başlık ve kontroller ----------
st.title("📈 Ay Sonu Dağılımı — Sunum & Basit BES Projeksiyon")
st.caption("Hızlı, çarpıcı özetler. CSV indirme kaldırıldı; görünüm light-friendly (dark mod ile karıştırılmadı).")

# kontrol paneli
col1, col2, col3, col4 = st.columns([1.2,1,1,1])
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
    package_label = st.selectbox("Paket (grafik için)", ALL_PACKAGES, index=1)
with c2:
    mean_tx = st.select_slider("Günlük işlem adedi", options=[1,2,3,4,5], value=2)

st.markdown("---")

# simülasyon
df_month_one = simulate_month_total_one(
    mean_tx_per_day=float(mean_tx),
    days_in_month=DAYS,
    trials=TRIALS,
    seed=SEED,
    profile_name=PROFILE,
    package_label=package_label
)

# ---------- KPI'lar ----------
median_v = float(df_month_one["Toplam_Katki_TL"].median())
mean_v = float(df_month_one["Toplam_Katki_TL"].mean())
p5 = float(df_month_one["Toplam_Katki_TL"].quantile(0.05))
p95 = float(df_month_one["Toplam_Katki_TL"].quantile(0.95))
max_v = float(df_month_one["Toplam_Katki_TL"].max())

# eşik olasılıkları (izleyici etkisi için)
thr_list = [10, 25, 50, 100]
probs = {t: (df_month_one["Toplam_Katki_TL"] >= t).mean() for t in thr_list}

k1, k2, k3, k4 = st.columns([1,1,1,1])
k1.metric("Medyan (ay sonu)", tl(median_v), delta=f"P5–P95: {tl(p5)} — {tl(p95)}")
k2.metric("Ortalama (ay sonu)", tl(mean_v), delta=f"Maks: {tl(max_v)}")
k3.metric(f"%≥{thr_list[2]} TL olasılığı", f"{probs[thr_list[2]]:.1%}")
if probs[thr_list[2]] >= 0.5:
    punch = "Yüksek ihtimal!"
elif probs[thr_list[2]] >= 0.2:
    punch = "Kayda değer ihtimal"
else:
    punch = "Düşük ihtimal"
k4.metric("50 TL üzeri özet", punch, delta=f"%≥50 = {probs[thr_list[2]]:.1%}")

st.markdown("---")

# ---------- GRAFİK: histogram + yoğunluk + çizgiler ----------
base = alt.Chart(df_month_one)

hist = base.mark_bar(opacity=0.6).encode(
    x=alt.X("Toplam_Katki_TL:Q", bin=alt.Bin(maxbins=50), title="Ay Sonu Toplam Katkı (TL)"),
    y=alt.Y("count():Q", title="Adet"),
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
    align="left", dx=5, dy=-10, fontWeight="bold", color="#1f77b4"
).encode(x="x:Q", text="label:N")

txt_mean = alt.Chart(pd.DataFrame({"x":[mean_v], "label":[f"Ortalama: {tl(mean_v)}"]})).mark_text(
    align="left", dx=5, dy=10, fontWeight="bold", color="#ff7f0e"
).encode(x="x:Q", text="label:N")

chart = (hist + density + rule_median + rule_mean + txt_median + txt_mean).properties(title=f"{package_label} • Dağılım (n={TRIALS})")
st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# ---------- HIZLI İSTATİSTİKLER TABLOSU ----------
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

# ---------- BASİT BES PROJEKSİYONU ----------
st.subheader("🔒 Basit BES Projeksiyonu — Bu aylık yuvarlamalar ile ne kadar olur?")

# varsayımlar / kullanıcı girişi
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    years_to_retire = st.number_input("Kalan süre (yıl)", min_value=1, value=30, step=1)
with colB:
    annual_return = st.slider("Beklenen yıllık brüt getiri (%)", min_value=0.0, max_value=20.0, value=12.0) / 100.0
with colC:
    annual_fee = st.slider("Yıllık masraf (%)", min_value=0.0, max_value=5.0, value=1.0) / 100.0
with colD:
    payout_years = st.number_input("Emeklilik ödeme süresi (yıl, annuity)", min_value=5, value=20, step=1)

st.markdown("**Açıklama (basit):** Simülasyondan elde edilen *aylık ortalama yuvarlama* (medyan) her ay BES'e yatırılıyor. Getiri ve masraflar sabit kabul ediliyor; vergi/komisyonlar dahil edilmedi.")

# monthly contribution assumption: use medyan from simulation as typical monthly contributed amount
monthly_contrib = max(0.0, median_v)  # median_v is monthly total rounding
st.markdown(f"**Varsayılan Aylık Katkı (simülasyon medyanı):** {tl(monthly_contrib)}")

# hesaplama
months = int(years_to_retire * 12)
# basit net yıllık getiri = brüt - masraf (yaklaşık); aylık net oran:
net_annual = annual_return - annual_fee
monthly_rate = net_annual / 12.0
if abs(monthly_rate) < 1e-12:
    fv = monthly_contrib * months
else:
    fv = monthly_contrib * (( (1 + monthly_rate) ** months - 1) / monthly_rate)

fv = float(fv)

# emeklilikte aylık gelir (basit sabit annuity assumption)
months_payout = int(payout_years * 12)
# varsayılan payout rate (emeklilik dönemi getiri, konservatif)
payout_annual_return = 0.04
payout_monthly = payout_annual_return / 12.0
if months_payout > 0 and payout_monthly > 0:
    annuity_monthly = fv * (payout_monthly) / (1 - (1 + payout_monthly) ** (-months_payout))
else:
    annuity_monthly = fv / max(1, months_payout)

# gösterimler
col1, col2 = st.columns([1,1])
col1.metric("Proj. BES Bakiye (emeklilikte, nominal)", tl(fv))
col2.metric(f"Beklenen aylık gelir (~{payout_years} yıl ömür)", tl(annuity_monthly))

# yıllara göre bakiye çizgisi (yearly)
balances = []
balance = 0.0
for y in range(1, years_to_retire + 1):
    # yıllık contribution:
    annual_contrib = monthly_contrib * 12
    if monthly_rate == 0:
        balance = balance + annual_contrib
    else:
        # grow previous balance for 12 months
        balance = balance * (1 + monthly_rate) ** 12 + annual_contrib * (( (1 + monthly_rate) ** 12 - 1) / monthly_rate)
    balances.append({"Yıl": y, "Bakiye": round(balance, 2)})

bal_df = pd.DataFrame(balances)
# chart
line = alt.Chart(bal_df).mark_line(point=True).encode(
    x=alt.X("Yıl:O", title="Yıl"),
    y=alt.Y("Bakiye:Q", title="Bakiye (TL)"),
    tooltip=[alt.Tooltip("Yıl:O"), alt.Tooltip("Bakiye:Q", format=".2f")]
).properties(height=240, title="Projeksiyon: Yıllara Göre BES Bakiyesi (basit model)")

st.altair_chart(line, use_container_width=True)

st.markdown(
    f"**Kısa yorum:** Eğer ayda ortalama {tl(monthly_contrib)} yatırılırsa ve yıllık net getiri %{(net_annual*100):.2f} alınırsa, {years_to_retire} yıl sonra yaklaşık **{tl(fv)}** birikmiş olur. "
    f"Bu bakiye, emeklilikte yılda %{payout_annual_return*100:.1f} getiri ve {payout_years} yıl ödeme varsayımlarına göre yaklaşık **{tl(annuity_monthly)}**/ay verir."
)

st.markdown(f"<div style='color: #6b7280; font-size:12px'>Simülasyon oluşturuldu: {datetime.utcnow().date().isoformat()}</div>", unsafe_allow_html=True)
