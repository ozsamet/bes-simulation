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

# ---------- simülasyon (cache'li: ağır işlem burada) ----------
@st.cache_data(show_spinner=False)
def simulate_month_poisson(mean_tx_per_day: float,
                           days_in_month: int = DAYS,
                           trials: int = TRIALS,
                           seed: int | None = SEED,
                           package_label: str = "10'luk Yuvarla") -> pd.DataFrame:
    """Poisson günlük işlem sayısı ile aylık toplam yuvarlama dağılımı üretir.
       Cache'li: aynı parametrelerle tekrar hesaplamaz, hızlı döner."""
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed) + 1)

    base = PACKAGE_BASES[package_label]
    cats, probs = zip(*CATEGORIES)
    probs = np.array(probs) / np.sum(probs)

    rows = []
    for t in range(int(trials)):
        total = 0.0
        tx_count = 0
        for _ in range(int(days_in_month)):
            n_tx = np.random.poisson(lam=mean_tx_per_day)
            if n_tx <= 0:
                continue
            chosen_idx = np.random.choice(len(cats), size=n_tx, p=probs)
            for idx in chosen_idx:
                amount = float(np.random.lognormal(mean=3.6, sigma=0.5))
                amount *= CATEGORY_SCALE.get(cats[idx], 1.0)
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
            tx_count += n_tx
        rows.append({"trial": t+1, "Toplam_Katki_TL": round(total, 2), "Toplam_Islem": tx_count})
    return pd.DataFrame(rows)

# ---------- ANNUNITY / FV yardımcıları ----------
def future_value_of_annuity(monthly, monthly_rate, months):
    if months <= 0:
        return 0.0
    if abs(monthly_rate) < 1e-12:
        return monthly * months
    return monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate)

def level_annuity_payment_from_lump(lump, monthly_payout_rate, months_payout):
    """Verilen birikmiş ana para için aylık sabit ödeme (level annuity) hesapla."""
    if months_payout <= 0:
        return 0.0
    if abs(monthly_payout_rate) < 1e-12:
        return lump / months_payout
    return lump * (monthly_payout_rate) / (1 - (1 + monthly_payout_rate) ** (-months_payout))

# ---------- ARAYÜZ (çok sade) ----------
st.title("📈 Ay Sonu Dağılımı — Üstü BES’te Kalsın")
st.caption("Ay = 30 gün. Sadece paket ve günlük ort. işlem (λ) seçiniz.")

col1, col2 = st.columns([1,1])
with col1:
    package_label = st.selectbox("Yuvarlama Paketi", ALL_PACKAGES, index=1)
with col2:
    mean_tx = st.slider("Günlük ortalama işlem sayısı (λ)", 0.5, 8.0, 2.0, 0.1)

st.markdown("---")

# simülasyon (cache sayesinde hızlı tekrar)
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
density = base.transform_density("Toplam_Katki_TL", as_=["Toplam_Katki_TL","Yoğunluk"]).mark_line(strokeWidth=2).encode(
    x="Toplam_Katki_TL:Q", y="Yoğunluk:Q"
)
rule_median = alt.Chart(pd.DataFrame({"x":[median_v]})).mark_rule(color="#1f77b4", strokeWidth=2).encode(x="x:Q")
rule_mean = alt.Chart(pd.DataFrame({"x":[mean_v]})).mark_rule(color="#ff7f0e", strokeWidth=2, strokeDash=[6,4]).encode(x="x:Q")
txt_median = alt.Chart(pd.DataFrame({"x":[median_v], "label":[f"Medyan: {tl(median_v)}"]})).mark_text(align="left", dx=5, dy=-10, color="#1f77b4").encode(x="x:Q", text="label:N")
txt_mean = alt.Chart(pd.DataFrame({"x":[mean_v], "label":[f"Ortalama: {tl(mean_v)}"]})).mark_text(align="left", dx=5, dy=10, color="#ff7f0e").encode(x="x:Q", text="label:N")

chart = (hist + density + rule_median + rule_mean + txt_median + txt_mean).properties(
    height=320, title=f"{package_label} • Dağılım (n={TRIALS}, 30 gün)"
)
st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# ---------- Basit BES Projeksiyonu (detaylı görünür) ----------
st.subheader("🔒 Basit BES Projeksiyonu")

# girişler
colA, colB, colC = st.columns([1,1,1])
with colA:
    years_to_retire = st.number_input("Kalan süre - birikim dönemi (yıl)", min_value=1, value=30, step=1)
with colB:
    annual_return = st.slider("Beklenen yıllık brüt getiri (%)", 0.0, 20.0, 12.0) / 100.0
with colC:
    annual_fee = st.slider("Yıllık masraf (%)", 0.0, 5.0, 1.0) / 100.0

st.markdown("")  # küçük boşluk

# Kullanıcıdan ödeme süresi ayrı alınır — bu, "emeklilikte kaç yıl ödeme" olduğunu gösterir.
payout_years = st.number_input("Ödeme süresi (yıl) — emeklilikte kaç yıl ödenecek", min_value=1, value=20, step=1)

# monthly contribution using simulation median
monthly_contrib = max(0.0, median_v)

# birikim hesapları
months = int(years_to_retire * 12)
net_annual = annual_return - annual_fee
monthly_rate = net_annual / 12.0
fv = future_value_of_annuity(monthly_contrib, monthly_rate, months)

# payout hesapları: kullanıcı seçeneği ve 20 yıl karşılaştırması (hedef olarak 20y sabit)
months_payout_sel = int(payout_years * 12)
payout_monthly_rate = 0.04 / 12.0  # emeklilikte varsayılan getiri (konservatif)
annuity_sel = level_annuity_payment_from_lump(fv, payout_monthly_rate, months_payout_sel)

# 20 yıl (karşılaştırma)
months_payout_20 = 20 * 12
annuity_20 = level_annuity_payment_from_lump(fv, payout_monthly_rate, months_payout_20)

# gösterimler: bakiye ve iki annuity senaryosu
c1, c2, c3 = st.columns([1,1,1])
c1.metric("Proj. BES Bakiye (emeklilikte)", tl(fv))
c2.metric(f"Aylık gelir — seçilen ödeme süresi ({payout_years}y)", tl(annuity_sel))
c3.metric("Aylık gelir — 20 yıl ödeme (karşılaştırma)", tl(annuity_20))

# fark ve hızlı yorum
diff_pct = 0.0
if annuity_20 != 0:
    diff_pct = (annuity_sel - annuity_20) / annuity_20 * 100.0

st.markdown("---")
st.markdown("**Hızlı Özet:**")
st.markdown(f"- Birikim dönemi: **{years_to_retire}** yıl → yıllık net getiri %{(net_annual*100):.2f}.")
st.markdown(f"- Aylık medyan katkı (simülasyon): **{tl(monthly_contrib)}**.")
st.markdown(f"- Birikmiş tutar (emeklilik başında, nominal): **{tl(fv)}**.")
st.markdown(f"- Seçilen ödeme süresi ({payout_years} yıl) ile aylık gelir: **{tl(annuity_sel)}**.")
st.markdown(f"- 20 yıl ödeme varsayımı ile aylık gelir: **{tl(annuity_20)}**.")
st.markdown(f"- Fark: seçilen ödeme süresi vs 20 yıl = **{diff_pct:.1f}%**.")

# yıllara göre bakiye grafiği (cache edilmesi gereksiz; hafif)
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
    x="Yıl:O", y=alt.Y("Bakiye:Q", title="Bakiye (TL)"),
    tooltip=[alt.Tooltip("Yıl:O"), alt.Tooltip("Bakiye:Q", format=".2f")]
).properties(height=260, title="Projeksiyon: Yıllara Göre BES Bakiyesi")
st.altair_chart(line, use_container_width=True)

st.markdown(f"<div style='color: #6b7280; font-size:12px'>Simülasyon oluşturuldu: {datetime.utcnow().date().isoformat()}</div>", unsafe_allow_html=True)
