import streamlit as st
import pandas as pd
import numpy as np
import random
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
ALL_PACKAGES = list(PACKAGE_BASES.keys())

# ---------- simülasyon ----------
@st.cache_data(show_spinner=False)
def simulate_month_total_one(mean_tx_per_day: float,
                             days_in_month: int,
                             trials: int,
                             seed: int | None,
                             profile_name: str,
                             package_label: str) -> pd.DataFrame:
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
            # her işlem için kategori seç, tutarı üret, seçilen paket bazına göre katkıyı ekle
            chosen = np.random.choice(len(cats), size=n_tx, p=np.array(probs)/np.sum(probs))
            for idx in chosen:
                cat = cats[idx]
                amount = float(np.random.lognormal(mean=prof["lognorm_mean"], sigma=prof["lognorm_sd"]))
                amount *= CATEGORY_SCALE.get(cat, 1.0) * prof["spend_mult"]
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
            tx_count += n_tx
        rows.append({"trial": t+1, "Toplam_Katki_TL": round(total, 2), "Toplam_Islem": tx_count})
    return pd.DataFrame(rows)

# ---------- SAYFA: Ay Sonu Dağılımı ----------
st.title("📈 Ay Sonu Dağılımı (Tüm Kategoriler)")

# Üst kontrol paneli
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
with c1:
    profile = st.selectbox("Profil", list(INCOME_PROFILES.keys()), index=1)
with c2:
    days_in_month = st.slider("Gün sayısı (ay)", 7, 62, 30, step=1)
with c3:
    trials = st.selectbox("Deneme adedi", [1000, 2000, 5000, 10000], index=2)
with c4:
    seed = st.number_input("Rastgelelik (seed)", min_value=0, value=123, step=1)
with c5:
    mean_tx = st.select_slider("Günlük işlem adedi", options=[1,2,3,4,5], value=2)

# Tek paket odaklı grafik seçimi
package_label = st.selectbox("Paket (grafik)", ALL_PACKAGES, index=1, help="Grafik ve detayları bu paket için gösterilir.")

# Simülasyon (seçilen paket)
df_month_one = simulate_month_total_one(
    mean_tx_per_day=float(mean_tx),
    days_in_month=days_in_month,
    trials=trials,
    seed=seed,
    profile_name=profile,
    package_label=package_label
)

st.markdown(
    f"**Profil:** {profile} • **Gün:** {days_in_month} • **Deneme:** {trials} • **Günlük İşlem:** {mean_tx} • **Paket:** {package_label}"
)

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

# Özet metrikler (seçilen paket)
st.subheader("📌 Özet (Seçilen Paket)")
s = df_month_one["Toplam_Katki_TL"].describe()
p5  = float(df_month_one["Toplam_Katki_TL"].quantile(0.05))
p95 = float(df_month_one["Toplam_Katki_TL"].quantile(0.95))

m1, m2, m3 = st.columns(3)
m1.metric("Medyan", tl(float(df_month_one["Toplam_Katki_TL"].median())))
m2.metric("Ortalama", tl(float(s["mean"])))
m3.metric("P5–P95", f"{tl(p5)} — {tl(p95)}")

st.markdown("**İşlem Sayısı (ay)**")
st.table(df_month_one["Toplam_Islem"].describe().to_frame(name="Toplam İşlem").T.round(2))

# --------- Paket Karşılaştırma (Mini vs Midi vs Maxi) ---------
st.divider()
st.subheader("🎯 Paket Karşılaştırma (Aynı parametrelerle)")

def package_summary(pkg: str) -> dict:
    df = simulate_month_total_one(
        mean_tx_per_day=float(mean_tx),
        days_in_month=days_in_month,
        trials=trials,
        seed=seed,
        profile_name=profile,
        package_label=pkg
    )
    return {
        "Paket": pkg,
        "Ortalama_TL": df["Toplam_Katki_TL"].mean(),
        "Medyan_TL": df["Toplam_Katki_TL"].median(),
        "P5_TL": df["Toplam_Katki_TL"].quantile(0.05),
        "P95_TL": df["Toplam_Katki_TL"].quantile(0.95)
    }

compare_df = pd.DataFrame([package_summary(p) for p in ALL_PACKAGES])
st.dataframe(
    compare_df.assign(
        Ortalama=compare_df["Ortalama_TL"].apply(tl),
        Medyan=compare_df["Medyan_TL"].apply(tl),
        P5=compare_df["P5_TL"].apply(tl),
        P95=compare_df["P95_TL"].apply(tl),
    )[["Paket","Ortalama","Medyan","P5","P95"]],
    use_container_width=True
)

# İsteğe bağlı: üç paketi tek grafikte gösteren yoğunluk eğrileri
overlay = None
for pkg in ALL_PACKAGES:
    dfp = simulate_month_total_one(float(mean_tx), days_in_month, trials, seed, profile, pkg)
    ch = alt.Chart(dfp).transform_density(
        "Toplam_Katki_TL",
        as_=["Toplam_Katki_TL","Yoğunluk"]
    ).mark_line().encode(
        x="Toplam_Katki_TL:Q",
        y="Yoğunluk:Q",
        tooltip=["Toplam_Katki_TL:Q","Yoğunluk:Q"],
    ).properties(title=pkg)
    overlay = ch if overlay is None else overlay + ch

st.altair_chart(overlay.resolve_scale(y='independent'), use_container_width=True)

# --------- Dışa Aktar ---------
st.divider()
c_dl1, c_dl2 = st.columns(2)
with c_dl1:
    st.download_button(
        "Seçilen Paket — Ay Sonu Toplam Katkılar (CSV)",
        data=df_month_one.to_csv(index=False).encode("utf-8"),
        file_name=f"ay_sonu_toplam_{package_label.replace(' ','').replace('(','').replace(')','')}.csv",
        mime="text/csv"
    )
with c_dl2:
    st.download_button(
        "Paket Karşılaştırma Özeti (CSV)",
        data=compare_df.to_csv(index=False).encode("utf-8"),
        file_name="paket_karsilastirma_ozet.csv",
        mime="text/csv"
    )

st.caption(
    "Amaç: Kullanıcı hangi paketi seçerse, ay sonunda yaklaşık ne kadar **yuvarlama katkısı** birikeceğini görsün. "
    "Tutar önce tam TL'ye yuvarlanır; paket bazları: Mini=5 TL, Midi=10 TL, Maxi=20 TL."
)
