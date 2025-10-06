import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import altair as alt

hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""
st.set_page_config(page_title="Üstü BES'te Kalsın", layout="wide")

# ---------- yardımcılar ----------
def next_multiple(x: int, base: int) -> int:
    k = x // base
    return (k + 1) * base  # tam kat olsa da bir sonrakine

def contribution(amount: float, base: int) -> float:
    # önce TUTARI TAM SAYI TL'ye yuvarla (0.5 ve üzeri yukarı)
    amt_int = int(round(amount))
    return float(max(0, next_multiple(amt_int, base) - amt_int))

def tl(x): return f"{x:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")

def pick_weighted(weights):
    items, probs = zip(*weights)
    return random.choices(items, weights=probs, k=1)[0]

# ---------- kategoriler / işletmeler ----------
CATEGORIES = [
    ("Market", 24), ("Kafe", 12), ("Restoran", 14), ("Ulaşım", 10),
    ("Eczane", 6), ("Giyim", 8), ("Elektronik", 5),
    ("Online Alışveriş", 12), ("Fatura/Servis", 9),
]
MERCHANTS = {
    "Market": ["Migrosea","KarpuzEx","Şokup","A1010","BenimGross","CarrefourSA"],
    "Kafe": ["Kahve Kız","BeanBros","Taze Çekirdek","LatteLab"],
    "Restoran": ["Anadolu Sofrası","BurgerHan","BalıkTime","ÇiğköfteX"],
    "Ulaşım": ["İETT Dolum","MetroKart","TaksiApp"],
    "Eczane": ["Sağlık Eczanesi","Şifa Eczane","Derman+ "],
    "Giyim": ["ModaPark","Tekstilix","UrbanWear"],
    "Elektronik": ["TeknoCity","E-Market","Volt&Ohm","Teknosa"],
    "Online Alışveriş": ["Trendiol","HepsOrada","N11.5"],
    "Fatura/Servis": ["Elektrik A.Ş.","Su İdaresi","NetNet","GSM+","Brisa"],
}

# ---------- gelir profilleri ----------
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

# ---------- çekirdek simülasyon (genel) ----------
def simulate_group(group_name, days, start_date, daily_tx_target, seed=None):
    if seed is not None:
        np.random.seed(seed + hash(group_name)%10000); random.seed(seed + hash(group_name)%10000)
    prof = INCOME_PROFILES[group_name]
    rows = []
    for d in range(days):
        date = start_date + timedelta(days=d)
        n_tx = int(daily_tx_target)
        if n_tx <= 0: continue
        hours = np.random.randint(8, 23, size=n_tx)
        minutes = np.random.randint(0, 60, size=n_tx)

        for i in range(n_tx):
            cat = pick_weighted(CATEGORIES)
            merchant = random.choice(MERCHANTS[cat])
            amount = float(np.random.lognormal(mean=prof["lognorm_mean"], sigma=prof["lognorm_sd"]))
            amount *= CATEGORY_SCALE.get(cat,1.0) * prof["spend_mult"]
            amount = round(max(5.0, amount), 2)

            mini = contribution(amount, 5)
            midi = contribution(amount, 10)
            maxi = contribution(amount, 20)  # Maxi 20

            amt_int = int(round(amount))
            rows.append({
                "Grup": group_name,
                "Tarih": date.date().isoformat(),
                "Saat": f"{hours[i]:02d}:{minutes[i]:02d}",
                "Kategori": cat,
                "İşletme": merchant,
                "Tutar_TL": amount,
                "Mini_Katkı_TL": round(mini,2),
                "Midi_Katkı_TL": round(midi,2),
                "Maxi_Katkı_TL": round(maxi,2),
                "Mini_Yuvarlanan_Tutar": next_multiple(amt_int, 5),
                "Midi_Yuvarlanan_Tutar": next_multiple(amt_int, 10),
                "Maxi_Yuvarlanan_Tutar": next_multiple(amt_int, 20),
            })
    return pd.DataFrame(rows)

def summarize(df):
    if df.empty: return pd.DataFrame()
    g = df.groupby("Grup", as_index=False).agg(
        İşlem_Sayısı=("Tutar_TL","count"),
        Toplam_Harcama_TL=("Tutar_TL","sum"),
        Mini_Toplam_Katkı_TL=("Mini_Katkı_TL","sum"),
        Midi_Toplam_Katkı_TL=("Midi_Katkı_TL","sum"),
        Maxi_Toplam_Katkı_TL=("Maxi_Katkı_TL","sum"),
    )
    for col in ["Mini","Midi","Maxi"]:
        g[f"{col}_Etkin_Oran_%"] = (g[f"{col}_Toplam_Katkı_TL"]/g["Toplam_Harcama_TL"]*100).round(2)
    return g

def fmt_money(df):
    df = df.copy()
    for c in df.columns:
        if c.endswith("_TL"): df[c] = df[c].apply(tl)
    return df

# ---------- tek paket için aylık toplam simülasyonu (TÜM KATEGORİLER) ----------
PACKAGE_BASES = {"Mini (5)": 5, "Midi (10)": 10, "Maxi (20)": 20}

def simulate_month_total_one(mean_tx_per_day: float,
                             days_in_month: int = 30,
                             trials: int = 5000,   # 5000 deneme
                             seed: int | None = 123,
                             profile_name: str = "Orta Gelir",
                             package_label: str = "Midi (10)") -> pd.DataFrame:
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
            for _ in range(n_tx):
                cat = random.choices(cats, weights=probs, k=1)[0]
                amount = float(np.random.lognormal(mean=prof["lognorm_mean"], sigma=prof["lognorm_sd"])
                amount *= CATEGORY_SCALE.get(cat, 1.0) * prof["spend_mult"]
                amount = round(max(5.0, amount), 2)
                total += contribution(amount, base)
            tx_count += n_tx
        rows.append({"trial": t+1, "Toplam_Katki_TL": round(total, 2), "Toplam_Islem": tx_count})
    return pd.DataFrame(rows)

# ---------- UI: 2 sekme ----------
tab1, tab2 = st.tabs(["🧪 Simülatör", "📈 Ay Sonu Dağılımı"])

with tab1:
    st.title("🪙 BES Yuvarla-Ekle Simülatörü — Mini(5) / Midi(10) / Maxi(20)")
    with st.sidebar:
        st.header("Ayarlar")
        days = st.slider("Simülasyon süresi (gün)", 7, 60, 30)
        seed = st.number_input("Rastgelelik (seed)", min_value=0, value=42, step=1)
        start = st.date_input("Başlangıç tarihi", value=datetime.now().date() - timedelta(days=days))
        aktif_gruplar = st.multiselect("Gelir grupları", list(INCOME_PROFILES.keys()), default=list(INCOME_PROFILES.keys()))
        daily_tx_target = st.slider("Günlük işlem hedefi (her grup için)", 1, 5, 1, step=1)
        gosterim_limiti = st.slider("İşlem tablosu satır limiti (görünüm)", 20, 1000, 200, step=10)

    frames = [
        simulate_group(g, days, datetime.combine(start, datetime.min.time()), daily_tx_target, seed)
        for g in aktif_gruplar
    ]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_df = summarize(data)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("📊 Özet (Grup Bazında)")
        if not summary_df.empty:
            st.dataframe(summary_df.style.format({
                "Toplam_Harcama_TL": "{:,.2f}",
                "Mini_Toplam_Katkı_TL": "{:,.2f}",
                "Midi_Toplam_Katkı_TL": "{:,.2f}",
                "Maxi_Toplam_Katkı_TL": "{:,.2f}",
                "Mini_Etkin_Oran_%": "{:,.2f}",
                "Midi_Etkin_Oran_%": "{:,.2f}",
                "Maxi_Etkin_Oran_%": "{:,.2f}",
            }), use_container_width=True)
        else:
            st.info("Henüz veri yok.")
    with c2:
        if not summary_df.empty:
            st.metric("Toplam Harcama", tl(summary_df["Toplam_Harcama_TL"].sum()))
            st.metric("Mini Toplam Katkı", tl(summary_df["Mini_Toplam_Katkı_TL"].sum()))
            st.metric("Midi Toplam Katkı", tl(summary_df["Midi_Toplam_Katkı_TL"].sum()))
            st.metric("Maxi Toplam Katkı", tl(summary_df["Maxi_Toplam_Katkı_TL"].sum()))

    st.divider()
    st.subheader("🧾 İşlem Bazında Görünüm")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        grup_f = st.multiselect("Grup", sorted(data["Grup"].unique()) if not data.empty else [],
                                 default=sorted(data["Grup"].unique()) if not data.empty else [])
    with f2:
        kat_f = st.multiselect("Kategori", sorted(data["Kategori"].unique()) if not data.empty else [],
                               default=sorted(data["Kategori"].unique()) if not data.empty else [])
    with f3:
        paket = st.selectbox("Paket", ["Mini (5)", "Midi (10)", "Maxi (20)"])
    with f4:
        min_katkı = st.number_input("Katkı (TL) min filtresi", min_value=0.0, value=0.0, step=1.0)

    df_f = data.copy()
    if not df_f.empty:
        if grup_f: df_f = df_f[df_f["Grup"].isin(grup_f)]
        if kat_f:  df_f = df_f[df_f["Kategori"].isin(kat_f)]

        if paket.startswith("Mini"):
            df_f = df_f[df_f["Mini_Katkı_TL"] >= min_katkı]
            show = ["Grup","Tarih","Saat","Kategori","İşletme","Tutar_TL","Mini_Katkı_TL","Mini_Yuvarlanan_Tutar"]
        elif paket.startswith("Midi"):
            df_f = df_f[df_f["Midi_Katkı_TL"] >= min_katkı]
            show = ["Grup","Tarih","Saat","Kategori","İşletme","Tutar_TL","Midi_Katkı_TL","Midi_Yuvarlanan_Tutar"]
        else:
            df_f = df_f[df_f["Maxi_Katkı_TL"] >= min_katkı]
            show = ["Grup","Tarih","Saat","Kategori","İşletme","Tutar_TL","Maxi_Katkı_TL","Maxi_Yuvarlanan_Tutar"]

        st.dataframe(fmt_money(df_f[show]).head(gosterim_limiti), use_container_width=True)
    else:
        st.info("Veri yok. Soldan ayarları kontrol et.")

    st.subheader("🧮 Kategori Bazında Katkı Özeti")
    if not df_f.empty:
        cat_sum = df_f.groupby(["Grup","Kategori"], as_index=False).agg(
            İşlem_Sayısı=("Tutar_TL","count"),
            Toplam_Harcama_TL=("Tutar_TL","sum"),
            Mini_Toplam_Katkı_TL=("Mini_Katkı_TL","sum"),
            Midi_Toplam_Katkı_TL=("Midi_Katkı_TL","sum"),
            Maxi_Toplam_Katkı_TL=("Maxi_Katkı_TL","sum"),
        )
        st.dataframe(cat_sum.style.format({
            "Toplam_Harcama_TL": "{:,.2f}",
            "Mini_Toplam_Katkı_TL": "{:,.2f}",
            "Midi_Toplam_Katkı_TL": "{:,.2f}",
            "Maxi_Toplam_Katkı_TL": "{:,.2f}",
        }), use_container_width=True)

    st.divider()
    st.subheader("⬇️ Dışa Aktar")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Tüm İşlemler (CSV)", data=data.to_csv(index=False).encode("utf-8"),
                           file_name="islemler_simulasyon.csv", mime="text/csv")
    with c2:
        st.download_button("Özet (CSV)", data=summary_df.to_csv(index=False).encode("utf-8"),
                           file_name="ozet_simulasyon.csv", mime="text/csv")

    st.caption("Not: Katkı hesabında tutar önce tam TL'ye yuvarlanır. Paket bazları: Mini=5 TL, Midi=10 TL, Maxi=20 TL.")

with tab2:
    st.title("📈 Ay Sonu Dağılımı (Tüm Kategoriler)")

    # seçimler
    c1, c2 = st.columns([1,1])
    with c1:
        package_label = st.selectbox("Paket", ["Mini (5)", "Midi (10)", "Maxi (20)"], index=1)
    with c2:
        mean_tx = st.select_slider(
            "Günlük işlem adedi",
            options=[1, 2, 3, 4, 5],
            value=2,
            help="Günlük ortalama toplam işlem adedi (λ)."
        )

    # Tanıtım varsayılanları 
    DAYS = 30
    TRIALS = 5000
    SEED = 123
    PROFILE = "Orta Gelir"

    df_month_one = simulate_month_total_one(
        mean_tx_per_day=float(mean_tx),
        days_in_month=DAYS,
        trials=TRIALS,
        seed=SEED,
        profile_name=PROFILE,
        package_label=package_label
    )

    st.markdown(f"**Profil:** {PROFILE} • **Ay:** {DAYS} gün • **Deneme:** {TRIALS}")

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

    # Özet metrikler
    st.subheader("📌 Özet (Ay Sonu)")
    s = df_month_one["Toplam_Katki_TL"].describe()
    p5  = float(df_month_one["Toplam_Katki_TL"].quantile(0.05))
    p95 = float(df_month_one["Toplam_Katki_TL"].quantile(0.95))

    m1, m2, m3 = st.columns(3)
    m1.metric("Medyan", tl(float(df_month_one["Toplam_Katki_TL"].median())))
    m2.metric("Ortalama", tl(float(s["mean"])))
    m3.metric("P5–P95", f"{tl(p5)} — {tl(p95)}")

    st.markdown("**İşlem Sayısı (ay)**")
    st.table(df_month_one["Toplam_Islem"].describe().to_frame(name="Toplam İşlem").T.round(2))

    st.divider()
    st.download_button(
        "Ay Sonu Toplam Katkılar (CSV)",
        data=df_month_one.to_csv(index=False).encode("utf-8"),
        file_name=f"ay_sonu_toplam_{package_label.replace(' ','').replace('(','').replace(')','')}_tum_kategoriler.csv",
        mime="text/csv"
    )

    st.caption("Not: Tutar önce tam TL'ye yuvarlanır; paket bazları: Mini=5 TL, Midi=10 TL, Maxi=20 TL.")
