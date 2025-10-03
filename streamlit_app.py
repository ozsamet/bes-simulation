# bessim.py (Ay Sonu Toplam Katkı Dağılımı sekmesi; bonus yok, Maxi 25->20)
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import altair as alt

st.set_page_config(page_title="BES Yuvarla-Ekle Simülatörü (Mini/Midi/Maxi)", layout="wide")

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
            maxi = contribution(amount, 20)

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

# ---------- yeni: Ay sonu toplam katkı dağılımı (Market) ----------
def simulate_month_totals_market(profile_name: str,
                                 days_in_month: int = 30,
                                 mean_tx_per_day: float = 1.5,
                                 trials: int = 1000,
                                 seed: int | None = None) -> pd.DataFrame:
    """
    Sadece Market kategorisi: her trial bir 'ay'.
    Her günün işlem adedi ~ Poisson(mean_tx_per_day).
    Dönüş: her trial için Mini/Midi/Maxi toplam katkılar ve toplam işlem sayısı.
    """
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    prof = INCOME_PROFILES[profile_name]

    totals = []
    for t in range(trials):
        mini_sum = midi_sum = maxi_sum = 0.0
        tx_count = 0
        for _ in range(days_in_month):
            n_tx = np.random.poisson(mean_tx_per_day)
            if n_tx <= 0:
                continue
            for _ in range(n_tx):
                amount = float(np.random.lognormal(mean=prof["lognorm_mean"], sigma=prof["lognorm_sd"]))
                amount *= CATEGORY_SCALE["Market"] * prof["spend_mult"]
                amount = round(max(5.0, amount), 2)

                mini_sum += contribution(amount, 5)
                midi_sum += contribution(amount, 10)
                maxi_sum += contribution(amount, 20)
            tx_count += n_tx
        totals.append({
            "trial": t+1,
            "Mini_Toplam_TL": round(mini_sum, 2),
            "Midi_Toplam_TL": round(midi_sum, 2),
            "Maxi_Toplam_TL": round(maxi_sum, 2),
            "Toplam_Islem": tx_count
        })
    return pd.DataFrame(totals)

# ---------- UI: 2 sekme ----------
tab1, tab2 = st.tabs(["🧪 Simülatör", "📈 Ay Sonu Dağılımı (Market)"])

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
    st.title("📈 Ay Sonu Toplam Katkı Dağılımı — Market")
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        prof_name = st.selectbox("Gelir Profili", list(INCOME_PROFILES.keys()), index=1)  # Orta Gelir
    with col2:
        days_in_month = st.number_input("Ay (gün)", min_value=20, max_value=35, value=30, step=1)
    with col3:
        mean_tx = st.number_input("Günlük ort. Market işlemi (λ)", min_value=0.0, value=1.5, step=0.1)
    with col4:
        trials = st.number_input("Deneme sayısı (trial)", min_value=200, max_value=20000, value=1000, step=100)

    seed2 = st.number_input("Seed (dağılım)", min_value=0, value=123, step=1)

    df_month = simulate_month_totals_market(
        profile_name=prof_name,
        days_in_month=int(days_in_month),
        mean_tx_per_day=float(mean_tx),
        trials=int(trials),
        seed=int(seed2)
    )

    st.markdown("**Ay sonu toplam katkı (TL) dağılımları**")
    # Uzun formata çevir
    plot_df = df_month.melt(id_vars=["trial","Toplam_Islem"],
                            value_vars=["Mini_Toplam_TL","Midi_Toplam_TL","Maxi_Toplam_TL"],
                            var_name="Paket", value_name="Toplam_Katki_TL")
    paket_map = {"Mini_Toplam_TL":"Mini (5)", "Midi_Toplam_TL":"Midi (10)", "Maxi_Toplam_TL":"Maxi (20)"}
    plot_df["Paket"] = plot_df["Paket"].map(paket_map)

    # Histogram + yoğunluk eğrisi (Altair)
    bins = st.slider("Histogram kutu sayısı", 10, 80, 30, step=5)
    base = alt.Chart(plot_df)

    hist = base.mark_bar(opacity=0.5).encode(
        x=alt.X("Toplam_Katki_TL:Q", bin=alt.Bin(maxbins=bins), title="Ay Sonu Toplam Katkı (TL)"),
        y=alt.Y("count():Q", title="Adet"),
        tooltip=["Paket:N","count():Q"]
    ).properties(height=260)

    density = base.transform_density(
        "Toplam_Katki_TL",
        groupby=["Paket"],
        as_=["Toplam_Katki_TL","Yoğunluk"]
    ).mark_line().encode(
        x="Toplam_Katki_TL:Q",
        y="Yoğunluk:Q",
        tooltip=["Paket:N","Toplam_Katki_TL:Q","Yoğunluk:Q"]
    )

    st.altair_chart((hist.encode(color="Paket:N") + density.encode(color="Paket:N")).resolve_scale(y='independent'), use_container_width=True)

    # İstatistikler
    st.subheader("📌 Özet İstatistikler (Ay Sonu)")
    stats = plot_df.groupby("Paket")["Toplam_Katki_TL"].agg(
        Ortalama="mean",
        Medyan="median",
        P5=lambda s: s.quantile(0.05),
        P25=lambda s: s.quantile(0.25),
        P75=lambda s: s.quantile(0.75),
        P95=lambda s: s.quantile(0.95),
        Max="max",
        Min="min"
    ).round(2).reset_index()

    # İşlem sayısı dağılımı da bilgi amaçlı
    ops = df_month["Toplam_Islem"].describe().to_frame(name="Toplam İşlem").T.round(2)

    cA, cB = st.columns([2,1])
    with cA:
        st.dataframe(stats, use_container_width=True)
    with cB:
        st.markdown("**İşlem Sayısı (ay)**")
        st.table(ops)

    st.divider()
    st.download_button(
        "Ay Sonu Toplam Katkılar (CSV)",
        data=df_month.to_csv(index=False).encode("utf-8"),
        file_name="ay_sonu_toplam_katki_dagilimi_market.csv",
        mime="text/csv"
    )
