import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from datetime import datetime, timedelta

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
    r_m = (annual_return_pct / 100.0) / 12.0
    n = years * 12
    if abs(r_m) < 1e-12:
        return monthly_amount * n
    return monthly_amount * (((1 + r_m) ** n - 1) / r_m)

# ---------- örnek işlem üretici (paket mantığı tabı) ----------
@st.cache_data(show_spinner=False)
def generate_sample_transactions(n_ops: int,
                                 categories: list,
                                 seed: int = SEED,
                                 span_days: int = 1) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed + 7)
    now = datetime.now()
    rows = []
    cats = categories if categories else ["Market"]

    for _ in range(n_ops):
        cat = random.choice(cats)
        amount = float(np.random.lognormal(mean=3.6, sigma=0.5))
        amount *= CATEGORY_SCALE.get(cat, 1.0) * 1.15
        amount = round(max(5.0, amount), 2)

        dt = now - timedelta(days=random.randint(0, max(0, span_days-1)),
                             hours=random.randint(0, 23),
                             minutes=random.randint(0, 59))
        rows.append({
            "Zaman": dt.replace(second=0, microsecond=0),
            "Kategori": cat,
            "Tutar": amount
        })

    df = pd.DataFrame(rows).sort_values("Zaman").reset_index(drop=True)
    df["5'lik"]  = df["Tutar"].apply(lambda x: contribution(x, 5))
    df["10'luk"] = df["Tutar"].apply(lambda x: contribution(x, 10))
    df["20'lik"] = df["Tutar"].apply(lambda x: contribution(x, 20))
    return df

# ===================== TABS =====================
tab_sim, tab_mech = st.tabs(["📈 Simülasyon", "🛒 Paket Mantığı (Market Örneği)"])

# ============ 📈 SİMÜLASYON TAB ============
with tab_sim:
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

    # ---------- BES PROJEKSİYONU ----------
    st.subheader("💰 BES Projeksiyonu")

    colA, colB = st.columns([1,1])
    with colA:
        years_in_system = st.slider("Sistemde Kalınacak Süre (Yıl)", 5, 40, 20, 1)
    with colB:
        expected_return = st.slider("Reel Beklenen Yıllık Getiri (%)", 0.0, 10.0, 4.0, 1.0)

    monthly_typical = median_v
    balance_fv      = fv_of_monthly(monthly_typical, expected_return, years_in_system)

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

    total_principal = monthly_typical * 12 * years_in_system
    gain_component  = max(0.0, balance_fv - total_principal)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tipik Aylık Katkı", tl(monthly_typical))
    c2.metric("Toplam Katkı (Ana Para)", tl(total_principal))
    c3.metric("Getiri Kazancı", tl(gain_component))

    st.markdown(
        f"**Özet:** {years_in_system} yıl boyunca aylık ~{tl(monthly_typical)} katkı ve yıllık %{expected_return:.1f} getiri varsayımıyla "
        f"emeklilik başlangıcında yaklaşık **{tl(balance_fv)}** birikim oluşur."
    )

# ============ 🛒 PAKET MANTIĞI TAB ============
with tab_mech:
    st.title("🛒 Paketlerin Çalışma Mantığı — Market Örneği")
    st.caption("Her işlem tutarı, seçilen paketin *üst çokluğuna* yuvarlanır. Fark → **BES’e aktarılır**.")

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        n_ops = st.slider("Örnek İşlem Adedi", 5, 40, 12, 1)
    with colB:
        span = st.selectbox("Zaman Aralığı", ["Bugün", "Son 7 Gün"], index=0)
        span_days = 1 if span == "Bugün" else 7
    with colC:
        selectable_cats = [c for c,_ in CATEGORIES]
        chosen = st.multiselect("Kategoriler", options=selectable_cats, default=["Market"])

    df_tx = generate_sample_transactions(n_ops, chosen, span_days=span_days)

    tot5  = float(df_tx["5'lik"].sum())
    tot10 = float(df_tx["10'luk"].sum())
    tot20 = float(df_tx["20'lik"].sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Toplam Katkı (5’lik)", tl(tot5))
    k2.metric("Toplam Katkı (10’luk)", tl(tot10))
    k3.metric("Toplam Katkı (20’lik)", tl(tot20))

    st.markdown("### İşlem Listesi (örnek)")
    show_df = df_tx.copy()
    show_df["Zaman"] = show_df["Zaman"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        show_df[["Zaman", "Kategori", "Tutar", "5'lik", "10'luk", "20'lik"]],
        use_container_width=True
    )

    sum_df = pd.DataFrame({
        "Paket": ["5'lik", "10'luk", "20'lik"],
        "Toplam_Katki_TL": [tot5, tot10, tot20]
    })
    bar = alt.Chart(sum_df).mark_bar().encode(
        x=alt.X("Paket:N", title="Paket"),
        y=alt.Y("Toplam_Katki_TL:Q", title="Toplam Katkı (TL)"),
        tooltip=[alt.Tooltip("Paket:N"), alt.Tooltip("Toplam_Katki_TL:Q", format=".2f")]
    ).properties(height=260, title="Paketlere Göre Toplam BES Katkısı")
    st.altair_chart(bar, use_container_width=True)

    with st.expander("Mantık (kısaca)"):
        st.markdown(
            "- Her **işlem tutarı** (ör. Market sepeti) için seçilen paketin **üst çokluğuna** yuvarlama yapılır.\n"
            "- **Fark = Yuvarlanan tutar − işlem tutarı** → **BES’e aktarılır**.\n"
            "- Örn: 73 TL → 5’likte 75’e yuvarlanır → fark **2 TL**; 10’lukta 80 → **7 TL**; 20’likte 80 → **7 TL**."
        )
