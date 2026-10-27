"""
Live demo: soil nutrient prediction for Kolhapur and Sangli (our own app, not upstream).

Run:  .venv/bin/streamlit run demo_app.py
Standalone: reads data/demo_data.csv and models/.
"""
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

NUTRIENTS = {"N": ("Nitrogen", "kg/ha"), "P": ("Phosphorus", "kg/ha"),
             "K": ("Potassium", "kg/ha"), "OC": ("Organic carbon", "%")}
CUTS = {"N": (280, 560), "P": (10, 25), "K": (120, 280), "OC": (0.50, 0.75)}
RGB = {"Low": [215, 48, 39], "Medium": [253, 174, 97], "High": [26, 152, 80]}
FEATURES = ['temp', 'SI', 'RI', 'precipitation', 'sand05', 'silt05', 'sand515', 'TGSI',
            'NDVI_Kharif', 'NDVI_Rabi', 'NDVI_Zaid', 'pH_0-5', 'pH_5-15',
            'NIRv_Kharif', 'NIRv_Rabi', 'NIRv_Zaid']
# Features that make sense to adjust live, with friendly labels
WHATIF = {"NDVI_Kharif": "Crop greenness, monsoon season (NDVI)",
          "NDVI_Rabi": "Crop greenness, winter season (NDVI)",
          "pH_0-5": "Soil pH, top 5 cm",
          "sand05": "Sand content, top 5 cm (g/kg)",
          "temp": "Temperature (°C)"}

st.set_page_config(page_title="Soil Health Demo: Kolhapur & Sangli", layout="wide")


def status(n, v):
    lo, hi = CUTS[n]
    return "Low" if v < lo else ("Medium" if v < hi else "High")


@st.cache_data
def load():
    return pd.read_csv("data/demo_data.csv").assign(village=lambda x: x.village.astype(str))


@st.cache_resource
def models():
    return {n: joblib.load(f"models/rf_KS_{n}.joblib") for n in NUTRIENTS}


df, mdl = load(), models()

st.title("🌱 Soil Health Prediction: Kolhapur & Sangli")
st.caption("Predicts N, P, K and organic carbon from satellite-derived features. "
           "Lab values come from the Soil Health Card programme (2023–24).")

with st.sidebar:
    st.header("Pick a location")
    dist = st.selectbox("District", sorted(df.district.unique()), format_func=str.title)
    vills = sorted(df[df.district == dist].village.unique())
    vill = st.selectbox(f"Village ({len(vills)})", vills)
    rows = df[(df.district == dist) & (df.village == vill)]
    pick = rows.iloc[0] if len(rows) == 1 else rows.iloc[
        st.number_input(f"{len(rows)} samples here: choose one", 1, len(rows), 1) - 1]
    st.info("Predictions in the first tab come from models that never saw this village. "
            "That is the honest accuracy: only phosphorus is predicted reliably.")

tab1, tab2, tab3 = st.tabs(["📍 Predict for a village", "🎚️ What-if sliders", "🗺️ District map"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader(f"{vill.title()}, {dist.title()}")
    cols = st.columns(4)
    for c, (n, (name, unit)) in zip(cols, NUTRIENTS.items()):
        lab, pred = pick[n], pick[f"pred_{n}"]
        with c:
            st.metric(f"{name} ({unit})", f"{pred:.2f}" if n == "OC" else f"{pred:.0f}",
                      f"lab: {lab:.2f}" if n == "OC" else f"lab: {lab:.0f}", delta_color="off")
            same = status(n, lab) == status(n, pred)
            st.markdown(f"Predicted **{status(n, pred)}**, lab **{status(n, lab)}** "
                        f"{'✅' if same else '❌'}")
    ok = sum(status(n, pick[n]) == status(n, pick[f"pred_{n}"]) for n in NUTRIENTS)
    st.caption(f"{ok} of 4 nutrient classes match the lab result for this sample. "
               "Individual fields can be off; the model is validated on all 2,078 samples, not one.")

    d2 = df.copy()
    d2["color"] = [[120, 120, 120, 90]] * len(d2)
    sel = pd.DataFrame([{"longitude": pick.longitude, "latitude": pick.latitude}])
    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=pick.latitude, longitude=pick.longitude, zoom=9),
        layers=[pdk.Layer("ScatterplotLayer", d2, get_position="[longitude, latitude]",
                          get_fill_color="color", get_radius=900),
                pdk.Layer("ScatterplotLayer", sel, get_position="[longitude, latitude]",
                          get_fill_color=[220, 30, 30, 255], get_radius=2500)]))

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Change the conditions, watch the prediction move")
    st.caption("Starts from the selected sample. Sliders alter satellite/soil inputs; "
               "the model re-predicts instantly. Relationships are statistical, not causal advice.")
    x = pick[FEATURES].astype(float).copy()
    left, right = st.columns([1, 1])
    with left:
        for f, label in WHATIF.items():
            lo, hi = float(df[f].quantile(0.02)), float(df[f].quantile(0.98))
            x[f] = st.slider(label, lo, hi, float(np.clip(x[f], lo, hi)))
    base = pd.DataFrame([pick[FEATURES].astype(float)])
    new = pd.DataFrame([x])
    with right:
        for n, (name, unit) in NUTRIENTS.items():
            b, p = mdl[n].predict(base)[0], mdl[n].predict(new)[0]
            st.metric(f"{name} ({unit})", f"{p:.2f}" if n == "OC" else f"{p:.0f}",
                      f"{p - b:+.2f}" if n == "OC" else f"{p - b:+.0f}", delta_color="off")
            st.markdown(f"Status: **{status(n, p)}**")
    st.caption("Models here were trained on a random 80% of Kolhapur+Sangli samples.")

# ---------------- Tab 3 ----------------
with tab3:
    c1, c2 = st.columns(2)
    n = c1.radio("Nutrient", list(NUTRIENTS), horizontal=True,
                 format_func=lambda k: NUTRIENTS[k][0])
    src = c2.radio("Show", ["Lab value", "Model prediction"], horizontal=True)
    col = n if src == "Lab value" else f"pred_{n}"
    m = df.copy()
    m["color"] = m[col].map(lambda v: RGB[status(n, v)] + [200])
    m["label"] = m[col].round(2)
    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=df.latitude.mean(), longitude=df.longitude.mean(), zoom=7.8),
        layers=[pdk.Layer("ScatterplotLayer", m, get_position="[longitude, latitude]",
                          get_fill_color="color", get_radius=1100, pickable=True)],
        tooltip={"text": "{village}\n{label}"}))
    share = m[col].map(lambda v: status(n, v)).value_counts(normalize=True)
    st.write("🔴 Low · 🟠 Medium · 🟢 High   |   " +
             "  ·  ".join(f"{k}: {share.get(k, 0):.0%}" for k in ["Low", "Medium", "High"]))
    mp = df[n].map(lambda v: status(n, v)).value_counts(normalize=True).max()
    ag = (df[n].map(lambda v: status(n, v)) == df[f"pred_{n}"].map(lambda v: status(n, v))).mean()
    st.caption(f"Class agreement with lab: {ag:.0%} (always guessing the most common class: {mp:.0%}).")
