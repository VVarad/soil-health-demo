"""
Micronutrient Estimator: Kolhapur and Sangli.

Estimates sulphur, zinc and iron from a basic soil test (N, P, K, OC, pH, EC)
plus satellite data. Data and models are produced by train.py.

Run:  streamlit run app.py
"""
import json

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

from train import BASIC_TEST, FEATURES, TARGETS

NAMES = {"S": "Sulphur", "Zn": "Zinc", "Fe": "Iron"}
BASIC_LABELS = {"N": ("Nitrogen", "kg/ha"), "P": ("Phosphorus", "kg/ha"),
                "K": ("Potassium", "kg/ha"), "OC": ("Organic carbon", "%"),
                "pH": ("pH", ""), "EC": ("Electrical conductivity", "dS/m")}
DEFICIENT_RGB, SUFFICIENT_RGB = [217, 119, 43], [43, 108, 176]

st.set_page_config(page_title="Micronutrient Estimator", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
h1 {font-weight: 700; letter-spacing: -0.5px;}
.card {background: #FFFFFF; border: 1px solid #E1E4DD; border-radius: 12px;
       padding: 18px 20px; height: 100%;}
.card-title {font-size: 0.85rem; font-weight: 600; color: #5B6660;
             text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;}
.big {font-size: 2.1rem; font-weight: 700; color: #1F2A24; line-height: 1.15;}
.unit {font-size: 1rem; font-weight: 500; color: #5B6660;}
.row {display: flex; justify-content: space-between; align-items: center;
      margin-top: 10px; font-size: 0.95rem; color: #3A4540;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px;
       font-size: 0.8rem; font-weight: 600;}
.def {background: #FBE6D3; color: #8A4510;}
.ok {background: #DCE9F5; color: #1E4C7A;}
.note {font-size: 0.85rem; color: #5B6660; margin-top: 12px;
       border-top: 1px solid #ECEEE8; padding-top: 8px;}
.kv {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;}
.kv div span {display: block; font-size: 0.8rem; color: #5B6660;}
.kv div b {font-size: 1.15rem; color: #1F2A24;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_fields():
    return pd.read_csv("data/fields.csv")


@st.cache_data
def load_metrics():
    with open("data/metrics.json") as f:
        return json.load(f)


@st.cache_resource
def load_models():
    return {t: joblib.load(f"models/{t}.joblib") for t in TARGETS}


def status(t, v):
    return "Deficient" if v < TARGETS[t] else "Sufficient"


def pill(t, v):
    s = status(t, v)
    return f'<span class="pill {"def" if s == "Deficient" else "ok"}">{s}</span>'


def result_card(t, pred, lab=None):
    html = (f'<div class="card"><div class="card-title">{NAMES[t]} ({t})</div>'
            f'<div class="big">{pred:.2f} <span class="unit">mg/kg</span></div>'
            f'<div class="row"><span>Estimated status</span>{pill(t, pred)}</div>')
    if lab is not None:
        agree = status(t, pred) == status(t, lab)
        html += (f'<div class="row"><span>Lab measured: <b>{lab:.2f}</b> mg/kg</span>{pill(t, lab)}</div>'
                 f'<div class="note">{"Status agrees with the lab" if agree else "Status differs from the lab"}'
                 f' &middot; deficient below {TARGETS[t]:g} mg/kg</div>')
    else:
        html += f'<div class="note">Deficient below {TARGETS[t]:g} mg/kg</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def fmt(k, v):
    return f"{v:.2f}" if k in ("OC", "pH", "EC") else f"{v:.0f}"


df, metrics, models = load_fields(), load_metrics(), load_models()

# ---------------- header & field picker ----------------
st.title("Micronutrient Estimator")
st.markdown("Estimates **sulphur, zinc and iron** for farm fields in **Kolhapur and Sangli** "
            "from a basic soil test and satellite data. Lab values: Soil Health Card programme, 2023-24.")

with st.sidebar:
    st.subheader("Select a field")
    district = st.selectbox("District", sorted(df.district.unique()))
    villages = sorted(df[df.district == district].village.unique())
    village = st.selectbox(f"Village ({len(villages)} available)", villages)
    rows = df[(df.district == district) & (df.village == village)].reset_index(drop=True)
    i = 0
    if len(rows) > 1:
        i = st.selectbox("Field", range(len(rows)), format_func=lambda k: f"Field {k + 1} of {len(rows)}")
    field = rows.iloc[i]
    st.divider()
    st.caption(f"{metrics['fields']:,} lab-tested fields across {metrics['villages']} villages.")

tab_check, tab_custom, tab_acc = st.tabs(["Field check", "Try your own values", "Model accuracy"])

# ---------------- Field check ----------------
with tab_check:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader(f"{village}, {district}")
        kv = "".join(f"<div><span>{BASIC_LABELS[k][0]} {('(' + BASIC_LABELS[k][1] + ')') if BASIC_LABELS[k][1] else ''}</span>"
                     f"<b>{fmt(k, field[k])}</b></div>" for k in BASIC_TEST)
        st.markdown(f'<div class="card"><div class="card-title">Inputs: basic soil test</div>'
                    f'<div class="kv">{kv}</div><div class="note">Satellite data for this location '
                    f'(crop greenness, soil texture, temperature, rainfall) is added automatically.</div></div>',
                    unsafe_allow_html=True)
    with right:
        st.pydeck_chart(pdk.Deck(
            map_style="light",
            initial_view_state=pdk.ViewState(latitude=field.latitude, longitude=field.longitude, zoom=8.5),
            layers=[pdk.Layer("ScatterplotLayer", df, get_position="[longitude, latitude]",
                              get_fill_color=[150, 160, 150, 110], get_radius=700),
                    pdk.Layer("ScatterplotLayer", pd.DataFrame([field]), get_position="[longitude, latitude]",
                              get_fill_color=[47, 107, 74, 255], get_line_color=[255, 255, 255],
                              stroked=True, line_width_min_pixels=2, get_radius=2200)]),
            height=260)

    st.subheader("Estimated micronutrients")
    for col, t in zip(st.columns(3, gap="medium"), TARGETS):
        with col:
            result_card(t, field[f"pred_{t}"], field[t])
    st.caption("These estimates come from a model that never saw this village during training, "
               "so the comparison with the lab value is a fair test.")

# ---------------- Try your own values ----------------
with tab_custom:
    st.subheader("Enter basic soil test values")
    st.caption(f"Satellite data is taken from the selected location: {village}, {district}. "
               "Change any value to see the estimate update.")
    x = field[FEATURES].astype(float).copy()
    cols = st.columns(3, gap="medium")
    steps = {"N": 10.0, "P": 5.0, "K": 10.0, "OC": 0.05, "pH": 0.1, "EC": 0.05}
    for j, k in enumerate(BASIC_TEST):
        name, unit = BASIC_LABELS[k]
        lo, hi = float(df[k].min()), float(df[k].max())
        x[k] = cols[j % 3].number_input(f"{name} {('(' + unit + ')') if unit else ''}", lo, hi,
                                        float(min(max(field[k], lo), hi)), step=steps[k], key=f"in_{k}")
    st.subheader("Estimated micronutrients")
    X = pd.DataFrame([x])[FEATURES]
    for col, t in zip(st.columns(3, gap="medium"), TARGETS):
        with col:
            result_card(t, float(models[t].predict(X)[0]))
    st.caption("Values outside the range seen in Kolhapur and Sangli are not allowed, "
               "because the model cannot be trusted there.")

# ---------------- Model accuracy ----------------
with tab_acc:
    st.subheader("How accurate is it?")
    st.markdown("Tested on all fields, each predicted by a model that **never saw its village**. "
                "Status means Deficient or Sufficient.")
    m = metrics["targets"]
    table = pd.DataFrame({
        "Nutrient": [f"{NAMES[t]} ({t})" for t in TARGETS],
        "Status correct": [f"{m[t]['status_accuracy']:.0%}" for t in TARGETS],
        "Guessing the common answer": [f"{m[t]['baseline_accuracy']:.0%}" for t in TARGETS],
        "R² with basic test + satellite": [f"{m[t]['r2']:.2f}" for t in TARGETS],
        "R² satellite only": [f"{m[t]['r2_satellite_only']:.2f}" for t in TARGETS],
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.markdown("Zinc and sulphur are clearly better than guessing; iron is weaker. "
                "Adding the basic soil test improves every nutrient compared with satellite data alone.")

    st.subheader("Map of deficiency")
    c1, c2 = st.columns(2)
    t = c1.radio("Nutrient", list(TARGETS), format_func=lambda k: NAMES[k], horizontal=True)
    src = c2.radio("Show", ["Lab measured", "Estimated"], horizontal=True)
    col = t if src == "Lab measured" else f"pred_{t}"
    mp = df[["village", "latitude", "longitude", col]].copy()
    mp["color"] = mp[col].map(lambda v: (DEFICIENT_RGB if v < TARGETS[t] else SUFFICIENT_RGB) + [190])
    mp["value"] = mp[col].round(2)
    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=df.latitude.mean(), longitude=df.longitude.mean(), zoom=7.8),
        layers=[pdk.Layer("ScatterplotLayer", mp, get_position="[longitude, latitude]",
                          get_fill_color="color", get_radius=1000, pickable=True)],
        tooltip={"text": "{village}\n{value} mg/kg"}), height=430)
    share = (mp[col] < TARGETS[t]).mean()
    st.markdown(f'<span class="pill def">Deficient</span> {share:.0%} of fields &nbsp;&nbsp; '
                f'<span class="pill ok">Sufficient</span> {1 - share:.0%} of fields',
                unsafe_allow_html=True)
