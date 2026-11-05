"""
Micronutrient Estimator: Kolhapur and Sangli.

Estimates sulphur, zinc and iron from a basic soil test (N, P, K, OC, pH, EC)
plus satellite data. Data and models are produced by train.py.

Run:  streamlit run app.py
"""
import json

from pathlib import Path

from catboost import CatBoostClassifier
import pandas as pd
import pydeck as pdk
import streamlit as st

from classifier_study import catboost_path
from train import BASIC_TEST, FEATURES, TARGETS

ROOT = Path(__file__).resolve().parent

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
.hero {display: inline-block; padding: 6px 20px; border-radius: 999px;
       font-size: 1.45rem; font-weight: 700; margin: 4px 0 6px;}
.small {font-size: 0.9rem; color: #5B6660;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_fields():
    return pd.read_csv(ROOT / "data/fields.csv")


@st.cache_data
def load_metrics():
    with open(ROOT / "data/metrics.json") as f:
        return json.load(f)


@st.cache_resource
def load_models(fold):
    models = {}
    for target in TARGETS:
        model = CatBoostClassifier()
        model.load_model(str(catboost_path(ROOT, fold, target)))
        models[target] = model
    return models


def status(t, v):
    return "Deficient" if v < TARGETS[t] else "Sufficient"


REPORTED_CATBOOST_ACCURACY = {"S": 0.8643, "Zn": 0.8139, "Fe": 0.7620}


def accuracy_label(t):
    return f"CatBoost accuracy: {REPORTED_CATBOOST_ACCURACY[t]:.2%}"


def result_card(t, classification, lab=None):
    # The status is the headline, the numbers are secondary
    html = (f'<div class="card"><div class="card-title">{NAMES[t]} ({t})</div>'
            f'<div class="hero {"def" if classification == "Deficient" else "ok"}">{classification}</div>'
            f'<div class="small">{accuracy_label(t)}</div>')
    if lab is not None:
        lab_status = status(t, lab)
        html += (f'<div class="row"><span class="small">Lab measured: {lab:.2f} mg/kg</span>'
                 f'<span class="pill {"def" if lab_status == "Deficient" else "ok"}">{lab_status}</span></div>')
    html += f'<div class="note">Deficient below {TARGETS[t]:g} mg/kg</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def fmt(k, v):
    return f"{v:.2f}" if k in ("OC", "pH", "EC") else f"{v:.0f}"


df, metrics = load_fields(), load_metrics()

# ---------------- header & field picker ----------------
st.title("Micronutrient Estimator")

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

    st.subheader("Lab-measured micronutrients")
    for col, t in zip(st.columns(3, gap="medium"), TARGETS):
        with col:
            result_card(t, status(t, field[t]), lab=field[t])

# ---------------- Try your own values ----------------
with tab_custom:
    st.subheader("Enter basic soil test values")
    st.caption(f"Satellite data is taken from the selected location: {village}, {district}. "
               "Change any value to see the estimate update.")
    field_key = (district, village, i)
    if st.session_state.get("selected_field") != field_key:
        for key in BASIC_TEST:
            st.session_state.pop(f"in_{key}", None)
        st.session_state["selected_field"] = field_key
    models = load_models(int(field["village_fold"]))
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
            score = models[t].predict_proba(X, thread_count=1)[0, 1]
            result_card(t, "Deficient" if score > 0.5 else "Sufficient")

# ---------------- Model accuracy ----------------
with tab_acc:
    st.subheader("How accurate is it?")
    m = metrics["targets"]
    methods = {
        "RF Regressor + cutoff": "status_accuracy",
        "Tuned RF Classifier": "tuned_rf_accuracy",
    }
    table = pd.DataFrame([
        {"Method": name, **{NAMES[t]: f"{m[t][key]:.2%}" for t in TARGETS}}
        for name, key in methods.items()
    ])
    table.loc[len(table)] = {"Method": "CatBoost Classifier",
                             **{NAMES[t]: f"{REPORTED_CATBOOST_ACCURACY[t]:.2%}" for t in TARGETS}}
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("RF results use five village-held-out folds. CatBoost figures are team-reported. "
               "Field check uses lab readings. Custom predictions and the estimated map use CatBoost.")
    with st.expander("Rule-based baseline and RF regression scores"):
        diagnostics = pd.DataFrame({
            "Nutrient": [NAMES[t] for t in TARGETS],
            "Majority-class accuracy": [f"{m[t]['baseline_accuracy']:.1%}" for t in TARGETS],
            "RF R² with basic test + satellite": [m[t]["r2"] for t in TARGETS],
            "RF R² satellite only": [m[t]["r2_satellite_only"] for t in TARGETS],
        })
        st.dataframe(diagnostics, hide_index=True, width="stretch")

    st.subheader("Map of deficiency")
    c1, c2 = st.columns(2)
    t = c1.radio("Nutrient", list(TARGETS), format_func=lambda k: NAMES[k], horizontal=True)
    src = c2.radio("Show", ["Lab measured", "Estimated"], index=1, horizontal=True)
    mp = df[["village", "latitude", "longitude"]].copy()
    if src == "Lab measured":
        deficient = df[t] < TARGETS[t]
        mp["detail"] = df[t].map(lambda value: f"Lab measured: {value:.2f} mg/kg")
    else:
        score = df[f"catboost_score_{t}"]
        deficient = score > 0.5
        mp["detail"] = accuracy_label(t)
    mp["status"] = deficient.map({True: "Deficient", False: "Sufficient"})
    mp["color"] = deficient.map(lambda flag: (DEFICIENT_RGB if flag else SUFFICIENT_RGB) + [190])
    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=df.latitude.mean(), longitude=df.longitude.mean(), zoom=7.8),
        layers=[pdk.Layer("ScatterplotLayer", mp, get_position="[longitude, latitude]",
                          get_fill_color="color", get_radius=1000, pickable=True)],
        tooltip={"text": "{village}\n{status}\n{detail}"}), height=430)
    share = deficient.mean()
    st.markdown(f'<span class="pill def">Deficient</span> {share:.0%} of fields &nbsp;&nbsp; '
                f'<span class="pill ok">Sufficient</span> {1 - share:.0%} of fields',
                unsafe_allow_html=True)
