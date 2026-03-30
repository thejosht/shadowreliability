import math
import streamlit as st

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="ShadowReliability",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

lr_model = joblib.load(ARTIFACTS_DIR / "lr_model.joblib")
svm_model = joblib.load(ARTIFACTS_DIR / "svm_model.joblib")
nb_model = joblib.load(ARTIFACTS_DIR / "nb_model.joblib")
shadow_model = joblib.load(ARTIFACTS_DIR / "shadow_model.joblib")
shadow_metadata = joblib.load(ARTIFACTS_DIR / "shadow_metadata.joblib")

SHADOW_FEATURE_NAMES = shadow_metadata["feature_names"]
RISK_THRESHOLD_DEFAULT = shadow_metadata["risk_threshold_default"]

# Static display values
feature_importance = [
    ("Max confidence", 0.457, "#6D95C7"),
    ("Entropy", 0.301, "#8CB7DA"),
    ("Margin", 0.239, "#A8B9A4"),
    ("Disagreement", 0.197, "#D9A25E"),
    ("Text length", 0.167, "#D9BE72"),
]

summary_metrics = [
    ("Clean In Domain Accuracy", "99.4%"),
    ("Test Challenge Accuracy", "86.1%"),
    ("Selective Accuracy at ~80% Coverage", "91.4%"),
    ("Shadow ROC-AUC", "0.844"),
    ("Shadow AP", "0.502"),
]

evaluation_rows = [
    ("Clean In-Domain", "99.4%", "—", "—"),
    ("Challenge", "86.1%", "91.4%", "80.6%"),
    ("OOD", "Lower confidence", "—", "—"),
]

example_text = (
    "I can’t access my medical records online. The portal keeps giving me an error "
    "message when I try to log in, and I can’t reset my password. Please help me regain access."
)


NAV_OPTIONS = ["Overview", "Methodology", "Dataset"]

SAMPLE_REQUESTS = [
    (
        "Portal access",
        "I forgot my password and cannot log into my patient portal. The reset link is not working, and I need help getting back into my account.",
    ),
    (
        "Billing issue",
        "I think I was billed twice for my last appointment. Can someone review the charges on my account and explain the balance?",
    ),
    (
        "Prescription refill",
        "My pharmacy says I have no refills left for my medication. Please send a refill authorization as soon as possible.",
    ),
    (
        "Referral request",
        "My doctor told me I may need to see a specialist. Can you help me with the referral process and let me know what comes next?",
    ),
    (
        "Medical records",
        "I need a copy of my medical records and recent lab results for another appointment. Please let me know how I can request them.",
    ),
    (
        "Appointment scheduling",
        "I would like to schedule a follow up appointment for next week if there are any afternoon openings available.",
    ),
]


st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.72), rgba(255,255,255,0) 24%),
            radial-gradient(circle at 85% 20%, rgba(203,212,229,0.42), rgba(203,212,229,0) 30%),
            linear-gradient(180deg, #e8edf5 0%, #dde5f0 100%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 0.85rem;
        padding-bottom: 1.2rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
    }
    
    
    html, body, [class*="css"] {
        color: #2D3A54;
    }

    .hero-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 2rem;
        margin-bottom: 0.55rem;
        padding-left: 0.15rem;
        padding-right: 0.15rem;
    }

    .hero-title {
        font-size: 3.05rem;
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #2B3954;
        margin-bottom: 0.42rem;
        max-width: 100%;
        overflow-wrap: anywhere;
        pointer-events: none;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #617087;
        max-width: 100%;
        overflow-wrap: anywhere;
        pointer-events: none;
    }

    .top-nav {
        display: flex;
        gap: 0.6rem;
        padding-top: 0;
        white-space: nowrap;
    }

    .mini-note {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 0;
        margin-bottom: 0.2rem;
    }


    .page-copy {
        color: #42536d;
        font-size: 0.98rem;
        line-height: 1.68;
    }

    .bullet-list {
        margin: 0;
        padding-left: 1.15rem;
        color: #42536d;
        line-height: 1.7;
    }

    .soft-panel {
        background: linear-gradient(180deg, rgba(244,248,253,0.95) 0%, rgba(236,242,249,0.95) 100%);
        border: 1px solid #d8e1ec;
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        padding: 0.48rem 0.78rem;
        border-radius: 999px;
        background: linear-gradient(180deg, #edf3fa 0%, #e4ebf5 100%);
        border: 1px solid #d2dcea;
        color: #41516b;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .flow-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .flow-step {
        background: linear-gradient(180deg, #f4f8fd 0%, #ebf1f8 100%);
        border: 1px solid #d6dfeb;
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
        min-height: 92px;
    }

    .flow-step-title {
        color: #30415b;
        font-size: 0.92rem;
        font-weight: 800;
        margin-bottom: 0.22rem;
    }

    .flow-step-copy {
        color: #5d6b80;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 760;
        color: #34425C;
        margin-bottom: 0.75rem;
        letter-spacing: -0.01em;
    }

    .sub-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #3B4A63;
        margin-bottom: 0.48rem;
    }

    .pred-class-box {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, #e6edf7 0%, #dde7f3 100%);
        border: 1px solid #cfd9e8;
        border-radius: 16px;
        color: #314057;
        font-size: 1.06rem;
        font-weight: 700;
        margin-bottom: 0.95rem;
    }

    .pred-icon {
        width: 34px;
        height: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: rgba(255,255,255,0.68);
        font-size: 1.05rem;
    }

    .big-number {
        font-size: 3.05rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #2A3751;
        margin-bottom: 0.12rem;
    }

    .small-muted {
        color: #687587;
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .prob-grid {
        display: grid;
        grid-template-columns: 1fr 150px;
        gap: 0.7rem;
        align-items: center;
    }

    .gauge-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .risk-text {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.08;
        margin-bottom: 0.4rem;
    }

    .risk-score-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 0.8rem;
        align-items: center;
        margin-bottom: 0.68rem;
    }

    .risk-score-value {
        color: #56637B;
        font-size: 1rem;
        font-weight: 700;
    }

    .decision-pill {
        display: inline-block;
        padding: 0.52rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.94rem;
        margin-top: 0.72rem;
    }

    .decision-review {
        background: #FCEDEB;
        color: #B0534D;
        border: 1px solid #EFC9C6;
    }

    .decision-accept {
        background: #EBF7EF;
        color: #2F7F55;
        border: 1px solid #CCE8D7;
    }

    .feature-wrap {
        margin-bottom: 0.95rem;
    }

    .feature-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.95rem;
        color: #3B4860;
        margin-bottom: 0.26rem;
    }

    .feature-track {
        width: 100%;
        height: 16px;
        background: #e7edf5;
        border: 1px solid #d3dce8;
        border-radius: 999px;
        overflow: hidden;
    }

    .feature-fill {
        height: 100%;
        border-radius: 999px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.4);
    }

    .prob-mini-wrap {
        margin-bottom: 0.9rem;
    }

    .prob-mini-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.93rem;
        color: #3B4860;
        margin-bottom: 0.22rem;
    }

    .prob-mini-track {
        width: 100%;
        height: 12px;
        background: #e7edf5;
        border: 1px solid #d3dce8;
        border-radius: 999px;
        overflow: hidden;
    }


    .prob-mini-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(180deg, #8FB7E0 0%, #76A4D3 100%);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.35);
    }

    details {
        background: linear-gradient(180deg, #f7f9fd 0%, #edf2f8 100%) !important;
        border: 1px solid #d7e0ec !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        margin-top: 0.45rem !important;
        box-shadow: 0 10px 22px rgba(53, 74, 110, 0.08) !important;
    }

    details summary {
        background: linear-gradient(180deg, #e6edf7 0%, #dfe8f4 100%) !important;
        color: #35527a !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        padding: 0.58rem 0.82rem !important;
        cursor: pointer !important;
        border-bottom: 1px solid transparent !important;
        min-height: auto !important;
        line-height: 1.2 !important;
    }

    details summary:hover {
        background: linear-gradient(180deg, #cfdcf0 0%, #c3d4eb 100%) !important;
        color: #29476f !important;
    }

    details[open] summary {
        background: linear-gradient(180deg, #cfdcf0 0%, #c3d4eb 100%) !important;
        color: #29476f !important;
        border-bottom: 1px solid #c7d5e7 !important;
    }

    details[open] summary:hover {
        background: linear-gradient(180deg, #c7d7ee 0%, #bbcee8 100%) !important;
        color: #243f63 !important;
    }
    
    .metric-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.94rem;
        color: #34445F;
    }

    .metric-table td,
    .metric-table th {
        padding: 0.58rem 0.46rem;
        border-bottom: 1px solid #EBF0F6;
        vertical-align: top;
    }

    .metric-table tr:last-child td,
    .metric-table tr:last-child th {
        border-bottom: none;
    }

    .metric-table th {
        text-align: left;
        font-weight: 700;
        color: #48566D;
        font-size: 0.9rem;
    }

    .summary-table td:first-child {
        color: #3B4760;
    }

    .summary-table td:last-child {
        text-align: right;
        font-weight: 700;
        color: #30415B;
    }

    .stTextArea textarea {
        background: #fbfcfe !important;
        color: #2B3952 !important;
        border: 1px solid #cad4e2 !important;
        border-radius: 17px !important;
        padding: 1rem !important;
        box-shadow: inset 0 1px 3px rgba(20, 35, 70, 0.05) !important;
        font-size: 1.02rem !important;
        line-height: 1.55 !important;
        min-height: 255px !important;
    }

    .stButton button {
        background: linear-gradient(180deg, #8FB7E0 0%, #76A4D3 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 13px !important;
        font-weight: 700 !important;
        padding: 0.78rem 1.35rem !important;
        box-shadow: 0 8px 20px rgba(94, 132, 187, 0.23) !important;
    }

    .stButton button:hover {
        background: linear-gradient(180deg, #84ADD8 0%, #6A97C8 100%) !important;
        color: white !important;
    }

    .stProgress > div > div > div > div {
        background-color: #7EA9D6;
    }

    .stProgress > div > div {
        background-color: #dde6f1;
    }
    
    [data-testid="stTextAreaRootElement"] label,
    label,
    .section-title,
    .sub-title,
    .small-muted {
        text-shadow: 0 1px 0 rgba(255,255,255,0.35);
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    button[kind="header"] {
        display: none !important;
    }

    .stAppDeployButton {
        display: none !important;
    }

    [data-testid="stAlertContainer"] {
        background: linear-gradient(180deg, #edf2f8 0%, #e7edf6 100%) !important;
        border: 1px solid #d7e0ec !important;
        border-radius: 18px !important;
    }
    
    code {
        background: transparent !important;
        color: inherit !important;
    }

    pre,
    [data-testid="stMarkdownContainer"] code,
    [data-testid="stCodeBlock"] {
        display: none !important;
    }

</style>
""",
    unsafe_allow_html=True,
)


def gauge_svg(value: float) -> str:
    cx, cy = 86, 86

    def polar(angle_deg: float, radius: float):
        rad = math.radians(angle_deg)
        return cx + radius * math.cos(rad), cy + radius * math.sin(rad)

    def arc_path(start_deg: float, end_deg: float, radius: float):
        x1, y1 = polar(start_deg, radius)
        x2, y2 = polar(end_deg, radius)
        large_arc = 1 if abs(end_deg - start_deg) > 180 else 0
        return f"M {x1:.2f} {y1:.2f} A {radius} {radius} 0 {large_arc} 1 {x2:.2f} {y2:.2f}"

    start_angle = 200
    end_angle = 340
    value_angle = start_angle + (end_angle - start_angle) * max(0.0, min(1.0, value))
    needle_x, needle_y = polar(value_angle, 44)

    return f"""
    <div class="gauge-wrap">
        <svg width="150" height="96" viewBox="0 0 170 110" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="{arc_path(200, 255, 62)}" stroke="#EFCE86" stroke-width="14" stroke-linecap="round"/>
            <path d="{arc_path(255, 300, 62)}" stroke="#DFA062" stroke-width="14" stroke-linecap="round"/>
            <path d="{arc_path(300, 340, 62)}" stroke="#EADFD8" stroke-width="14" stroke-linecap="round"/>
            <line x1="86" y1="86" x2="{needle_x:.2f}" y2="{needle_y:.2f}" stroke="#5F6678" stroke-width="5" stroke-linecap="round"/>
            <circle cx="86" cy="86" r="7" fill="#D88542"/>
        </svg>
    </div>
    """


def feature_bars_html(items) -> str:
    rows = []
    for label, value, color in items:
        rows.append(
            f"""
            <div class="feature-wrap">
                <div class="feature-row">
                    <span>{label}</span>
                    <span><b>{value:.3f}</b></span>
                </div>
                <div class="feature-track">
                    <div class="feature-fill" style="width:{min(max(value, 0.0), 1.0) * 100:.1f}%; background:{color};"></div>
                </div>
            </div>
            """
        )
    return "".join(rows)


def summary_table_html(rows) -> str:
    body = "".join([f"<tr><td>{label}</td><td><b>{value}</b></td></tr>" for label, value in rows])
    return f'<table class="metric-table summary-table">{body}</table>'


def evaluation_table_html(rows) -> str:
    body = "".join(
        [f"<tr><td>{name}</td><td>{acc}</td><td>{sel}</td><td>{cov}</td></tr>" for name, acc, sel, cov in rows]
    )
    return f"""
    <table class="metric-table">
        <thead>
            <tr>
                <th>Set</th>
                <th>Main Accuracy</th>
                <th>Selective Accuracy</th>
                <th>Coverage</th>
            </tr>
        </thead>
        <tbody>
            {body}
        </tbody>
    </table>
    """

def class_probability_html(prob_map: dict) -> str:
    rows = []
    for label, value in prob_map.items():
        rows.append(
            f'''
            <div class="prob-mini-wrap">
                <div class="prob-mini-row">
                    <span>{label}</span>
                    <span><b>{value:.3f}</b></span>
                </div>
                <div class="prob-mini-track">
                    <div class="prob-mini-fill" style="width:{value * 100:.1f}%;"></div>
                </div>
            </div>
            '''
        )
    return "".join(rows)


def entropy_from_probas(probas: np.ndarray) -> np.ndarray:
    p = np.clip(probas, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)

def margin_from_probas(probas: np.ndarray) -> np.ndarray:
    sorted_p = np.sort(probas, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]

def agreement_features(lr_pred, svm_pred, nb_pred) -> pd.DataFrame:
    lr_pred = np.array(lr_pred)
    svm_pred = np.array(svm_pred)
    nb_pred = np.array(nb_pred)

    all_agree = ((lr_pred == svm_pred) & (lr_pred == nb_pred)).astype(int)
    any_disagree = 1 - all_agree
    num_unique_preds = np.array([len(set(x)) for x in zip(lr_pred, svm_pred, nb_pred)])
    majority_vote_size = np.array([
        max(pd.Series([a, b, c]).value_counts()) for a, b, c in zip(lr_pred, svm_pred, nb_pred)
    ])

    return pd.DataFrame({
        "all_agree": all_agree,
        "any_disagree": any_disagree,
        "num_unique_preds": num_unique_preds,
        "majority_vote_size": majority_vote_size,
    })

def prob_feature_block(probas: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({
        f"{prefix}_max_conf": probas.max(axis=1),
        f"{prefix}_margin": margin_from_probas(probas),
        f"{prefix}_entropy": entropy_from_probas(probas),
    })

def build_shadow_features_for_text(text: str) -> pd.DataFrame:
    texts = pd.Series([text])

    lr_proba = lr_model.predict_proba(texts)
    svm_proba = svm_model.predict_proba(texts)
    nb_proba = nb_model.predict_proba(texts)

    lr_pred = lr_model.predict(texts)
    svm_pred = svm_model.predict(texts)
    nb_pred = nb_model.predict(texts)

    shadow_x = pd.concat([
        prob_feature_block(lr_proba, "lr"),
        prob_feature_block(svm_proba, "svm"),
        prob_feature_block(nb_proba, "nb"),
        agreement_features(lr_pred, svm_pred, nb_pred),
        pd.DataFrame({
            "text_len_chars": [len(str(text))],
            "text_len_words": [len(str(text).split())],
        }),
    ], axis=1)

    shadow_x = shadow_x[SHADOW_FEATURE_NAMES]
    return shadow_x

def run_inference(text: str):
    texts = pd.Series([text])

    lr_proba = lr_model.predict_proba(texts)
    lr_pred = lr_model.predict(texts)[0]

    class_prob_map = {
        cls: float(prob)
        for cls, prob in zip(lr_model.classes_, lr_proba[0])
    }
    class_prob_map = dict(sorted(class_prob_map.items(), key=lambda x: x[1], reverse=True))

    predicted_class = str(lr_pred)
    predicted_prob = float(lr_proba.max(axis=1)[0])

    shadow_x = build_shadow_features_for_text(text)
    shadow_risk = float(shadow_model.predict_proba(shadow_x)[:, 1][0])

    possible_ood = predicted_prob < 0.55

    risk_label = "High Risk" if shadow_risk >= 0.65 else "Medium Risk" if shadow_risk >= 0.4 else "Low Risk"
    risk_color = "#B55750" if shadow_risk >= 0.65 else "#C78A2E" if shadow_risk >= 0.4 else "#2E8A5B"
    decision_label = (
        "Manual review recommended"
        if shadow_risk >= 0.65
        else "Review advised"
        if shadow_risk >= 0.4
        else "Accept prediction"
    )
    decision_class = "decision-review" if shadow_risk >= 0.4 else "decision-accept"

    return {
        "predicted_class": predicted_class,
        "predicted_prob": predicted_prob,
        "class_prob_map": class_prob_map,
        "shadow_risk": shadow_risk,
        "possible_ood": possible_ood,
        "risk_label": risk_label,
        "risk_color": risk_color,
        "decision_label": decision_label,
        "decision_class": decision_class,
    }


if "current_text" not in st.session_state:
    st.session_state.current_text = example_text

if "current_result" not in st.session_state:
    st.session_state.current_result = run_inference(example_text)

if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"


def set_active_page(page_name: str):
    st.session_state.active_page = page_name


def apply_sample_request(sample_text: str):
    st.session_state.current_text = sample_text
    st.session_state.request_text_input = sample_text
    st.session_state.current_result = run_inference(sample_text)


predicted_class = st.session_state.current_result["predicted_class"]
predicted_prob = st.session_state.current_result["predicted_prob"]
class_prob_map = st.session_state.current_result["class_prob_map"]
shadow_risk = st.session_state.current_result["shadow_risk"]
possible_ood = st.session_state.current_result["possible_ood"]
risk_label = st.session_state.current_result["risk_label"]
risk_color = st.session_state.current_result["risk_color"]
decision_label = st.session_state.current_result["decision_label"]
decision_class = st.session_state.current_result["decision_class"]

st.markdown(
    '''
    <div>
        <div class="hero-title">ShadowReliability</div>
        <div class="hero-subtitle">Failure Prediction Framework for Text Classification Systems</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown('<div style="height:0.18rem;"></div>', unsafe_allow_html=True)

nav_cols = st.columns(3, gap="small")
for idx, page_name in enumerate(NAV_OPTIONS):
    with nav_cols[idx]:
        st.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
            on_click=set_active_page,
            args=(page_name,),
        )

st.markdown('<div style="height:0.28rem;"></div>', unsafe_allow_html=True)

if st.session_state.active_page == "Overview":
    st.markdown(
        '<div class="mini-note">Overview shows the live interactive demo, current performance summary, and reliability signals.</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.active_page == "Methodology":
    st.markdown(
        '<div class="mini-note">Methodology summarizes the modeling pipeline, shadow model features, and why the system uses a second reliability layer.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="mini-note">Dataset summarizes the task setting, class labels, and the role of the clean and challenge evaluation splits.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="height:0.12rem;"></div>', unsafe_allow_html=True)


if st.session_state.active_page in ["Methodology", "Dataset"]:
    top_left, top_right = st.columns([2.2, 1.45], gap="large")
else:
    top_left, top_right = st.columns([2.9, 1.15], gap="large")

if st.session_state.active_page == "Methodology":
    with top_left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
            st.markdown(
                '''
                <div class="page-copy">
                    ShadowReliability uses a two layer design. The main classifier predicts the healthcare administrative request class, while a second reliability layer estimates when that prediction may be wrong.
                </div>
                <div style="height:0.8rem;"></div>
                <div class="flow-grid">
                    <div class="flow-step">
                        <div class="flow-step-title">1. Input Request</div>
                        <div class="flow-step-copy">A user submits a healthcare administrative request written in natural language.</div>
                    </div>
                    <div class="flow-step">
                        <div class="flow-step-title">2. Main Prediction</div>
                        <div class="flow-step-copy">A TF-IDF with Logistic Regression classifier predicts the most likely class.</div>
                    </div>
                    <div class="flow-step">
                        <div class="flow-step-title">3. Reliability Signals</div>
                        <div class="flow-step-copy">Confidence, margin, entropy, disagreement, and text length features are extracted.</div>
                    </div>
                    <div class="flow-step">
                        <div class="flow-step-title">4. Risk Decision</div>
                        <div class="flow-step-copy">A shadow model estimates failure risk and supports accept versus review decisions.</div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:0.55rem;"></div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-title">Shadow Model Features</div>', unsafe_allow_html=True)
            st.markdown(
                '''
                <div class="soft-panel">
                    <ul class="bullet-list">
                        <li><b>Maximum confidence</b>: the highest class probability assigned by a model</li>
                        <li><b>Margin</b>: the gap between the top class probability and the second highest one</li>
                        <li><b>Entropy</b>: a simple measure of uncertainty; higher entropy means the model is more spread across classes</li>
                        <li><b>Disagreement</b>: whether the models agree or predict different classes</li>
                        <li><b>Vote structure</b>: number of unique predictions and majority vote size</li>
                        <li><b>Text length</b>: number of words and characters in the request</li>
                    </ul>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    with top_right:
        with st.container(border=True):
            st.markdown('<div class="section-title">Model Stack</div>', unsafe_allow_html=True)
            st.markdown(
                '''
                <div class="chip-row">
                    <span class="chip">TF-IDF with Logistic Regression</span>
                    <span class="chip">SVM</span>
                    <span class="chip">Naive Bayes</span>
                    <span class="chip">Corrected OOF Shadow Model</span>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:0.35rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Current Results</div>', unsafe_allow_html=True)
            st.markdown(summary_table_html(summary_metrics), unsafe_allow_html=True)

    st.stop()

if st.session_state.active_page == "Dataset":
    with top_left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Dataset and Task</div>', unsafe_allow_html=True)
            st.markdown(
                '''
                <div class="page-copy">
                    The task is to classify healthcare administrative requests into one of six categories. The system is evaluated on both cleaner in domain data and a more difficult challenge setting, which makes it possible to test not only accuracy, but also reliability under harder conditions.
                </div>
                ''',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:0.55rem;"></div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-title">Class Labels</div>', unsafe_allow_html=True)
            st.markdown(
                '''
                <div class="chip-row">
                    <span class="chip">Appointment Scheduling</span>
                    <span class="chip">Billing and Insurance</span>
                    <span class="chip">Prescription Refill</span>
                    <span class="chip">Referral Request</span>
                    <span class="chip">Medical Records and Forms</span>
                    <span class="chip">Portal and Account Access</span>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    with top_right:
        with st.container(border=True):
            st.markdown('<div class="section-title">Evaluation Setting</div>', unsafe_allow_html=True)
            st.markdown(evaluation_table_html(evaluation_rows), unsafe_allow_html=True)
            st.markdown('<div style="height:0.35rem;"></div>', unsafe_allow_html=True)
            st.markdown(
                '''
                <div class="soft-panel">
                    <div class="mini-note" style="margin:0;">
                        Clean in domain data reflects easier examples that are closer to the training distribution. The challenge set contains harder cases that better test whether the reliability layer can separate safer predictions from riskier ones.
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    st.stop()


with top_left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Enter a request:</div>', unsafe_allow_html=True)
        user_text = st.text_area(
            "",
            value=st.session_state.current_text,
            height=255,
            label_visibility="collapsed",
            key="request_text_input",
        )

        if st.button("Analyze"):
            st.session_state.current_text = user_text
            st.session_state.current_result = run_inference(user_text)

        st.markdown('<div style="height:0.7rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Try Sample Requests</div>', unsafe_allow_html=True)
        st.markdown(
            '''
            <div class="mini-note">
                Start with one of the example requests below, or write your own realistic healthcare administrative message to explore how the model and reliability layer respond.
                <br><br>
                Best results come from short, natural requests similar to what a patient might send to a clinic, hospital office, or patient portal support team.
            </div>
            ''',
            unsafe_allow_html=True,
        )

        sample_cols_top = st.columns(3, gap="small")
        for idx, (label, sample_text) in enumerate(SAMPLE_REQUESTS[:3]):
            with sample_cols_top[idx]:
                st.button(
                    label,
                    key=f"sample_top_{idx}",
                    use_container_width=True,
                    on_click=apply_sample_request,
                    args=(sample_text,),
                )

        sample_cols_bottom = st.columns(3, gap="small")
        for idx, (label, sample_text) in enumerate(SAMPLE_REQUESTS[3:]):
            with sample_cols_bottom[idx]:
                st.button(
                    label,
                    key=f"sample_bottom_{idx}",
                    use_container_width=True,
                    on_click=apply_sample_request,
                    args=(sample_text,),
                )

with top_right:
    with st.container(border=True):
        st.markdown('<div class="section-title">Predicted Class</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pred-class-box"><span class="pred-icon">🗂</span><span>{predicted_class}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sub-title">Predicted Probability</div>', unsafe_allow_html=True)
        st.markdown(
            f'''
            <div class="prob-grid">
                <div>
                    <div style="height:0.15rem;"></div>
                    <div class="big-number">{predicted_prob:.2f}</div>
                    <div class="small-muted">Healthcare Requests Dataset</div>
                </div>
                {gauge_svg(predicted_prob)}
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
        st.progress(predicted_prob)
        with st.expander("See class probability breakdown"):
            st.markdown(class_probability_html(class_prob_map), unsafe_allow_html=True)

st.markdown('<div style="height:0.65rem;"></div>', unsafe_allow_html=True)

left_col, center_col, right_col = st.columns([1.18, 1.38, 1.08], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">Shadow Model Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-text" style="color:{risk_color};">⚠ {risk_label}</div>', unsafe_allow_html=True)
        st.progress(shadow_risk)
        st.markdown(
            f'<div class="risk-score-value" style="margin-top:0.45rem; margin-bottom:0.55rem;">{shadow_risk:.2f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sub-title">Risk Factors:</div>', unsafe_allow_html=True)
        st.markdown(
            '''
            <div class="small-muted">
                ✔ Low confidence score<br>
                ✔ High prediction entropy<br>
                ✔ Model disagreement
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">ShadowReliability Summary</div>', unsafe_allow_html=True)
        st.markdown(summary_table_html(summary_metrics), unsafe_allow_html=True)

with center_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">Risk Level</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-text" style="color:{risk_color};">⚠ {risk_label}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="small-muted" style="font-size:1rem; max-width:95%;">There is a high risk that the classifier is incorrect in this case.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="decision-pill {decision_class}">{decision_label}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">ShadowReliability Evaluation Metrics</div>', unsafe_allow_html=True)
        st.markdown(evaluation_table_html(evaluation_rows), unsafe_allow_html=True)

with right_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        st.markdown(feature_bars_html(feature_importance), unsafe_allow_html=True)
        st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
        if possible_ood:
            st.info("Possible OOD\n\nThis request may be out of the model's strongest support region.")