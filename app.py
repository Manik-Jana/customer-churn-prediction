



"""Streamlit interface for the trained churn model (animated gauge edition)."""
import uuid
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/churn_pipeline.joblib")

st.set_page_config(page_title="Customer churn predictor", page_icon="📉", layout="wide")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #12233A;
  --muted: #5B6B7F;
  --mist: #EEF3F8;
  --panel: #FFFFFF;
  --line: #D6E0EB;
  --teal: #0F9D9A;
  --amber: #E8A02E;
  --coral: #DC4A68;
}

.stApp { background: var(--mist); color: var(--ink); font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }

h1.page-title {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
  font-size: clamp(2rem, 4vw, 3rem); letter-spacing: -0.02em;
  line-height: 1.05; margin: 0 0 .4rem 0; color: var(--ink);
}
p.page-sub { color: var(--muted); font-size: 1.05rem; margin: 0 0 1.6rem 0; max-width: 46ch; }

/* Input sections */
.st-key-account, .st-key-services, .st-key-billing {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 14px; padding: 1.1rem 1.3rem .6rem 1.3rem; margin-bottom: 1rem;
}
.section-title {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
  font-size: 1.15rem; margin: 0 0 .6rem 0; color: var(--ink);
}
label, .stSelectbox label p, .stSlider label p, .stNumberInput label p, .stCheckbox p {
  color: var(--ink) !important; font-weight: 500;
}

/* Primary button */
.stButton > button[kind="primary"] {
  background: var(--ink); color: #fff; border: 0; border-radius: 12px;
  padding: .85rem 1.6rem; font-weight: 600; font-size: 1.02rem; width: 100%;
  transition: transform .15s ease, background .2s ease, box-shadow .2s ease;
}
.stButton > button[kind="primary"]:hover {
  background: var(--teal); transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(15,157,154,.28);
}
.stButton > button[kind="primary"]:active { transform: translateY(0); }
.stButton > button:focus-visible { outline: 3px solid var(--teal); outline-offset: 2px; }

/* Result panel */
.st-key-result { position: sticky; top: 1.2rem; }
.result-card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
  padding: 1.6rem 1.4rem 1.4rem 1.4rem; text-align: center;
}
.gauge { width: 100%; max-width: 360px; display: block; margin: 0 auto; overflow: visible; }
.gauge .track { fill: none; stroke: #E3EAF2; stroke-width: 14; stroke-linecap: round; }
.gauge .fill  { fill: none; stroke-width: 14; stroke-linecap: round; stroke-dasharray: 100 100; }
.gauge .tick  { font: 500 9px 'DM Sans', sans-serif; fill: var(--muted); }
.readout {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
  font-size: 4.2rem; line-height: 1; letter-spacing: -0.03em;
  margin-top: -3.6rem; color: var(--ink);
}
.readout .unit { font-size: 1.8rem; font-weight: 500; margin-left: .1rem; color: var(--muted); }
.tier {
  display: inline-block; margin-top: 1rem; padding: .3rem .9rem; border-radius: 999px;
  font-weight: 600; font-size: .95rem; color: #fff;
}
.advice { color: var(--muted); margin: .9rem auto 0 auto; max-width: 34ch; line-height: 1.5; }
.exact { color: var(--muted); font-size: .85rem; margin-top: .6rem; }
.idle-note { color: var(--muted); margin: .9rem auto 0 auto; max-width: 30ch; line-height: 1.5; }

/* Signal chips */
.flags-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; margin: 1.3rem 0 .2rem 0; }
.flags-note { color: var(--muted); font-size: .82rem; margin: 0 0 .6rem 0; }
.chip {
  display: inline-block; margin: 0 .35rem .4rem 0; padding: .28rem .7rem;
  border: 1px solid var(--line); border-radius: 999px; background: #F6F9FC;
  font-size: .85rem; color: var(--ink);
}

@media (prefers-reduced-motion: reduce) {
  .gauge .fill, .readout .num { animation-duration: .01s !important; animation-delay: 0s !important; }
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def choice(label, values, default=0, key=None):
    return st.selectbox(label, values, index=default, key=key)


def risk_tier(p: float):
    if p < 0.30:
        return ("Low risk", "var(--teal)", "This account looks stable. Keep service steady and check in around renewal.")
    if p < 0.60:
        return ("Moderate risk", "var(--amber)", "Reach out before the next billing cycle. A plan review or usage tip can help.")
    return ("High risk", "var(--coral)", "Prioritize a personal call. Offer a contract or bundle that fits how they use the service.")


def gauge_html(p: float | None, nonce: str) -> str:
    """Semicircle gauge. When p is given, the arc sweeps and the number counts up once."""
    arc = "M 20 110 A 90 90 0 0 1 200 110"
    svg_head = '<svg class="gauge" viewBox="0 0 220 130" role="img" aria-label="Churn risk gauge">'
    ticks = ('<text class="tick" x="20" y="128" text-anchor="middle">0%</text>'
             '<text class="tick" x="200" y="128" text-anchor="middle">100%</text>')

    if p is None:
        return (
            '<div class="result-card">' + svg_head
            + f'<path class="track" d="{arc}" pathLength="100"/>' + ticks + "</svg>"
            '<div class="readout">&ndash;<span class="unit">%</span></div>'
            '<p class="idle-note">Fill in the account details, then select '
            "<strong>Estimate churn risk</strong>.</p></div>"
        )

    pct = round(p * 100, 1)
    pct_int = round(p * 100)
    label, color, advice = risk_tier(p)
    sweep, count = f"sweep_{nonce}", f"count_{nonce}"

    css = f"""
    <style>
    @property --n {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
    @keyframes {sweep} {{ from {{ stroke-dashoffset: 100; }} to {{ stroke-dashoffset: {100 - pct}; }} }}
    @keyframes {count} {{ from {{ --n: 0; }} to {{ --n: {pct_int}; }} }}
    .fill-{nonce} {{ stroke: {color}; stroke-dashoffset: 100;
        animation: {sweep} 1.7s cubic-bezier(.2,.8,.2,1) .15s forwards; }}
    .num-{nonce} {{ --n: 0; counter-reset: n var(--n);
        animation: {count} 1.7s cubic-bezier(.2,.8,.2,1) .15s forwards; }}
    .num-{nonce}::after {{ content: counter(n); }}
    </style>
    """
    return (
        css
        + '<div class="result-card">' + svg_head
        + f'<path class="track" d="{arc}" pathLength="100"/>'
        + f'<path class="fill fill-{nonce}" d="{arc}" pathLength="100"/>'
        + ticks + "</svg>"
        + f'<div class="readout"><span class="num num-{nonce}"></span><span class="unit">%</span></div>'
        + f'<div class="tier" style="background:{color}">{label}</div>'
        + f'<p class="advice">{advice}</p>'
        + f'<p class="exact">Estimated churn probability: {pct:.1f}%</p></div>'
    )


def profile_signals(r: dict) -> list[str]:
    """Rule-of-thumb flags based on common telecom churn patterns (not model explanations)."""
    flags = []
    if r["Contract"] == "Month-to-month":
        flags.append("Month-to-month contract")
    if r["tenure"] <= 12:
        flags.append("First year of tenure")
    if r["InternetService"] == "Fiber optic":
        flags.append("Fiber optic plan")
    if r["PaymentMethod"] == "Electronic check":
        flags.append("Pays by electronic check")
    if r["InternetService"] != "No" and r["TechSupport"] == "No":
        flags.append("No tech support")
    if r["InternetService"] != "No" and r["OnlineSecurity"] == "No":
        flags.append("No online security")
    return flags


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.markdown('<h1 class="page-title">Customer churn predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-sub">Enter an account\'s details to see how likely the customer is to leave.</p>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.info("Train a model first by running `python -m src.train` after adding the dataset.")
    st.stop()

model = load_model(MODEL_PATH)

form_col, result_col = st.columns([1.35, 1], gap="large")

with form_col:
    with st.container(key="account"):
        st.markdown('<p class="section-title">Account</p>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            gender = choice("Gender", ["Female", "Male"])
            partner = choice("Has a partner", ["No", "Yes"])
            senior = st.checkbox("Senior citizen")
        with b:
            dependents = choice("Has dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

    with st.container(key="services"):
        st.markdown('<p class="section-title">Services</p>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            phone = choice("Phone service", ["Yes", "No"])
            multiple = choice(
                "Multiple lines",
                ["No", "Yes"] if phone == "Yes" else ["No phone service"],
                key=f"multiple_{phone}",
            )
            internet = choice("Internet service", ["DSL", "Fiber optic", "No"])
            has_net = internet != "No"
            addon = ["No", "Yes"] if has_net else ["No internet service"]
            online_security = choice("Online security", addon, key=f"sec_{has_net}")
            online_backup = choice("Online backup", addon, key=f"bak_{has_net}")
        with b:
            device_protection = choice("Device protection", addon, key=f"dev_{has_net}")
            tech_support = choice("Tech support", addon, key=f"tech_{has_net}")
            streaming_tv = choice("Streaming TV", addon, key=f"tv_{has_net}")
            streaming_movies = choice("Streaming movies", addon, key=f"mov_{has_net}")

    with st.container(key="billing"):
        st.markdown('<p class="section-title">Billing</p>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            contract = choice("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = choice("Paperless billing", ["Yes", "No"])
            payment = choice(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with b:
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            total = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

    submitted = st.button("Estimate churn risk", type="primary")

record = {
    "gender": gender, "SeniorCitizen": int(senior), "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multiple, "InternetService": internet,
    "OnlineSecurity": online_security, "OnlineBackup": online_backup,
    "DeviceProtection": device_protection, "TechSupport": tech_support,
    "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
    "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
}
signature = repr(sorted(record.items()))

if submitted:
    probability = float(model.predict_proba(pd.DataFrame([record]))[0, 1])
    st.session_state["last"] = {
        "signature": signature,
        "probability": probability,
        "nonce": uuid.uuid4().hex[:8],  # new id => animation replays on every estimate
    }

last = st.session_state.get("last")
current = last if last and last["signature"] == signature else None

with result_col:
    with st.container(key="result"):
        if current:
            st.markdown(gauge_html(current["probability"], current["nonce"]), unsafe_allow_html=True)
            flags = profile_signals(record)
            if flags:
                chips = "".join(f'<span class="chip">{f}</span>' for f in flags)
                st.markdown(
                    '<p class="flags-title">Signals worth a look</p>'
                    '<p class="flags-note">Rules of thumb from common telecom churn patterns. '
                    "They are not the model's own explanation.</p>" + chips,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(gauge_html(None, ""), unsafe_allow_html=True)
            if last:
                st.caption("Details changed since the last estimate. Select Estimate churn risk to update.")
