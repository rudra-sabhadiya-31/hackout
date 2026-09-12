"""
app.py - BharatBanker AI Interactive Unified Platform (Streamlit)
Features:
  1. Customer Portal:
     - 0-100 Financial Health Gauge & Metrics
     - Single Hyper-Personalized Recommendation Card with plain-language SHAP tags
     - Empathetic Hard Veto Substitution Banner when stress or DTI > 50% is detected
     - Embedded Vernacular Chat Assistant (Hindi / English / Hinglish)
  2. Banker / Judge Live Veto Auditor (The Demo Showstopper):
     - Real-time interactive sliders (Salary, EMI, DTI, Medical Surge, Savings Decay)
     - Live reaction: Unconstrained predatory ML offer vs. BharatBanker Hard Veto Floor
  3. Canonical Judge Pitch Scenarios (Priya Sharma, Amit Patel, Sunita Devi)
  4. Contract Inspector (Live JSON for Contract 1, 2, and 3)
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st

# Configure UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Page Config
st.set_page_config(
    page_title="BharatBanker AI — Unified Decisioning Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark-themed, elegant glassmorphic banking UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 24px;
    }
    .card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .veto-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(185, 28, 28, 0.2) 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
    }
    .healthy-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-healthy { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
    .badge-watch { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid #3B82F6; }
    .badge-early { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
    .badge-distressed { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }
    .shap-tag {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 13px;
        margin: 4px 0;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Lazy Module Importers & Singletons
@st.cache_resource
def load_person_1_pipeline():
    from feature_pipeline import CustomerFeaturePipeline
    from segmentation_engine import SegmentationEngine
    from recommendation_engine import RecommendationEngine

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pipeline = CustomerFeaturePipeline(data_dir=data_dir)
    feature_df = pipeline.load_features()

    seg_engine = SegmentationEngine()
    seg_engine.load()

    rec_engine = RecommendationEngine()
    rec_engine.segmentation_engine = seg_engine
    rec_engine.load_models()

    return pipeline, feature_df, seg_engine, rec_engine

@st.cache_resource
def load_chat_service():
    try:
        from chat_service import ChatService
        return ChatService()
    except Exception as e:
        return None

try:
    from veto_decision_layer import evaluate_veto_layer, EmpatheticPolicyEngine
except ImportError:
    evaluate_veto_layer = None

# Initialize Pipeline
pipeline, feature_df, seg_engine, rec_engine = load_person_1_pipeline()
chat_service = load_chat_service()

# --- Sidebar ---
st.sidebar.markdown("## 🏦 BharatBanker AI")
st.sidebar.caption("Unified Decisioning Platform for Digital Transformation in Lending")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    [
        "📱 Customer Portal & Personalization",
        "🛡️ Banker / Judge Live Veto Auditor",
        "🌟 Canonical Pitch Scenarios",
        "🔍 Architecture & Contract Inspector"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Select Customer Profile")

# Quick Buttons for Canonical Demo Users
col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("Priya (Jump)", use_container_width=True):
    st.session_state["selected_cid"] = "CUST_IND_1042"
if col_b2.button("Amit (Stress)", use_container_width=True):
    st.session_state["selected_cid"] = "CUST_IND_1088"

col_b3, col_b4 = st.sidebar.columns(2)
if col_b3.button("Ramesh (Stable)", use_container_width=True):
    st.session_state["selected_cid"] = "CUST_IND_1015"
if col_b4.button("Sunita (Tier-2)", use_container_width=True):
    st.session_state["selected_cid"] = "CUST_IND_1002"

customer_list = feature_df["customer_id"].tolist()
default_index = 0
if "selected_cid" in st.session_state and st.session_state["selected_cid"] in customer_list:
    default_index = customer_list.index(st.session_state["selected_cid"])

selected_cid = st.sidebar.selectbox(
    "All 100 Customer Profiles",
    customer_list,
    index=default_index,
    format_func=lambda x: f"{x} - {feature_df[feature_df['customer_id'] == x]['name'].values[0]}"
)
st.session_state["selected_cid"] = selected_cid

# Fetch Active Customer Feature Vector
cust_row = pipeline.get_feature_vector(selected_cid)
contract1_vec = rec_engine.get_contract_1_vector(cust_row)

# Calculate Financial Health Score (0 to 100)
# Formula: based on DTI, savings decay, balance volatility, and disposable income
base_dti = cust_row.get("dti_ratio", 0.0)
sr_decay = cust_row.get("savings_rate_decay", 0.0)
med_growth = cust_row.get("medical_spend_growth", 0.0)

# Compute normalized 0-100 score
raw_health = 100.0 - (base_dti * 80.0) + (cust_row.get("savings_ratio", 0.1) * 30.0) - (max(0.0, -sr_decay) * 50.0) - (max(0.0, med_growth) * 20.0)
health_score = max(5.0, min(98.0, raw_health))

if health_score >= 80:
    health_cat = "Healthy"
    badge_cls = "badge-healthy"
elif health_score >= 60:
    health_cat = "Watch"
    badge_cls = "badge-watch"
elif health_score >= 40:
    health_cat = "Early Stress"
    badge_cls = "badge-early"
elif health_score >= 20:
    health_cat = "Distressed"
    badge_cls = "badge-distressed"
else:
    health_cat = "Critical"
    badge_cls = "badge-distressed"

# Evaluate Veto Layer
veto_result = None
if evaluate_veto_layer:
    raw_recs_dicts = [
        {"product_type": r.product_type, "raw_propensity_score": r.raw_propensity_score, "shap_reasons": r.shap_reasons}
        for r in contract1_vec.candidate_recommendations
    ]
    veto_result = evaluate_veto_layer(
        customer_id=selected_cid,
        raw_propensity_recs=raw_recs_dicts,
        stress_score=100.0 - health_score,
        dti=base_dti
    )


# ==============================================================================
# MODE 1: CUSTOMER PORTAL & PERSONALIZATION
# ==============================================================================
if app_mode == "📱 Customer Portal & Personalization":
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: #F8FAFC; font-size: 28px;">Welcome, {cust_row['name']}</h1>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">
                    {cust_row['occupation']} • {cust_row['city']} • Account Vintage: {cust_row['account_vintage_months']} Months
                </p>
            </div>
            <div style="text-align: right;">
                <span class="badge {badge_cls}">{health_cat} ({health_score:.0f}/100)</span>
                <p style="margin: 4px 0 0 0; color: #64748B; font-size: 12px;">DPDP Consent Tier {cust_row.get('consent_tier', 1)}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Key Financial Health Indicators
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Monthly Inflow", f"₹{cust_row['monthly_salary']:,.0f}", f"{cust_row.get('salary_growth_rate', 0.0)*100:+.0f}% 3mo trend")
    with c2:
        st.metric("Debt-to-Income (DTI)", f"{cust_row['dti_ratio']*100:.1f}%", "- Safety Cap: 50%", delta_color="inverse" if cust_row['dti_ratio'] > 0.45 else "normal")
    with c3:
        st.metric("Monthly Surplus", f"₹{cust_row['disposable_income']:,.0f}", "After EMIs & Bills")
    with c4:
        st.metric("Behavioral Segment", contract1_vec.cluster_name, f"{len(contract1_vec.detected_life_stages)} Active Triggers")

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.subheader("🎯 Single Hyper-Personalized Action")

        # Check if Veto Triggered
        if veto_result and veto_result.get("veto_triggered"):
            final_action = veto_result["final_action"]
            st.markdown(f"""
            <div class="veto-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #EF4444; font-weight: 700; font-size: 14px; letter-spacing: 0.5px;">
                        🛡️ ETHICAL VETO TRIGGERED (PREDATORY OFFERS BLOCKED)
                    </span>
                    <span class="badge badge-distressed">Hard Floor Active</span>
                </div>
                <h3 style="color: #F8FAFC; margin-top: 10px; margin-bottom: 8px;">{final_action.get('display_title', 'Empathetic Financial Relief')}</h3>
                <p style="color: #CBD5E1; font-size: 14px; margin-bottom: 12px;">
                    {final_action.get('message_en', 'We noticed your monthly balance is tighter than usual. Would you like to reschedule your upcoming EMI with zero penalty?')}
                </p>
                <div style="background: rgba(0,0,0,0.25); padding: 10px 14px; border-radius: 8px; margin-bottom: 16px;">
                    <p style="color: #FBBF24; font-size: 14px; margin: 0; font-style: italic;">
                        🇮🇳 Vernacular (Hindi): "{final_action.get('vernacular_message_hi', 'Humne dekha ki is mahine aapke kharche badh gaye hain. Kya aap apni agli EMI aage badhana chahte hain?')}"
                    </p>
                </div>
                <p style="color: #94A3B8; font-size: 12px; margin-bottom: 14px;">
                    <strong>Decision Discipline:</strong> Unconstrained ML propensity attempted to surface high-interest credit, but BharatBanker's hard rule suppressed all loan pushes to protect you from debt traps.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.button(f"👉 {final_action.get('cta_action', 'REQUEST_EMI_RELIEF')}", type="primary", use_container_width=True)

        else:
            # Healthy / Approved Personalization Flow
            top_rec = contract1_vec.candidate_recommendations[0] if contract1_vec.candidate_recommendations else None
            if top_rec:
                st.markdown(f"""
                <div class="healthy-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #10B981; font-weight: 700; font-size: 14px; letter-spacing: 0.5px;">
                            ⭐ PROACTIVE RECOMMENDATION (AFFINITY: {top_rec.raw_propensity_score*100:.0f}%)
                        </span>
                        <span class="badge badge-healthy">Approved by Veto Layer</span>
                    </div>
                    <h3 style="color: #F8FAFC; margin-top: 10px; margin-bottom: 8px;">{top_rec.product_type.replace('_', ' ')}</h3>
                    <p style="color: #CBD5E1; font-size: 14px; margin-bottom: 12px;">
                        Matched specifically to your recent transaction trajectory and life-stage event signals.
                    </p>
                    <div style="margin-bottom: 16px;">
                        <strong style="color: #94A3B8; font-size: 12px; text-transform: uppercase;">SHAP-Generated Rationale:</strong>
                        {"".join([f'<div class="shap-tag">• {reason}</div>' for reason in top_rec.shap_reasons])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.button(f"👉 Explore {top_rec.product_type.replace('_', ' ')}", type="primary", use_container_width=True)
            else:
                st.info("No active credit offers surfaced today. Your finances are running smoothly!")

    # Embedded Vernacular Conversational Assistant
    with col_right:
        st.subheader("💬 Vernacular Banking Assistant")
        st.caption("Hindi, Hinglish & English • Aadhaar Verhoeff Validation • Grounded RBI RAG")

        # Chat history session state
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {"role": "assistant", "content": f"Namaste {cust_row['name']}! Main BharatBanker Sahayak hoon. Main aapki loan aavedan, KYC, ya banking niyam samajhne mein kaise madad kar sakta hoon?"}
            ]

        chat_container = st.container(height=380)
        for msg in st.session_state["chat_messages"]:
            with chat_container.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citation" in msg and msg["citation"]:
                    st.caption(f"📚 {msg['citation']}")

        user_input = st.chat_input("Type in Hindi, English, or Hinglish (e.g., 'Cooling off period kya hai?')...")
        if user_input:
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})
            with chat_container.chat_message("user"):
                st.markdown(user_input)

            # Route through chat service
            if chat_service:
                resp = chat_service.process_message(session_id=f"sess_{selected_cid}", user_text=user_input)
                bot_text = resp.get("bot_message", "Aapka aavedan prakriya mein hai.")
                citation = resp.get("grounded_citation")
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": bot_text,
                    "citation": citation
                })
            else:
                bot_text = "Chat service dependencies are loading. Deterministic loan slot-filling is available."
                st.session_state["chat_messages"].append({"role": "assistant", "content": bot_text})
            st.rerun()


# ==============================================================================
# MODE 2: BANKER / JUDGE LIVE VETO AUDITOR (THE DEMO SHOWSTOPPER)
# ==============================================================================
elif app_mode == "🛡️ Banker / Judge Live Veto Auditor":
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #F8FAFC; margin: 0; font-size: 26px;">Banker & Judge Live Veto Auditor</h1>
        <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 15px;">
            The Demo Showstopper: Dynamically manipulate customer stress and watch the Hard Veto Floor revoke predatory credit offers in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_sim, col_res = st.columns([1, 1.2])

    with col_sim:
        st.subheader("🎛️ Live Customer Stress Controls")
        st.caption(f"Auditing Profile: {cust_row['name']} ({selected_cid})")

        sim_salary = st.slider("Monthly Salary (₹)", 20000, 200000, int(cust_row["monthly_salary"]), step=5000)
        sim_emi = st.slider("Monthly EMI Outflow (₹)", 0, 100000, int(cust_row["monthly_emi"]), step=2000)
        sim_dti = sim_emi / (sim_salary + 1e-5)

        st.markdown(f"**Live Calculated DTI:** `{sim_dti*100:.1f}%` (Safety Threshold: 50.0%)")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            med_spike = st.toggle("Simulate Medical Surge (>50%)", value=bool(cust_row.get("medical_spend_growth", 0) > 0.4))
        with col_t2:
            sim_fraud = st.toggle("Simulate Device/SIM Anomaly", value=False)

        sim_savings_decay = st.slider("Savings Trajectory Slope", -0.25, 0.25, float(cust_row.get("savings_rate_decay", 0.0)), step=0.02)

        # Dynamic Stress Score calculation
        sim_stress = min(100.0, max(0.0, (sim_dti * 90.0) + (max(0.0, -sim_savings_decay) * 60.0) + (30.0 if med_spike else 0.0) + (50.0 if sim_fraud else 0.0)))
        st.progress(sim_stress / 100.0, text=f"Calculated Stress Level: {sim_stress:.0f}/100")

    with col_res:
        st.subheader("⚖️ Real-Time Arbitration Engine")

        # Candidate unconstrained ML recommendation
        predatory_score = 0.88
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #F8FAFC; margin: 0 0 6px 0;">1. Unconstrained ML Propensity Model</h4>
            <p style="color: #94A3B8; font-size: 13px; margin: 0;">
                Candidate Offer: <strong style="color: #60A5FA;">Instant ₹2,00,000 Personal Loan</strong> (Raw Propensity Score: {predatory_score:.2f})
            </p>
            <p style="color: #64748B; font-size: 12px; margin-top: 4px;">
                Status quo banking apps push high-margin loans regardless of borrower distress.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Run Live Veto Evaluation
        sim_recs = [
            {"product_type": "PERSONAL_LOAN", "raw_propensity_score": predatory_score, "shap_reasons": ["High transaction count"]},
            {"product_type": "HEALTH_INSURANCE", "raw_propensity_score": 0.72, "shap_reasons": ["Family protection requirement"]},
            {"product_type": "SIP_INVESTMENT", "raw_propensity_score": 0.65, "shap_reasons": ["Automated savings surplus"]}
        ]

        if evaluate_veto_layer:
            live_veto = evaluate_veto_layer(
                customer_id=selected_cid,
                raw_propensity_recs=sim_recs,
                stress_score=sim_stress,
                dti=sim_dti
            )

            if live_veto["veto_triggered"]:
                st.markdown(f"""
                <div class="veto-card">
                    <h3 style="color: #EF4444; margin: 0 0 8px 0;">🚨 HARD VETO FLOOR ACTIVATED</h3>
                    <p style="color: #F8FAFC; font-weight: 600; font-size: 15px; margin-bottom: 6px;">
                        Action: Personal Loan Offer REVOKED & BLOCKED
                    </p>
                    <p style="color: #CBD5E1; font-size: 13px;">
                        <strong>Trigger Reason:</strong> {live_veto.get('veto_reason', 'DTI > 50% or Critical Financial Stress')}
                    </p>
                    <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; margin-top: 12px;">
                        <strong style="color: #10B981; font-size: 13px;">Substituted Empathetic Action:</strong>
                        <p style="color: #F8FAFC; margin: 4px 0 0 0; font-size: 14px;">
                            "{live_veto['final_action'].get('display_title')}" — {live_veto['final_action'].get('message_en')}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="healthy-card">
                    <h3 style="color: #10B981; margin: 0 0 8px 0;">✅ VETO CLEARED — RESPONSIBLE OFFER ALLOWED</h3>
                    <p style="color: #F8FAFC; font-size: 14px; margin-bottom: 4px;">
                        Borrower debt capacity is healthy (DTI: {sim_dti*100:.1f}% &lt; 50%).
                    </p>
                    <p style="color: #94A3B8; font-size: 13px;">
                        Approved Action: <strong>{live_veto['final_action'].get('action_type', 'PRODUCT_RECOMMENDATION')}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.caption("🔒 Every veto evaluation generates an immutable audit record and complies with RBI Digital Lending Fair Practices.")


# ==============================================================================
# MODE 3: CANONICAL PITCH SCENARIOS SHOWCASE
# ==============================================================================
elif app_mode == "🌟 Canonical Pitch Scenarios":
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #F8FAFC; margin: 0; font-size: 26px;">Judge Pitch Scenarios Showcase</h1>
        <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 15px;">
            The three pre-configured canonical customer storylines satisfying the hackathon evaluation rubrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.tabs([
        "Scenario 1: Priya Sharma (Upward Earner)",
        "Scenario 2: Amit Patel (Stressed Merchant)",
        "Scenario 3: Sunita Devi (Vernacular User)"
    ])

    with sc1:
        st.subheader("Priya Sharma — Young Software Engineer (Bengaluru)")
        st.markdown("""
        - **Life Event Detected:** Salary credited on 1st of month increased from **₹45,000 to ₹75,000** (+67% jump over 2 months).
        - **Problem Status Quo:** Generic banks spam her with 5 different personal loan pop-ups.
        - **BharatBanker Action:** Surfaced **1 single high-affinity recommendation: Tax-Saving ELSS SIP**.
        - **Plain-Language SHAP Reason:** *"Surfaced because your monthly surplus grew by 67% and your debt-to-income is below 25%."*
        """)
        if st.button("Load Priya Sharma in Customer Portal"):
            st.session_state["selected_cid"] = "CUST_IND_1042"
            st.rerun()

    with sc2:
        st.subheader("Amit Patel — Small Merchant (Ahmedabad)")
        st.markdown("""
        - **Stress Situation:** High transaction volume, but medical expenses surged by **>60%** in the last 3 months, balance drifting to near zero.
        - **The Demonstration:**
          1. Propensity model generates an instant **₹2,00,000 Personal Loan**.
          2. The **Cross-Cutting Hard Veto Layer fires** instantly: Debt-to-income cap and stress threshold violated.
          3. Loan offer is **completely suppressed**, replaced with **30-Day Zero-Penalty EMI Grace Period & Health Cover**.
        - **Judge Takeaway:** Proves **decision discipline** and customer protection over predatory upselling.
        """)
        if st.button("Load Amit Patel in Customer Portal"):
            st.session_state["selected_cid"] = "CUST_IND_1088"
            st.rerun()

    with sc3:
        st.subheader("Sunita Devi — Artisan in Tier-2 City (Varanasi)")
        st.markdown("""
        - **Vernacular Journey:** Communicates in conversational Hindi.
        - **The Demonstration:**
          1. Sunita applies for a loan: dialogue state tracker deterministic validation (Aadhaar last-4 Verhoeff checksum algorithm, PAN regex).
          2. Mid-way through entering income, she asks: *"Cooling-off period kya hota hai?"*
          3. The **Intent Router** detours to Multilingual RAG, retrieves the exact clause from **RBI Digital Lending Guidelines**, and answers in Hindi with citations.
          4. Automatically resumes the pending loan slot without resetting user progress.
        """)
        if st.button("Load Sunita Devi in Customer Portal"):
            st.session_state["selected_cid"] = "CUST_IND_1002"
            st.rerun()


# ==============================================================================
# MODE 4: ARCHITECTURE & CONTRACT INSPECTOR
# ==============================================================================
elif app_mode == "🔍 Architecture & Contract Inspector":
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #F8FAFC; margin: 0; font-size: 26px;">Architecture & API Contract Inspector</h1>
        <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 15px;">
            Inspect live JSON payloads conforming to Contract 1, Contract 2, and Contract 3 in team_distribution.md.
        </p>
    </div>
    """, unsafe_allow_html=True)

    t_c1, t_c2, t_c3 = st.tabs(["Contract 1 (Person 1 -> 2, 4)", "Contract 2 (Person 2 -> 1, 3, 4)", "Contract 3 (Person 3 -> 4)"])

    with t_c1:
        st.markdown("#### Contract 1: Customer Feature Vector (Produced by Person 1)")
        st.json(contract1_vec.model_dump())

    with t_c2:
        st.markdown("#### Contract 2: Veto & Decision Layer Outcome (Produced by Person 2)")
        if veto_result:
            st.json(veto_result)
        else:
            st.info("Veto decision layer module is loading.")

    with t_c3:
        st.markdown("#### Contract 3: Conversational Chatbot Payload (Produced by Person 3)")
        sample_c3 = {
            "session_id": f"sess_{selected_cid}",
            "detected_language": "hi",
            "intent_category": "TASK_SLOT_FILLING",
            "bot_message": "Dhanyawad! Kripya apna 10-digit PAN number darj karein (jaise: ABCDE1234F).",
            "current_slot": "pan_number",
            "collected_slots": {"full_name": cust_row['name']},
            "is_journey_complete": False,
            "grounded_citation": None
        }
        st.json(sample_c3)
