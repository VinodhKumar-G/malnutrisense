"""
dashboard/app.py — Streamlit frontend for MalnutriSense.
 
Run: streamlit run dashboard/app.py --server.port 8501
 
Requires FastAPI to be running at localhost:8000.
"""
 
import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
API_URL = 'http://localhost:8000'
 
st.set_page_config(
    page_title='MalnutriSense — Child Malnutrition Risk Screening',
    page_icon='🏥',
    layout='wide',
)
 
st.title('🏥 MalnutriSense')
st.subheader('Child Malnutrition Risk Prediction for ASHA Workers')

# ── API availability check ────────────────────────────────────────────────
try:
    _health = requests.get(f'{API_URL}/health', timeout=3)
    _health.raise_for_status()
except Exception:
    st.error(
        f'⚠️ Cannot reach the MalnutriSense API at {API_URL}. '
        "Start it with `uvicorn api.main:app --host 0.0.0.0 --port 8000` "
        '(or `bash scripts/start_services.sh`) before using Predict.'
    )

st.markdown('---')
 
# ── Input form ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
 
with col1:
    st.markdown('**Child Information**')
    age_months  = st.slider('Child Age (months)', 0, 59, 12)
    sex         = st.selectbox('Sex', ['male', 'female'])
    diarrhoea   = st.selectbox('Diarrhoea in last 2 weeks?', [0, 1],
                                format_func=lambda x: 'No' if x==0 else 'Yes')
    birth_wt    = st.number_input('Birth Weight (grams)', 500, 5000, 3000)
    breastfeed  = st.slider('Breastfeeding Duration (months)', 0, 24, 6)
 
with col2:
    st.markdown('**Household Information**')
    wealth      = st.selectbox('Wealth Quintile',
                                [1,2,3,4,5],
                                format_func=lambda x:
                                {1:'1 — Poorest',2:'2 — Poor',3:'3 — Middle',
                                 4:'4 — Rich',5:'5 — Richest'}[x])
    edu         = st.selectbox('Mother\'s Education',
                                ['no_education','primary','secondary','higher'])
    residence   = st.selectbox('Residence', ['rural','urban'])
 
with col3:
    st.markdown('**WASH (Water, Sanitation)**')
    water       = st.selectbox('Water Source',
                                ['piped_on_premises','tube_well','protected_well',
                                 'unprotected_well','surface_water'])
    toilet      = st.selectbox('Toilet Type',
                                ['flush_piped','pit_with_slab','pit_without_slab',
                                 'other'])
 
# ── Predict button ────────────────────────────────────────────────────────
st.markdown('---')
if st.button('🔍  Predict Malnutrition Risk', type='primary', use_container_width=True):
 
    payload = {
        'age_months': age_months, 'sex': sex,
        'wealth_quintile': wealth, 'mother_education': edu,
        'water_source': water, 'toilet_type': toilet,
        'diarrhoea_2weeks': diarrhoea, 'birth_weight_g': birth_wt,
        'breastfeed_months': breastfeed, 'residence': residence,
    }
 
    with st.spinner('Analysing risk factors...'):
        try:
            resp = requests.post(f'{API_URL}/explain', json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            st.error(f'API error: {e}')
            st.stop()
 
    # ── Risk score cards ─────────────────────────────────────────────────
    st.markdown('### Risk Assessment')
    risk_col1, risk_col2, risk_col3 = st.columns(3)
 
    risk_labels = [
        ('stunted',     'Stunting Risk\n(Height-for-Age)', risk_col1),
        ('underweight', 'Underweight Risk\n(Weight-for-Age)', risk_col2),
        ('wasted',      'Wasting Risk\n(Weight-for-Height)', risk_col3),
    ]
 
    for label, title, col in risk_labels:
        with col:
            data  = result[label]
            prob  = data['probability']
            pred  = data['prediction']
            color = '🔴' if pred == 1 else '🟢'
            risk_level = 'AT RISK' if pred == 1 else 'LOW RISK'
            st.metric(
                label=title,
                value=f'{prob:.1%}',
                delta=f'{color} {risk_level}',
            )
 
    # ── Overall risk banner ───────────────────────────────────────────────
    overall = result.get('overall_risk','unknown')
    if overall == 'high':
        st.error('⚠️ HIGH OVERALL RISK — Refer for nutritional assessment immediately')
    elif overall == 'medium':
        st.warning('⚠️ MEDIUM RISK — Schedule follow-up within 2 weeks')
    else:
        st.success('✅ LOW RISK — Continue routine monitoring')
 
    # ── SHAP explanation bar chart ────────────────────────────────────────
    shap_features = result.get('top_shap_features', [])
    if shap_features:
        st.markdown('### 🔍 Key Risk Factors (Top 3)')
        st.caption('Positive values push toward malnutrition risk. Negative values protect against it.')
 
        fig, ax = plt.subplots(figsize=(8, 3))
        names  = [f['feature'] for f in shap_features]
        values = [f['shap_value'] for f in shap_features]
        colors = ['#E53935' if v > 0 else '#43A047' for v in values]
 
        ax.barh(names, values, color=colors, edgecolor='white', height=0.6)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on prediction)')
        ax.set_title('Feature Contributions to Risk Score')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    # ── Equity flag ───────────────────────────────────────────────────────
    if result.get('equity_flag'):
        st.info(f'🏷️ Equity note: {result["equity_reason"]}')
 
# ── Footer ────────────────────────────────────────────────────────────────
st.markdown('---')
st.caption('MalnutriSense v1.0 · IEEE CS Bangalore Chapter Internship 2026 · NFHS-5 Data')
