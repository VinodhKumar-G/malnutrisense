"""
dashboard/pages/batch_predict.py — Batch prediction via CSV upload.
 
Allows ASHA supervisor to upload a CSV of children and get all risk scores at once.
Displays results as a sortable table with colour-coded risk columns.
"""
 
import streamlit as st
import requests
import pandas as pd
import io
 
API_URL = 'http://localhost:8000'
 
st.set_page_config(page_title='MalnutriSense — Batch Prediction', page_icon='📊')
st.title('📊 Batch Prediction — Upload CSV')
st.markdown('Upload a CSV file with child data. Each row = one child.')
 
# Required columns for the CSV
REQUIRED_COLS = ['age_months','sex','wealth_quintile','mother_education',
                 'water_source','toilet_type','diarrhoea_2weeks']
 
uploaded = st.file_uploader('Choose a CSV file', type=['csv'])
 
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f'Uploaded: {len(df)} children')
 
    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f'Missing columns: {missing}')
        st.stop()
 
    st.dataframe(df.head(5))
 
    if st.button('🔍 Predict All', type='primary', use_container_width=True):
        results = []
        progress = st.progress(0)
        status = st.empty()
 
        for i, row in df.iterrows():
            payload = {
                'age_months':       int(row.get('age_months', 12)),
                'sex':              str(row.get('sex', 'male')),
                'wealth_quintile':  int(row.get('wealth_quintile', 3)),
                'mother_education': str(row.get('mother_education', 'primary')),
                'water_source':     str(row.get('water_source', 'tube_well')),
                'toilet_type':      str(row.get('toilet_type', 'pit_with_slab')),
                'diarrhoea_2weeks': int(row.get('diarrhoea_2weeks', 0)),
                'residence':        str(row.get('residence', 'rural')),
            }
            try:
                resp = requests.post(f'{API_URL}/predict', json=payload, timeout=10)
                result = resp.json()
                results.append({
                    'child_index':       i,
                    'stunting_prob':      result['stunted']['probability'],
                    'stunting_flag':      result['stunted']['prediction'],
                    'underweight_prob':   result['underweight']['probability'],
                    'underweight_flag':   result['underweight']['prediction'],
                    'wasting_prob':       result['wasted']['probability'],
                    'wasting_flag':       result['wasted']['prediction'],
                    'overall_risk':       result['overall_risk'],
                    'equity_flag':        result.get('equity_flag', False),
                })
            except Exception as e:
                results.append({'child_index': i, 'error': str(e)})
 
            progress.progress((i+1) / len(df))
            status.text(f'Processing child {i+1} of {len(df)}')
 
        result_df = pd.DataFrame(results)
 
        # Colour-code risk levels
        st.markdown('### Results')
        high_risk = result_df[result_df['overall_risk']=='high']
        st.warning(f'⚠️ {len(high_risk)} children identified as HIGH RISK')
        st.dataframe(result_df)
 
        # Download button
        csv_out = result_df.to_csv(index=False)
        st.download_button(
            label='📥 Download Results CSV',
            data=csv_out,
            file_name='malnutrisense_batch_results.csv',
            mime='text/csv',
        )
