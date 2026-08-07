"""
dashboard/pages/about.py — About MalnutriSense with SHAP explanation guide.
"""
 
import streamlit as st
 
st.set_page_config(page_title='MalnutriSense — About', page_icon='ℹ️')
st.title('ℹ️ About MalnutriSense')
 
st.markdown('''
## What does MalnutriSense do?
 
MalnutriSense predicts whether a child under 5 is at risk of three types of malnutrition:
- **Stunting** — the child is too short for their age (chronic malnutrition)
- **Underweight** — the child is too light for their age
- **Wasting** — the child is too light for their height (acute malnutrition)
 
## How to read the risk score
 
Each phenotype gets a **probability** between 0% and 100%:
- 🔴 **Above threshold** → AT RISK — refer for nutritional assessment
- 🟢 **Below threshold** → LOW RISK — continue routine monitoring
 
## What are SHAP features?
 
After each prediction, MalnutriSense shows the **top 3 factors** that influenced the risk score:
- **Red bars** (positive SHAP) = factors that INCREASE malnutrition risk
- **Green bars** (negative SHAP) = factors that PROTECT against malnutrition
 
For example, if the top factor is "wealth_quintile = 1" with a red bar,
it means being in the poorest wealth group is the main driver of this child's risk.
 
## What is the equity flag?
 
Children from the **poorest two wealth quintiles** or **Scheduled Tribe** backgrounds
receive a lower classification threshold. This means the model is more sensitive
for these groups — it catches more at-risk children even at the cost of some false alarms.
This ensures the screening system does not miss the most vulnerable children.
 
## Data source
 
The model is trained on NFHS-5 (National Family Health Survey 2019-21) data
covering 232,920 children across all 36 states and union territories of India.
 
---
*MalnutriSense v1.0 · IEEE CS Bangalore Chapter Internship 2026*
''')
