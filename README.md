# Soil Health Demo: Kolhapur and Sangli

Live Streamlit demo that predicts soil nitrogen, phosphorus, potassium and organic carbon
from satellite-derived features, for 2,078 Soil Health Card samples in Kolhapur and Sangli (2023-24).

Run locally: `pip install -r requirements.txt && streamlit run app.py`

Data and method come from https://github.com/Singhratinder/Soil-Health-Mapping-PanIndia
(Soil Health Card lab values plus satellite features). Predictions on unseen villages are
only reliable for phosphorus; N, K and organic carbon are no better than a majority-class guess.
