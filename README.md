# Micronutrient Estimator: Kolhapur and Sangli

Estimates whether a farm soil sample is **deficient or sufficient in sulphur (S), zinc (Zn) and iron (Fe)**,
using only its routine basic soil test (N, P, K, organic carbon, pH, EC) plus satellite and soil-map data.

Micronutrient lab tests cost more and take longer than the basic panel, so many samples never get them.
This project checks how far the basic test and freely available satellite data can stand in for them,
and uses the answer to flag which samples most need a full micronutrient test.

## Results

All results come from village-held-out testing: each sample is predicted by a model that never saw its village.
The data is 2,078 Soil Health Card samples from 273 villages (2023-24).

| Nutrient | Deficient below | Status correct | Rule-based baseline | R² (basic test + satellite) | R² (satellite only) |
| --- | --- | --- | --- | --- | --- |
| Sulphur (S) | 10 mg/kg | 76% | 64% | 0.69 | 0.48 |
| Zinc (Zn) | 0.6 mg/kg | 80% | 56% | 0.41 | 0.24 |
| Iron (Fe) | 4.5 mg/kg | 67% | 51% | 0.19 | 0.11 |

The **rule-based baseline** gives every sample the most common status. The model beats it for all three
nutrients. Zinc and sulphur are useful for screening; iron is not yet reliable.

## The app

The Streamlit dashboard has three tabs:

- **Field check**: pick a district, village and sample. It shows the sample's basic soil test,
  its location, and the estimated status for S, Zn and Fe next to the lab-measured values.
- **Try your own values**: enter basic soil-test values and get the estimated status straight away.
- **Model accuracy**: the results table above and a deficiency map of all samples, lab-measured or estimated.

## How it works

1. **Inputs (21 per sample)**
   - Basic soil test: N, P, K (kg/ha), organic carbon (%), pH, EC (dS/m).
   - Remote sensing: seasonal NDVI and NIRv for Kharif, Rabi and Zaid, plus salinity (SI) and
     grain-size (TGSI) indices, all from Landsat 8/9. Also land-surface temperature (MODIS),
     rainfall (CHIRPS), and sand, silt and pH maps (SoilGrids).
2. **Model**: one random forest per nutrient, with 200 trees.
3. **Validation**: 5-fold cross-validation grouped by village, so no village appears in both training and testing.
4. **Status**: the estimated concentration (mg/kg) is compared with the Soil Health Card critical limit.

N, P and K are in kg/ha and micronutrients in mg/kg. These are the units the Soil Health Card itself uses.

## Run locally

Requires Python 3.12 or newer.

```bash
git clone https://github.com/VVarad/soil-health-demo.git
cd soil-health-demo
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Retrain the models (optional)

The repository already includes the processed data and trained models. To rebuild them you need the raw
agro-ecological-zone CSVs (`AEZ_6.csv` and `AEZ_19.csv`) from the data source below:

```bash
python train.py --shc path/to/shc_data/AEZS/AGRI_2023-24
```

This rewrites `data/fields.csv`, `data/metrics.json` and `models/*.joblib`.

## Project structure

```
app.py               Streamlit dashboard
train.py             data preparation, training and evaluation
data/fields.csv      one row per sample: location, inputs, lab values, out-of-fold estimates
data/metrics.json    accuracy summary shown in the app
models/              trained random forests for S, Zn and Fe
.streamlit/          app theme
```

## Limitations

- Covers one season (2023-24) and two districts only. It has not been tested elsewhere.
- Sample coordinates come from the Soil Health Card portal. Several samples in a village often share one
  location, so estimates are per sample point, not per surveyed field.
- Copper, manganese and boron are not modelled, because almost no samples are deficient in them.
- It screens samples; it does not replace a lab test. Samples flagged as deficient should be confirmed in the lab.

## Data source

Soil Health Card laboratory values come from the Government of India's
[Soil Health Card programme](https://soilhealth.dac.gov.in). The georeferenced samples and pre-extracted
satellite features come from the public
[Soil-Health-Mapping-PanIndia](https://github.com/Singhratinder/Soil-Health-Mapping-PanIndia) dataset.
