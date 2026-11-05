# Micronutrient Estimator: Kolhapur and Sangli

Estimates whether a farm soil sample is **deficient or sufficient in sulphur (S), zinc (Zn) and iron (Fe)**,
using only its routine basic soil test (N, P, K, organic carbon, pH, EC) plus satellite and soil-map data.

Micronutrient lab tests cost more and take longer than the basic panel, so many samples never get them.
This project checks how far the basic test and freely available satellite data can stand in for them,
and uses the answer to flag which samples most need a full micronutrient test.

## Results

All results come from village-held-out testing: each sample is predicted by a model that never saw its village.
The data is 2,078 Soil Health Card samples from 273 villages (2023-24).

| Model | Sulphur | Zinc | Iron |
| --- | ---: | ---: | ---: |
| RF Regressor + cutoff | 76.18% | 80.22% | 67.28% |
| Tuned RF Classifier | 83.01% | 81.23% | 69.63% |
| CatBoost Classifier | 83.11% | 80.80% | 70.02% |

These are deficiency classification accuracies. Lab readings below 10 mg/kg for S, 0.6 mg/kg for Zn and 4.5 mg/kg for Fe count as deficient. The Model accuracy tab also reports the majority-class baseline and the original RF regression R² scores.

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
2. **Predictions**: A binary CatBoost classifier per nutrient estimates the deficiency score from 21 inputs. Scores above 0.5 indicate deficient. Each card shows the status, the model's confidence in it (flagged as low when the score is between 0.35 and 0.65) and the measured village-held-out accuracy. Lab measurements are in mg/kg.
3. **Validation**: Five folds hold entire villages out of training. Field check and the Estimated map use the saved out-of-fold CatBoost scores. Custom values use the CatBoost models that held out the selected village, so the original inputs reproduce that field's scores. Editing a field's inputs does not alter the saved field or map.
4. **Comparison**: The original RF regressor predicts concentrations and applies the lab cutoffs. The tuned RF classifier predicts status directly. Eight RF settings compete on three grouped folds inside each outer training split. CatBoost uses 400 iterations, depth 5, and a learning rate of 0.05.

N, P and K are in kg/ha and micronutrients in mg/kg. These are the units the Soil Health Card itself uses.

## Run locally

Tested with Python 3.12. Extract the ZIP, open the directory containing `app.py`, and run:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For the existing Streamlit deployment, copy the **contents** of this project folder into the current app directory in the connected repository. Commit the updated source, requirements, data and all model files. Keep its existing entry point targeting `app.py`. The packaged models are ready for inference on first launch.

## Rebuild the models (optional)

From the project root, regenerate the classifiers, held-out scores and comparison with the included data:

```bash
python train.py --jobs 4
```

To rebuild the original RF concentration models and satellite-only regression benchmarks first, supply the original raw `AEZ_6.csv` and `AEZ_19.csv`:

```bash
python train.py --shc path/to/shc_data/AEZS/AGRI_2023-24 --jobs 4
```

## Project structure

```
app.py                Streamlit dashboard
train.py              original data preparation and training entry point
classifier_study.py   grouped CatBoost training and nested RF evaluation
data/fields.csv       inputs, lab values, RF estimates, village folds and CatBoost scores
data/metrics.json     model comparison and RF diagnostics
models/               original RF regressors and CatBoost models for the five village folds
.streamlit/           app theme
```

## Limitations

- Covers one season and two districts only. It has not been tested elsewhere.
- Sample coordinates come from the Soil Health Card portal. Several samples in a village often share one
  location, so estimates are per sample point, not per surveyed field.
- Copper, manganese and boron are not modelled, because almost no samples are deficient in them.
- It screens samples; it does not replace a lab test. Samples flagged as deficient should be confirmed in the lab.

## Data source

Soil Health Card laboratory values come from the Government of India's
[Soil Health Card programme](https://soilhealth.dac.gov.in). The georeferenced samples and pre-extracted
satellite features come from the public
[Soil-Health-Mapping-PanIndia](https://github.com/Singhratinder/Soil-Health-Mapping-PanIndia) dataset.
