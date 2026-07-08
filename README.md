# GeoVision-CLIP 🌍🛰️

**Satellite-based multimodal modeling of air quality in Cali, Colombia.**  
GeoVision-CLIP integrates **Sentinel-5P (NO₂, SO₂, O₃)** and **Sentinel-2 (NDVI, urban indices)** with **deep learning (CLIP, ConvLSTM)** and **geostatistics (spatio-temporal Kriging)** to estimate pollutant concentrations in unmonitored areas, delivering interactive maps with quantified uncertainty.

---

## 📐 Project Overview
- **Problem:** Limited and fragmented air quality monitoring in Cali (only 5/9 stations operational in 2023).  
- **Goal:** Generate high-resolution pollutant maps (NO₂, SO₂, O₃) using open satellite data + minimal ground validation.  
- **Approach:**  
  - Fine-tuned **CLIP** for satellite domain embeddings.  
  - **ConvLSTM** for temporal pollutant dynamics.  
  - **Spatio-temporal Kriging** for continuous surfaces + uncertainty maps.  
  - Cloud-native data lake (Bronze–Silver–Gold layers) for scalable ingestion and preprocessing.  

---

## ⚙️ Architecture
- **Data Ingestion:**  
  - Google Earth Engine (Sentinel-5P, Sentinel-2, MODIS, ERA5).  
  - Colombian open data (IDEAM, DAGMA).  
  - Pipelines in **Google Colab + Dask** for parallelized downloads (2020–2024).  

- **Storage:**  
  - **Wasabi Cloud Storage** with Medallion architecture.  
  - Bronze (raw), Silver (cleaned Zarr/CSV), Gold (interpolated + predictions).  

- **Preprocessing:**  
  - Cloud filters (`CLOUDY_PIXEL_PERCENTAGE < 60%`).  
  - Imputation strategies: linear interpolation, seasonal moving averages, SARIMA for long gaps.  
  - Spectral indices: NDVI, NDBI, MNDWI.  

---

## 🧠 Models
- **GeoCLIP + Sparse Autoencoder (SAE):** Multimodal embeddings combining spectral + textual/geographic features.  
- **ConvLSTM:** Captures spatio-temporal pollutant variability (T+1, T+3, T+7 horizons).  
- **Spatio-temporal Kriging:** Produces continuous pollutant surfaces with variance maps.  

---

## 📊 Results
- **Routing accuracy:** >99% for pollutant embeddings.  
- **Temporal coverage:** 5 years of fused satellite + meteorological data.  
- **Performance:** Seasonal pollutant cycles captured (O₃ peaks in summer, NO₂/SO₂ in colder months).  
- **Outputs:** Interactive maps with prediction + uncertainty layers.  

---

## 🖥️ Deployment
- **Training:** Lightning AI TPU/GPU acceleration.  
- **Serving:** Lightning AI service with GPU support.  
- **Visualization:** Web app with **Leaflet, Streamlit, Gradio**.  
- **Outputs:** GeoTIFF + CSV downloads, animated maps, diagnostic dashboards.  

---

## 🔧 For Developers
### Requirements
- Python 3.9+  
- Libraries: `torch`, `faiss`, `dask`, `earthengine-api`, `numpy`, `pandas`, `scikit-learn`, `pykrige`, `streamlit`, `gradio`, `leaflet`  

### Setup
```bash
# Clone repository
git clone https://github.com/ntlg72/project_geoCLIP.git
cd project_geoCLIP

# Install dependencies
pip install -r requirements.txt

# Authenticate Google Earth Engine
earthengine authenticate
```

### Run Training
```bash
python train_geoCLIP.py --epochs 50 --batch_size 64
```

### Launch Web App
```bash
python despliegue.py
```

---

## 📌 Applications
- Public health risk assessment in cities with incomplete monitoring.  
- Urban planning and environmental management.  
- Research on spatio-temporal pollutant dynamics using open satellite data.  

---

## 👩‍💻 Authors
- Michel Burgos Santos  
- Juan David Daza Rivera  
- Natalia López Gallego  
- Juan Andrés Ruiz Muñoz  
Universidad Autónoma de Occidente, Cali, Colombia
