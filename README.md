# GeoVision-CLIP Cali — Final Project 🌍🛰️

**Estimation of air pollution in Cali, Colombia using multimodal contrastive learning (Sentinel-2 + Sentinel-5P + text) with Sparse Autoencoders on ViT-B/32.**  
This project combines **deep learning (CLIP, ConvLSTM)** and **geostatistics (ST-Kriging)** to generate high-resolution maps of NO₂, SO₂, and O₃ in unmonitored areas, with quantified uncertainty.

---

## 📐 Objectives
- Address the lack of continuous monitoring stations in Cali.  
- Integrate satellite data (Sentinel-2, Sentinel-5P, MODIS) with local ground data (IDEAM, DAGMA).  
- Train a multimodal model (GeoVision-CLIP) to learn joint representations of imagery, gas columns, and text.  
- Generate spatio-temporal forecasts with ConvLSTM + ST-Kriging.  
- Provide an interactive platform with prediction and uncertainty maps.  

**Target KPIs:** Recall@1 ≥ 0.45 · Recall@5 ≥ 0.70 · Sparsity ≥ 0.70  

---

## 📂 Repository Structure

```
GeoVision-CLIP-Cali/
│
├── ProyectoFinal_GeoVisionCLIP_Cali.pdf   # Project report
│
├── situacion_1_eda/                       # Situation 1 — EDA and station data
│   ├── EDA_MODIs.ipynb
│   ├── EDA_dagma_001.ipynb
│   ├── EDA_ground_truth.ipynb
│   ├── eda_sisaire.ipynb
│   ├── tablas/                            # Exported tables
│   └── README_SIT1.md
│
├── situacion_2_clip/                      # Situation 2 — GeoVision-CLIP model
│   ├── clip_sae.ipynb                     # Main notebook (train here)
│   ├── remoteclip_inference_cali.ipynb
│   └── psychometric_validation_embeddings.ipynb
│
├── situacion_3_kriging/                   # Situation 3 — ConvLSTM + ST-Kriging
│   └── convlstm_geovision.ipynb
│
├── figuras/                               # Figures
│   ├── sit1/
│   └── sit2/
│
├── data/                                  # Processed data
│   ├── silver/                            # Parquets (SISAIRE, DAGMA, S5P)
│   └── cache/                             # Tiles .npy and df_pares.parquet
│
├── models/                                # Trained models
│   ├── RemoteCLIP-ViT-B-32.pt
│   ├── RemoteCLIP-ViT-B-32-cali-finetuned.pt
│   ├── embeddings/
│   └── remoteclip_outputs/
│
├── scripts/                               # Pipelines
│   └── build_gold_normalized.py
│
└── backups/                               # Notebook backups
    ├── clip_sae_backup.ipynb
    └── clip_sae_backup_prepenalty.ipynb
```

---

## 🧠 GeoVision-CLIP Architecture (Situation 2)

```
S2 (11ch tile) → PatchEmbed → ViT×4 frozen → CLS(768) ──┐
                                                           ├─ LateFusion ─→ SAE(768→512) ─→ e_img ∈ ℝ^256
S5P (8 vars)   → PatchEmbed → ViT×4        → CLS(768) ──┘                    ↕ InfoNCE
                                                                         e_txt ∈ ℝ^256
Text (class + location) → XLM-RoBERTa → SAE(384→512) ────────────────────────┘

L_total = L_InfoNCE + 0.1·(L_sae_img + L_sae_txt),   λ=1e-3
```

---

## ⚙️ Main Dependencies

```
torch, open_clip_torch, sentence-transformers
s3fs, zarr, xarray, boto3
scikit-learn, pykriging (Situation 3)
streamlit, gradio, leaflet
```

---

## 🔧 Installation & Usage

### Requirements
- Python 3.9+  
- Google Earth Engine account (for satellite data ingestion).  
- Wasabi Cloud or S3-compatible storage for data lake.  

### Setup
```bash
# Clone repository
git clone https://github.com/ntlg72/project_geoCLIP.git
cd project_geoCLIP-Cali

# Install dependencies
pip install -r requirements.txt

# Authenticate Google Earth Engine
earthengine authenticate
```

### Training
```bash
# Train GeoVision-CLIP with SAE
python situacion_2_clip/clip_sae.ipynb
```

### Forecasting
```bash
# Run ConvLSTM + Kriging
python situacion_3_kriging/convlstm_geovision.ipynb
```

### Visualization
```bash
# Launch interactive app
python despliegue.py
```

---

## 📊 Results
- Achieved **Recall@1 ≥ 0.45**, **Recall@5 ≥ 0.70**, **Sparsity ≥ 0.70** in validation.  
- Prediction maps for NO₂, SO₂, and O₃ with spatial uncertainty.  
- Forecasting of pollutant time series with ConvLSTM + ST-Kriging.  
- Interactive web platform with Leaflet, Streamlit, and Gradio.  

---

## 📌 Applications
- Public health risk assessment in cities with incomplete monitoring networks.  
- Urban planning and environmental management.  
- Research on spatio-temporal pollutant dynamics using open satellite data.  

---

## 👩‍💻 Authors
- Michel Burgos Santos  
- Juan David Daza Rivera  
- Natalia López Gallego  
- Juan Andrés Ruiz Muñoz  
Universidad Autónoma de Occidente, Cali, Colombia


Would you like me to also add an **ASCII pipeline diagram** (data ingestion → storage → training → prediction → visualization) at the end to make the workflow even clearer?
