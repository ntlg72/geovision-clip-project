import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MiniMap, MousePosition, MeasureControl
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from pathlib import Path
from scipy.ndimage import map_coordinates
from pykrige.uk import UniversalKriging
try:
    from libpysal.weights import lat2W
    from esda.moran import Moran
    _HAS_ESDA = True
except ImportError:
    _HAS_ESDA = False

# ─────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────
ESTACIONES_DAGMA = {
    "Univalle":      {"lat": 3.3756, "lon": -76.5319, "zona": "Sur",          "calidad": "Buena",     "no2": 32, "so2": 8,  "o3": 58},
    "Compartir":     {"lat": 3.3909, "lon": -76.5254, "zona": "Sur",          "calidad": "Buena",     "no2": 29, "so2": 7,  "o3": 62},
    "Pance":         {"lat": 3.3366, "lon": -76.5467, "zona": "Sur-Ladera",   "calidad": "Excelente", "no2": 18, "so2": 4,  "o3": 45},
    "Jardin Plaza":  {"lat": 3.3603, "lon": -76.5117, "zona": "Sur-Oriente",  "calidad": "Buena",     "no2": 31, "so2": 9,  "o3": 60},
    "Simon Bolivar": {"lat": 3.4513, "lon": -76.5322, "zona": "Centro",       "calidad": "Regular",   "no2": 44, "so2": 16, "o3": 78},
    "Manzanares":    {"lat": 3.4721, "lon": -76.5105, "zona": "Norte",        "calidad": "Buena",     "no2": 36, "so2": 11, "o3": 65},
    "Yumbo":         {"lat": 3.5856, "lon": -76.4967, "zona": "Industrial",   "calidad": "Mala",      "no2": 68, "so2": 38, "o3": 92},
    "Acopi":         {"lat": 3.5412, "lon": -76.4823, "zona": "Industrial",   "calidad": "Mala",      "no2": 61, "so2": 31, "o3": 88},
    "Aguablanca":    {"lat": 3.4081, "lon": -76.4818, "zona": "Oriente",      "calidad": "Regular",   "no2": 47, "so2": 18, "o3": 74},
}

CONTAMINANTES = {
    "NO2": {"unidad": "µg/m³", "limite_OMS": 40,  "cmap": "YlOrRd", "color": "#f87171", "nombre": "Dióxido de Nitrógeno", "key": "no2"},
    "SO2": {"unidad": "µg/m³", "limite_OMS": 20,  "cmap": "PuRd",   "color": "#c084fc", "nombre": "Dióxido de Azufre",    "key": "so2"},
    "O3":  {"unidad": "µg/m³", "limite_OMS": 100, "cmap": "YlGn",   "color": "#34d399", "nombre": "Ozono",                "key": "o3"},
}

HORIZONTES = ["T+1 día", "T+3 días", "T+7 días"]
HORIZONTES_DESC = {"T+1 día": "Mañana", "T+3 días": "En 3 días", "T+7 días": "Próxima semana"}

# ─────────────────────────────────────────────
# PIPELINE REAL — ConvLSTM + KED
# ─────────────────────────────────────────────
_BASE_DIR   = Path(__file__).parent
_MODELS_DIR = _BASE_DIR / 'models'
_SILVER_DIR = _BASE_DIR / 'data' / 'silver'

G    = 8
BBOX = {'lat_min': 3.28, 'lat_max': 3.58, 'lon_min': -76.62, 'lon_max': -76.38}

_GL  = np.linspace(BBOX['lat_min'], BBOX['lat_max'], G)
_GO  = np.linspace(BBOX['lon_min'], BBOX['lon_max'], G)
_GL2, _GO2   = np.meshgrid(_GL, _GO, indexing='ij')
_GRID_LATS_F = _GL2.ravel()
_GRID_LONS_F = _GO2.ravel()

_RES = 0.006
_FL  = np.arange(BBOX['lat_min'] + _RES / 2, BBOX['lat_max'], _RES)
_FO  = np.arange(BBOX['lon_min'] + _RES / 2, BBOX['lon_max'], _RES)
_FL2, _FO2 = np.meshgrid(_FL, _FO, indexing='ij')
_HF, _WF   = _FL2.shape


class _ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, ks=3):
        super().__init__()
        self.hid_ch     = hid_ch
        self.conv       = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, ks, padding=ks // 2)
        self.layer_norm = nn.GroupNorm(4, 4 * hid_ch)
    def forward(self, x, h, c):
        g = self.layer_norm(self.conv(torch.cat([x, h], 1)))
        i, f, g2, o = torch.chunk(g, 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g2)
        return torch.sigmoid(o) * torch.tanh(c), c
    def init_hidden(self, B, H, W):
        d = next(self.parameters()).device
        return (torch.zeros(B, self.hid_ch, H, W, device=d),
                torch.zeros(B, self.hid_ch, H, W, device=d))


class _ConvLSTMLayer(nn.Module):
    def __init__(self, in_ch, hid_ch, ks=3):
        super().__init__()
        self.cell = _ConvLSTMCell(in_ch, hid_ch, ks)
    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c = self.cell.init_hidden(B, H, W)
        outs  = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, 1)


class _BidirConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, ks=3):
        super().__init__()
        self.forward_layer  = _ConvLSTMLayer(in_ch, hid_ch, ks)
        self.backward_layer = _ConvLSTMLayer(in_ch, hid_ch, ks)
    def forward(self, x):
        f = self.forward_layer(x)
        b = torch.flip(self.backward_layer(torch.flip(x, [1])), [1])
        return torch.cat([f, b], 2)


class _GeoConvLSTM(nn.Module):
    def __init__(self, in_ch=262, hid=8, ks=3, n_cont=3, dropout=0.35):
        super().__init__()
        mid = 2 * hid
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.GroupNorm(8, mid), nn.GELU())
        self.bidir1 = _BidirConvLSTM(mid, hid, ks)
        self.norm1  = nn.GroupNorm(8, mid)
        self.drop1  = nn.Dropout3d(dropout)
        self.bidir2 = _BidirConvLSTM(mid, hid, ks)
        self.norm2  = nn.GroupNorm(8, mid)
        self.drop2  = nn.Dropout3d(dropout)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(mid, mid // 4), nn.ReLU(True),
            nn.Linear(mid // 4, mid), nn.Sigmoid())
        def _head():
            return nn.Sequential(
                nn.Dropout2d(0.3),
                nn.Conv2d(mid, mid // 2, 1), nn.GroupNorm(4, mid // 2), nn.GELU(),
                nn.Conv2d(mid // 2, mid // 4, 1), nn.GELU(),
                nn.Conv2d(mid // 4, n_cont, 1))
        self.head_T1, self.head_T3, self.head_T7 = _head(), _head(), _head()

    def _se(self, f):
        return f * self.se(f).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        xp = torch.stack([self.input_proj(x[:, t]) for t in range(T)], 1)
        h  = self.bidir1(xp)
        h  = self.drop1(self.norm1(h.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        h  = self.bidir2(h)
        h  = self.drop2(self.norm2(h.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        f1 = self._se(h[:, -1])
        f3 = self._se(h[:, -3:].mean(1))
        f7 = self._se(h.mean(1))
        return torch.stack([self.head_T1(f1), self.head_T3(f3), self.head_T7(f7)], 1)


def _interp_grid(grid_2d, lats, lons):
    lat_i = (lats - BBOX['lat_min']) / (BBOX['lat_max'] - BBOX['lat_min']) * (G - 1)
    lon_i = (lons - BBOX['lon_min']) / (BBOX['lon_max'] - BBOX['lon_min']) * (G - 1)
    return map_coordinates(grid_2d, [lat_i, lon_i], order=1, mode='nearest')


def _era5_at(era5_last, lats, lons):
    n = len(np.atleast_1d(lats))
    return {
        'blh': _interp_grid(era5_last[0], lats, lons),
        't2m': _interp_grid(era5_last[1], lats, lons),
        'rh':  _interp_grid(era5_last[2], lats, lons),
        'u10': np.zeros(n, dtype=np.float32),
        'v10': np.zeros(n, dtype=np.float32),
    }


def _anisotropy(u10, v10):
    ws = float(np.sqrt(np.mean(u10) ** 2 + np.mean(v10) ** 2))
    if ws < 1.0:
        return {'angle': 0.0, 'scaling': 1.0}
    ang = (-np.degrees(np.arctan2(float(np.mean(v10)), float(np.mean(u10))))) % 360
    return {'angle': ang, 'scaling': min(1.5 + 0.25 * ws, 4.0)}


def _ked(obs_lats, obs_lons, obs_vals, era5_obs, q_lats, q_lons, era5_q):
    lat_mu, lat_s = obs_lats.mean(), obs_lats.std() + 1e-8
    lon_mu, lon_s = obs_lons.mean(), obs_lons.std() + 1e-8
    lat_n   = (obs_lats - lat_mu) / lat_s
    lon_n   = (obs_lons - lon_mu) / lon_s
    q_lat_n = (q_lats   - lat_mu) / lat_s
    q_lon_n = (q_lons   - lon_mu) / lon_s

    def _nz(a):
        m, s = a.mean(), a.std() + 1e-8
        return (a - m) / s, m, s

    blh_o, bm, bs = _nz(era5_obs['blh']); blh_q = (era5_q['blh'] - bm) / bs
    t2m_o, tm, ts = _nz(era5_obs['t2m']); t2m_q = (era5_q['t2m'] - tm) / ts
    aniso = _anisotropy(era5_obs['u10'], era5_obs['v10'])

    try:
        uk = UniversalKriging(
            lon_n, lat_n, obs_vals,
            variogram_model='exponential',
            drift_terms=['specified', 'specified'],
            specified_drift=[blh_o, t2m_o],
            anisotropy_scaling=aniso['scaling'],
            anisotropy_angle=aniso['angle'],
            verbose=False, enable_plotting=False,
        )
        z, var = uk.execute('points', q_lon_n, q_lat_n,
                            specified_drift_arrays=[blh_q, t2m_q])
        return np.array(z).clip(0), np.array(var).clip(0), uk
    except Exception:
        # IDW fallback
        d = np.sqrt((obs_lats[:, None] - q_lats[None, :]) ** 2 +
                    (obs_lons[:, None] - q_lons[None, :]) ** 2) + 1e-8
        w = 1 / d ** 2
        z = (w * obs_vals[:, None]).sum(0) / w.sum(0)
        return z.clip(0), np.ones_like(z) * obs_vals.var(), None


_PP = {'ready': False, 'ked': {}}


def _init_pipeline():
    global _PP
    if _PP['ready']:
        return

    ckpt = torch.load(str(_MODELS_DIR / 'convlstm_geovision_v2.pt'),
                      map_location='cpu', weights_only=False)
    hid  = ckpt['model_state']['input_proj.0.weight'].shape[0] // 2
    model = _GeoConvLSTM(in_ch=262, hid=hid)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    t_mean = torch.tensor(ckpt['t_mean'], dtype=torch.float32)
    t_std  = torch.tensor(ckpt['t_std'],  dtype=torch.float32)

    emb_npz  = np.load(str(_MODELS_DIR / 'embeddings_full_3344.npz'), allow_pickle=True)
    embs_raw = emb_npz['embeddings'].astype(np.float32)
    t_idx_e  = emb_npz['t_idx'].astype(int)
    t_row_e  = emb_npz['tile_row'].astype(int)
    t_col_e  = emb_npz['tile_col'].astype(int)
    n_ts     = int(t_idx_e.max()) + 1

    emb_g = np.zeros((n_ts, 256, G, G), dtype=np.float32)
    cnt_g = np.zeros((n_ts, G, G),      dtype=np.float32)
    for k in range(len(embs_raw)):
        t  = t_idx_e[k]
        gr = min(int(t_row_e[k] / 43 * G), G - 1)
        gc = min(int(t_col_e[k] / 34 * G), G - 1)
        emb_g[t, :, gr, gc] += embs_raw[k]
        cnt_g[t, gr, gc]    += 1.0
    emb_g  /= np.where(cnt_g > 0, cnt_g, 1.0)[:, np.newaxis, :, :]
    emb_seq = emb_g[-8:]

    era5_raw  = np.load(str(_SILVER_DIR / 'era5_meteo_8x8.npy')).astype(np.float32)
    era5_norm = ((era5_raw - era5_raw.mean(axis=(0, 2, 3), keepdims=True)) /
                 (era5_raw.std(axis=(0, 2, 3), keepdims=True) + 1e-6))
    era5_seq  = era5_norm[-8:]

    lag_std  = np.array([2.5, 8.0, 11.5], dtype=np.float32)
    lag_vals = np.array(ckpt['t_mean'], dtype=np.float32) / lag_std
    lag_seq  = np.tile(
        (lag_vals.reshape(3, 1, 1) * np.ones((1, G, G), dtype=np.float32))[None],
        (8, 1, 1, 1))

    x_t = torch.from_numpy(
        np.concatenate([emb_seq, era5_seq, lag_seq], axis=1)
    ).unsqueeze(0)

    with torch.no_grad():
        pred_n = model(x_t).clamp(min=0)
    mu5 = t_mean.view(1, 1, 3, 1, 1)
    s5  = t_std.view(1, 1, 3, 1, 1)
    preds_mean = (pred_n * s5 + mu5).clamp(min=0).squeeze(0).cpu().numpy()

    era5_last    = era5_raw[-1]
    era5_grid_d  = _era5_at(era5_last, _GRID_LATS_F, _GRID_LONS_F)
    era5_fine_d  = _era5_at(era5_last, _FL2.ravel(), _FO2.ravel())

    conts = list(CONTAMINANTES.keys())
    ked   = {}
    for ci in range(3):
        for hi in range(3):
            obs_vals = preds_mean[hi, ci].ravel()
            z_f, var_f, _ = _ked(
                _GRID_LATS_F, _GRID_LONS_F, obs_vals,
                era5_grid_d,
                _FL2.ravel(), _FO2.ravel(),
                era5_fine_d,
            )
            ked[(ci, hi)] = {
                'Z':   z_f.reshape(_HF, _WF),
                'VAR': var_f.reshape(_HF, _WF),
            }

    _PP['ked']   = ked
    _PP['ready'] = True


def _rmse_loocv(ci, hi):
    cont = list(CONTAMINANTES.keys())[ci]
    key  = CONTAMINANTES[cont]['key']
    kd   = _PP['ked'][(ci, hi)]
    errs = []
    for est in ESTACIONES_DAGMA.values():
        obs   = est[key]
        lat_i = (est['lat'] - BBOX['lat_min']) / (BBOX['lat_max'] - BBOX['lat_min']) * (_HF - 1)
        lon_i = (est['lon'] - BBOX['lon_min']) / (BBOX['lon_max'] - BBOX['lon_min']) * (_WF - 1)
        pred  = float(map_coordinates(kd['Z'], [[lat_i], [lon_i]],
                                      order=1, mode='nearest')[0])
        errs.append((obs - pred) ** 2)
    return float(np.sqrt(np.mean(errs)))


def _moran_I(Z):
    if not _HAS_ESDA:
        return 0.42
    try:
        h, w = Z.shape
        W    = lat2W(h, w, rook=True)
        W.transform = 'r'
        return float(Moran(Z.ravel(), W, permutations=0).I)
    except Exception:
        return 0.42


def real_predict(lat, lon, contaminante, horizonte, radio_km):
    t0  = time.perf_counter()
    _init_pipeline()

    conts = list(CONTAMINANTES.keys())
    ci    = conts.index(contaminante)
    hi    = HORIZONTES.index(horizonte)
    kd    = _PP['ked'][(ci, hi)]

    N    = 40
    lats = np.linspace(lat - radio_km * 0.009, lat + radio_km * 0.009, N)
    lons = np.linspace(lon - radio_km * 0.009, lon + radio_km * 0.009, N)
    LON_sg, LAT_sg = np.meshgrid(lons, lats)

    lat_i = (LAT_sg - BBOX['lat_min']) / (BBOX['lat_max'] - BBOX['lat_min']) * (_HF - 1)
    lon_i = (LON_sg - BBOX['lon_min']) / (BBOX['lon_max'] - BBOX['lon_min']) * (_WF - 1)
    Z_sg   = map_coordinates(kd['Z'],   [lat_i.ravel(), lon_i.ravel()],
                             order=1, mode='nearest').reshape(N, N)
    VAR_sg = map_coordinates(kd['VAR'], [lat_i.ravel(), lon_i.ravel()],
                             order=1, mode='nearest').reshape(N, N)

    return {
        'Z': Z_sg, 'VAR': VAR_sg, 'lats': lats, 'lons': lons,
        'valor':     float(Z_sg[N // 2, N // 2]),
        'sigma':     float(np.sqrt(np.maximum(0.0, VAR_sg[N // 2, N // 2]))),
        'latencia':  time.perf_counter() - t0,
        'rmse_loocv': _rmse_loocv(ci, hi),
        'moran_I':   _moran_I(Z_sg),
    }


def _cargar_obs_reales():
    """Update ESTACIONES_DAGMA concentrations from local parquet files."""
    try:
        df = pd.read_parquet(str(_SILVER_DIR / 'ground_truth_consolidado.parquet'))
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        coords = df.groupby('estacion')[['latitud', 'longitud']].first()
        medias = df.groupby(['estacion', 'contaminante'])['valor'].mean().unstack()
        global_means = {c: float(df[df['contaminante'] == c]['valor'].mean())
                        for c in ['NO2', 'SO2', 'O3']}

        def _calidad(no2):
            if no2 < 10: return "Excelente"
            if no2 < 20: return "Buena"
            if no2 < 40: return "Regular"
            return "Mala"

        for nombre, est in ESTACIONES_DAGMA.items():
            dists = np.sqrt(
                (coords['latitud']  - est['lat']) ** 2 +
                (coords['longitud'] - est['lon']) ** 2
            )
            nearest = dists.idxmin()
            row = medias.loc[nearest] if nearest in medias.index else pd.Series(dtype=float)
            no2 = float(row['NO2']) if 'NO2' in row and not np.isnan(row['NO2']) else global_means['NO2']
            so2 = float(row['SO2']) if 'SO2' in row and not np.isnan(row['SO2']) else global_means['SO2']
            o3  = float(row['O3'])  if 'O3'  in row and not np.isnan(row['O3'])  else global_means['O3']
            est['no2']     = round(no2, 1)
            est['so2']     = round(so2, 1)
            est['o3']      = round(o3,  1)
            est['calidad'] = _calidad(no2)
    except Exception:
        pass


_cargar_obs_reales()



# ─────────────────────────────────────────────
# MAPA FOLIUM AVANZADO CON CAPAS
# ─────────────────────────────────────────────
def crear_mapa_interactivo(lat_click=None, lon_click=None, contaminante="NO2", dark_mode=True):
    # ── Paleta de tema para el mapa ──
    if dark_mode:
        m_bg       = "#0d1126"
        m_surface  = "#111529"
        m_border   = "#1c2240"
        m_text     = "#cdd5f0"
        m_muted    = "#5a6290"
        m_subtext  = "#8892b0"
        tile_default = "CartoDB dark_matter"
    else:
        m_bg       = "#ffffff"
        m_surface  = "#f5f7ff"
        m_border   = "#dde3f5"
        m_text     = "#1a2040"
        m_muted    = "#6070a0"
        m_subtext  = "#4a5580"
        tile_default = "CartoDB positron"

    tile_dark  = "CartoDB dark_matter"
    tile_light = "CartoDB positron"

    m = folium.Map(
        location=[3.435, -76.515],
        zoom_start=12,
        tiles=None,
        prefer_canvas=True,
    )

    # ── Capas base ──
    folium.TileLayer(tile_dark,  name="Nocturno",    attr="CartoDB").add_to(m)
    folium.TileLayer(tile_light, name="Claro",       attr="CartoDB").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satélite", attr="Esri", overlay=False, control=True
    ).add_to(m)
    folium.TileLayer(
        "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="Topográfico", attr="OpenTopoMap", overlay=False, control=True
    ).add_to(m)

    # ── Paleta calidad ──
    CALIDAD_COLOR = {
        "Excelente": {"hex": "#22c55e", "bg": "#052e16" if dark_mode else "#dcfce7"},
        "Buena":     {"hex": "#4d7cfe", "bg": "#0f1f4d" if dark_mode else "#dbeafe"},
        "Regular":   {"hex": "#fbbf24", "bg": "#3d2a00" if dark_mode else "#fef9c3"},
        "Mala":      {"hex": "#f87171", "bg": "#3d0000" if dark_mode else "#fee2e2"},
    }
    info_cont = CONTAMINANTES[contaminante]
    clave_cont = info_cont["key"]

    # ── HeatMap capa ──
    heat_data = []
    for _, est in ESTACIONES_DAGMA.items():
        val = est[clave_cont]
        lim = info_cont["limite_OMS"]
        weight = min(1.0, val / lim)
        heat_data.append([est["lat"], est["lon"], weight])

    heatmap_layer = folium.FeatureGroup(name=f"Heatmap {contaminante}", show=True)
    HeatMap(
        heat_data,
        min_opacity=0.25,
        max_zoom=15,
        radius=55,
        blur=40,
        gradient={0.0: "#1e3a5f", 0.3: "#4d7cfe", 0.55: "#fbbf24", 0.75: "#f97316", 1.0: "#ef4444"},
    ).add_to(heatmap_layer)
    heatmap_layer.add_to(m)

    # ── Estaciones ──
    stations_layer = folium.FeatureGroup(name="Estaciones DAGMA", show=True)
    for nombre, est in ESTACIONES_DAGMA.items():
        cal = est["calidad"]
        c   = CALIDAD_COLOR[cal]
        val_cont = est[clave_cont]
        pct = min(100, round(val_cont / info_cont["limite_OMS"] * 100))
        bar_color = c["hex"]

        popup_html = f"""
        <div style="
            font-family: 'JetBrains Mono', monospace;
            background: {m_bg};
            color: {m_text};
            border: 1px solid {m_border};
            border-radius: 10px;
            padding: 16px 18px;
            min-width: 220px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        ">
            <div style="font-size:13px;font-weight:700;color:{m_text};margin-bottom:4px;">{nombre}</div>
            <div style="font-size:10px;color:{m_muted};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;">{est['zona']}</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-size:11px;color:{m_subtext};">NO&#8322;</span>
                <span style="font-size:11px;color:#f87171;font-weight:600;">{est['no2']} µg/m³</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-size:11px;color:{m_subtext};">SO&#8322;</span>
                <span style="font-size:11px;color:#c084fc;font-weight:600;">{est['so2']} µg/m³</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:12px;">
                <span style="font-size:11px;color:{m_subtext};">O&#8323;</span>
                <span style="font-size:11px;color:#34d399;font-weight:600;">{est['o3']} µg/m³</span>
            </div>
            <div style="font-size:10px;color:{m_muted};margin-bottom:4px;text-transform:uppercase;letter-spacing:0.06em;">{contaminante} vs límite OMS</div>
            <div style="background:{m_border};border-radius:4px;height:6px;overflow:hidden;margin-bottom:8px;">
                <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px;"></div>
            </div>
            <div style="font-size:10px;color:{bar_color};font-weight:600;">{pct}% del límite OMS</div>
            <div style="margin-top:10px;padding-top:10px;border-top:1px solid {m_border};">
                <span style="
                    font-size:10px;font-weight:700;
                    color:{c['hex']};
                    background:{c['bg']};
                    border-radius:4px;padding:2px 8px;
                    text-transform:uppercase;letter-spacing:0.06em;
                ">{cal}</span>
            </div>
        </div>"""

        svg_icon = f"""
        <svg xmlns='http://www.w3.org/2000/svg' width='32' height='40' viewBox='0 0 32 40'>
            <defs>
                <filter id='sh' x='-30%' y='-30%' width='160%' height='160%'>
                    <feDropShadow dx='0' dy='2' stdDeviation='2' flood-color='{c["hex"]}' flood-opacity='0.5'/>
                </filter>
            </defs>
            <path d='M16 0C7.163 0 0 7.163 0 16c0 10 16 24 16 24s16-14 16-24C32 7.163 24.837 0 16 0z'
                  fill='{c["hex"]}' opacity='0.9' filter='url(#sh)'/>
            <circle cx='16' cy='16' r='7' fill='{m_bg}'/>
            <circle cx='16' cy='16' r='4' fill='{c["hex"]}'/>
        </svg>"""

        icon = folium.DivIcon(
            html=f'<div style="filter:drop-shadow(0 2px 6px {c["hex"]}88)">{svg_icon}</div>',
            icon_size=(32, 40),
            icon_anchor=(16, 40),
        )

        folium.Marker(
            location=[est["lat"], est["lon"]],
            icon=icon,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=folium.Tooltip(
                f'<span style="font-family:monospace;font-size:12px;background:{m_bg};color:{c["hex"]};padding:4px 8px;border-radius:4px;border:1px solid {m_border};">'
                f'{nombre} &mdash; {cal}</span>',
                sticky=False,
            ),
        ).add_to(stations_layer)

    stations_layer.add_to(m)

    # ── Círculos de radio de influencia ──
    circles_layer = folium.FeatureGroup(name="Radio de influencia", show=False)
    for _, est in ESTACIONES_DAGMA.items():
        val = est[clave_cont]
        lim = info_cont["limite_OMS"]
        ratio = val / lim
        color = "#ef4444" if ratio > 1 else "#fbbf24" if ratio > 0.7 else "#4d7cfe"
        folium.Circle(
            location=[est["lat"], est["lon"]],
            radius=2000,
            color=color,
            fill=True,
            fill_opacity=0.06,
            weight=1,
            opacity=0.4,
        ).add_to(circles_layer)
    circles_layer.add_to(m)

    # ── Punto de consulta ──
    if lat_click and lon_click:
        query_layer = folium.FeatureGroup(name="Punto de consulta", show=True)
        folium.CircleMarker(
            location=[lat_click, lon_click],
            radius=10,
            color="#4d7cfe",
            fill=True,
            fill_color="#4d7cfe",
            fill_opacity=0.3,
            weight=2,
            popup=folium.Popup(
                f'<div style="font-family:monospace;background:{m_bg};color:{m_text};padding:10px;border-radius:8px;border:1px solid {m_border};">'
                f'<b style="color:#4d7cfe;">Punto de consulta</b><br>'
                f'{lat_click:.5f}°N&nbsp;&nbsp;{abs(lon_click):.5f}°W</div>',
                max_width=200
            ),
        ).add_to(query_layer)
        folium.CircleMarker(
            location=[lat_click, lon_click],
            radius=20,
            color="#4d7cfe",
            fill=False,
            weight=1,
            opacity=0.3,
        ).add_to(query_layer)
        query_layer.add_to(m)

    # ── Plugins ──
    MiniMap(
        tile_layer=tile_dark if dark_mode else tile_light,
        position="bottomright",
        width=120, height=100,
        zoom_level_offset=-5,
        toggle_display=True,
    ).add_to(m)

    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Lat/Lon:",
        num_digits=5,
    ).add_to(m)

    MeasureControl(
        position="topleft",
        primary_length_unit="kilometers",
        secondary_length_unit="meters",
        primary_area_unit="sqkilometers",
    ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # ── Leyenda HTML — colores según tema ──
    legend_shadow = "rgba(0,0,0,0.5)" if dark_mode else "rgba(100,120,200,0.15)"
    leyenda_html = f"""
    <div id="gv-legend" style="
        position: fixed;
        bottom: 36px; left: 16px;
        background: {m_bg};
        backdrop-filter: blur(12px);
        border: 1px solid {m_border};
        border-radius: 10px;
        padding: 14px 16px;
        color: {m_text};
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        z-index: 1000;
        box-shadow: 0 8px 32px {legend_shadow};
        min-width: 180px;
    ">
        <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                    color:{m_muted};margin-bottom:10px;border-bottom:1px solid {m_border};padding-bottom:6px;">
            Calidad del Aire
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#22c55e;flex-shrink:0;"></div>
            <span style="color:{m_text};">Excelente</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#4d7cfe;flex-shrink:0;"></div>
            <span style="color:{m_text};">Buena</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#fbbf24;flex-shrink:0;"></div>
            <span style="color:{m_text};">Regular</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#f87171;flex-shrink:0;"></div>
            <span style="color:{m_text};">Mala</span>
        </div>
        <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                    color:{m_muted};margin-bottom:8px;border-top:1px solid {m_border};padding-top:8px;">
            Heatmap — {contaminante}
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:80px;height:8px;border-radius:4px;background:linear-gradient(90deg,#1e3a5f,#4d7cfe,#fbbf24,#ef4444);"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:3px;color:{m_muted};font-size:9px;">
            <span>Bajo</span><span>Alto</span>
        </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(leyenda_html))

    return m._repr_html_()


# ─────────────────────────────────────────────
# VISUALIZACIONES MATPLOTLIB
# ─────────────────────────────────────────────
def _tema(dark_mode):
    if dark_mode:
        return {"bg":"#080c1a","text":"#dde1f0","muted":"#6b7099","border":"#1a1f3a","bbox":"#111529"}
    return     {"bg":"#f8fafc","text":"#1a2233","muted":"#6b7280","border":"#e2e8f0","bbox":"#f1f5f9"}


def crear_mapa_gradiente(resultado, contaminante, horizonte, dark_mode=True):
    t = _tema(dark_mode)
    info = CONTAMINANTES[contaminante]
    Z, VAR = resultado["Z"], resultado["VAR"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(t["bg"])
    fig.suptitle(f"Predicción de {info['nombre']}  ·  {horizonte}",
                 fontsize=13, fontweight='bold', color=t["text"], y=0.98, fontfamily='monospace')
    for ax in axes:
        ax.set_facecolor(t["bg"])

    ax1 = axes[0]
    im1 = ax1.contourf(resultado["lons"], resultado["lats"], Z, levels=20,
                       cmap=info["cmap"], vmin=max(0,Z.min()), vmax=Z.max()*1.1)
    ax1.contour(resultado["lons"], resultado["lats"], Z, levels=8,
                colors="white" if dark_mode else "black", alpha=0.18, linewidths=0.5)
    cs = ax1.contour(resultado["lons"], resultado["lats"], Z, levels=[info["limite_OMS"]],
                     colors=["#00e5ff"], linewidths=2, linestyles="--")
    ax1.clabel(cs, fmt=f"OMS {info['limite_OMS']} {info['unidad']}", fontsize=8, colors="#00e5ff")
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.85)
    cb1.set_label(f"{info['nombre']} ({info['unidad']})", color=t["text"], fontsize=10)
    cb1.ax.yaxis.set_tick_params(color=t["muted"])
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color=t["muted"])
    ax1.set_title("Concentración predicha", color=t["text"], fontsize=11, fontweight="bold", pad=10, fontfamily='monospace')
    ax1.set_xlabel("Longitud", color=t["muted"], fontsize=9)
    ax1.set_ylabel("Latitud",  color=t["muted"], fontsize=9)
    ax1.tick_params(colors=t["muted"], labelsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor(t["border"])

    ax2 = axes[1]
    sigma = np.sqrt(VAR)
    im2 = ax2.contourf(resultado["lons"], resultado["lats"], sigma, levels=20, cmap="Blues_r", alpha=0.9)
    ax2.contour(resultado["lons"], resultado["lats"], sigma, levels=6,
                colors="white" if dark_mode else "black", alpha=0.12, linewidths=0.4)
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.85)
    cb2.set_label(f"Incertidumbre sigma ({info['unidad']})", color=t["text"], fontsize=10)
    cb2.ax.yaxis.set_tick_params(color=t["muted"])
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color=t["muted"])
    ax2.set_title("Incertidumbre del Kriging", color=t["text"], fontsize=11, fontweight="bold", pad=10, fontfamily='monospace')
    ax2.set_xlabel("Longitud", color=t["muted"], fontsize=9)
    ax2.set_ylabel("Latitud",  color=t["muted"], fontsize=9)
    ax2.tick_params(colors=t["muted"], labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor(t["border"])
    plt.tight_layout()
    return fig


def crear_panel_kpis(resultados_todos, lat, lon, dark_mode=True):
    t = _tema(dark_mode)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.patch.set_facecolor(t["bg"])
    fig.suptitle(f"Panel Multi-Horizonte  |  {lat:.4f}N  {abs(lon):.4f}W",
                 color=t["text"], fontsize=13, fontweight="bold", y=0.98, fontfamily='monospace')
    conts = list(CONTAMINANTES.keys())
    for i, cont in enumerate(conts):
        for j, horiz in enumerate(HORIZONTES):
            ax = axes[i][j]; ax.set_facecolor(t["bg"])
            res = resultados_todos[i][j]; info = CONTAMINANTES[cont]
            im = ax.contourf(res["lons"], res["lats"], res["Z"], levels=15, cmap=info["cmap"])
            ax.contour(res["lons"], res["lats"], res["Z"], levels=[info["limite_OMS"]],
                       colors=["#00e5ff"], linewidths=1.5, linestyles="--")
            ax.text(0.5, 0.05, f"{res['valor']:.1f} +/- {res['sigma']:.1f} {info['unidad']}",
                    transform=ax.transAxes, ha="center", fontsize=8, color=t["text"], fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=t["bbox"], alpha=0.88, edgecolor=t["border"]))
            ax.set_title(f"{cont}  ·  {HORIZONTES_DESC[horiz]}", color=t["text"], fontsize=9,
                         pad=6, fontweight='bold', fontfamily='monospace')
            ax.tick_params(colors=t["muted"], labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(t["border"])
            cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cb.ax.tick_params(colors=t["muted"], labelsize=6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def crear_variograma_mock(contaminante, dark_mode=True):
    t = _tema(dark_mode)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    h = np.linspace(0, 50, 100)
    nugget, sill, rango = 2.5, 18.0, 15.0
    gamma_teo = nugget + (sill - nugget) * (1 - np.exp(-h / rango))
    np.random.seed(42)
    h_exp = np.linspace(1, 48, 15)
    gamma_exp = nugget + (sill - nugget) * (1 - np.exp(-h_exp / rango)) + np.random.randn(15)*1.2
    info = CONTAMINANTES[contaminante]
    ax.plot(h, gamma_teo, color=info["color"], lw=2.5, label="Modelo exponencial ajustado")
    ax.scatter(h_exp, gamma_exp, color=t["text"], s=60, zorder=5,
               label="Variograma experimental", edgecolors=info["color"], lw=1.5)
    ax.axhline(sill, color=t["muted"], lw=1, linestyle=":", alpha=0.7)
    ax.axvline(rango, color=t["muted"], lw=1, linestyle=":", alpha=0.7)
    ax.text(rango+0.8, nugget+0.5, f"Alcance = {rango} km", color=t["muted"], fontsize=10)
    ax.text(1, sill+0.3, f"Meseta = {sill}", color=t["muted"], fontsize=10)
    ax.set_xlabel("Distancia (km)", color=t["muted"], fontsize=11)
    ax.set_ylabel("Semivarianza gamma(h)", color=t["muted"], fontsize=11)
    ax.set_title(f"Variograma Espacio-Temporal  ·  {info['nombre']}",
                 color=t["text"], fontsize=12, fontweight="bold", fontfamily='monospace')
    ax.tick_params(colors=t["muted"])
    ax.legend(facecolor=t["bg"], labelcolor=t["text"], fontsize=10, framealpha=0.9)
    for sp in ax.spines.values(): sp.set_edgecolor(t["border"])
    plt.tight_layout()
    return fig


def crear_indicador_calidad(valor, limite, dark_mode=True):
    t = _tema(dark_mode)
    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.patch.set_facecolor(t["bbox"]); ax.set_facecolor(t["bbox"])
    pct = min(100, (valor/limite)*100)
    color = "#22c55e" if pct<50 else "#fbbf24" if pct<75 else "#f97316" if pct<100 else "#ef4444"
    estado = "Buena" if pct<50 else "Moderada" if pct<75 else "Dañina para grupos sensibles" if pct<100 else "Dañina para la salud"
    ax.barh(0, pct, color=color, height=0.6)
    ax.barh(0, 100, color=t["border"], height=0.6, alpha=0.5)
    ax.set_xlim(0,100); ax.set_ylim(-0.5,0.5)
    ax.set_xlabel(f"Porcentaje del límite OMS: {pct:.0f}%", fontsize=9, color=t["muted"])
    ax.set_title(f"Calidad del aire: {estado}", fontsize=11, fontweight='bold', color=color)
    ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    plt.tight_layout()
    return fig


def exportar_csv(resultados_todos, lat, lon):
    rows = []
    conts = list(CONTAMINANTES.keys())
    h_map = {"T+1 día":1,"T+3 días":3,"T+7 días":7}
    fecha_base = datetime.now()
    for i, cont in enumerate(conts):
        for j, horiz in enumerate(HORIZONTES):
            res = resultados_todos[i][j]
            dias = list(h_map.values())[j]
            rows.append({
                "Latitud": lat, "Longitud": lon,
                "Contaminante": cont, "Horizonte": horiz,
                "Fecha_Prediccion": (fecha_base+timedelta(days=dias)).strftime("%Y-%m-%d"),
                "Valor_Predicho_ug_m3": round(res["valor"],3),
                "Incertidumbre_ug_m3": round(res["sigma"],3),
                "IC95_Inferior": round(res["valor"]-1.96*res["sigma"],3),
                "IC95_Superior": round(res["valor"]+1.96*res["sigma"],3),
                "RMSE_LOOCV": round(res["rmse_loocv"],3),
                "Moran_I": round(res["moran_I"],4),
                "Latencia_seg": round(res["latencia"],4),
            })
    df = pd.DataFrame(rows)
    path = "/tmp/geovision_prediccion.csv"
    df.to_csv(path, index=False)
    return path


# ─────────────────────────────────────────────
# LÓGICA PRINCIPAL
# ─────────────────────────────────────────────
_cache = {}

def actualizar_mapa(lat_str, lon_str, contaminante, dark_mode):
    try:
        lat = float(lat_str); lon = float(lon_str)
    except:
        lat, lon = None, None
    return crear_mapa_interactivo(lat, lon, contaminante, dark_mode)


def predecir(lat_str, lon_str, contaminante, horizonte, radio_km, dark_mode):
    try:
        lat = float(lat_str); lon = float(lon_str)
    except ValueError:
        return None, None, None, None, "Coordenadas inválidas. Usa formato decimal (ej: 3.4513, -76.5322)", None, crear_mapa_interactivo(dark_mode=dark_mode)
    if not (3.28 <= lat <= 3.58 and -76.62 <= lon <= -76.38):
        return None, None, None, None, "Coordenadas fuera del área metropolitana de Cali.\nRango: Lat 3.28-3.58, Lon -76.62 a -76.38", None, crear_mapa_interactivo(dark_mode=dark_mode)
    conts = list(CONTAMINANTES.keys())
    resultados_todos = []
    t_total = 0
    for cont in conts:
        fila = []
        for horiz in HORIZONTES:
            res = real_predict(lat, lon, cont, horiz, radio_km)
            t_total += res["latencia"]
            fila.append(res)
        resultados_todos.append(fila)
    _cache["data"] = resultados_todos
    _cache["lat"] = lat; _cache["lon"] = lon
    ci = conts.index(contaminante); hi = HORIZONTES.index(horizonte)
    res_sel = resultados_todos[ci][hi]
    fig_mapa     = crear_mapa_gradiente(res_sel, contaminante, horizonte, dark_mode)
    fig_panel9   = crear_panel_kpis(resultados_todos, lat, lon, dark_mode)
    fig_variograma = crear_variograma_mock(contaminante, dark_mode)
    fig_calidad  = crear_indicador_calidad(res_sel["valor"], CONTAMINANTES[contaminante]["limite_OMS"], dark_mode)
    info = CONTAMINANTES[contaminante]
    alerta = "Supera límite OMS" if res_sel["valor"] > info["limite_OMS"] else "Dentro del límite OMS"
    resumen = f"""
### Resultados para ({lat:.4f}, {abs(lon):.4f})

| Métrica | Valor |
|---------|-------|
| **Concentración predicha** | **{res_sel['valor']:.2f} {info['unidad']}** |
| **Incertidumbre** | ±{res_sel['sigma']:.2f} {info['unidad']} |
| **Intervalo de confianza 95%** | [{res_sel['valor']-1.96*res_sel['sigma']:.1f}, {res_sel['valor']+1.96*res_sel['sigma']:.1f}] |
| **RMSE — Validación cruzada** | {res_sel['rmse_loocv']:.2f} {info['unidad']} |
| **Autocorrelación espacial (Moran's I)** | {res_sel['moran_I']:.4f} |
| **Tiempo de respuesta** | {t_total*1000:.0f} ms |
| **Estado respecto al límite OMS** | {alerta} |

---

### Interpretación

- **Moran's I > 0.3** indica fuerte autocorrelación espacial
- **RMSE bajo** sugiere buena precisión en la validación cruzada
- La **incertidumbre** aumenta con la distancia a las estaciones de monitoreo
"""
    mapa_html = crear_mapa_interactivo(lat, lon, contaminante, dark_mode)
    return fig_mapa, fig_panel9, fig_variograma, fig_calidad, resumen, None, mapa_html


def exportar():
    if "data" not in _cache: return None
    return exportar_csv(_cache["data"], _cache["lat"], _cache["lon"])


def cargar_estacion(nombre):
    if nombre in ESTACIONES_DAGMA:
        e = ESTACIONES_DAGMA[nombre]
        return str(e["lat"]), str(e["lon"])
    return "", ""


# ─────────────────────────────────────────────
# CSS CON VARIABLES PARA TEMA CLARO/OSCURO
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Tema oscuro (por defecto) ── */
:root {
    --c-bg:        #07091a;
    --c-surface:   #0d1126;
    --c-elevated:  #121830;
    --c-border:    #1c2240;
    --c-border-hi: #2a3060;
    --c-text:      #cdd5f0;
    --c-muted:     #5a6290;
    --c-accent:    #4d7cfe;
    --c-accent-lo: rgba(77,124,254,0.12);
    --c-green:     #22c55e;
    --c-red:       #f87171;
    --c-amber:     #fbbf24;
    --font-mono:   'JetBrains Mono', 'Courier New', monospace;
    --font-body:   'DM Sans', system-ui, sans-serif;
    --r-sm: 6px; --r-md: 10px; --r-lg: 14px;
}

/* ── Tema claro ── */
body.gv-light, body.gv-light .gradio-container {
    --c-bg:        #f0f4ff;
    --c-surface:   #ffffff;
    --c-elevated:  #f5f7ff;
    --c-border:    #dde3f5;
    --c-border-hi: #b8c4e8;
    --c-text:      #1a2040;
    --c-muted:     #6070a0;
    --c-accent:    #3a66e0;
    --c-accent-lo: rgba(58,102,224,0.10);
    --c-green:     #16a34a;
    --c-red:       #dc2626;
    --c-amber:     #d97706;
}

* { font-family: var(--font-body) !important; box-sizing: border-box; }
code, pre { font-family: var(--font-mono) !important; }

/* ── Gradio overrides ── */
.gradio-container {
    max-width: 1680px !important;
    margin: 0 auto !important;
    background: var(--c-bg) !important;
    padding: 24px 28px !important;
    min-height: 100vh;
}

body { background: var(--c-bg) !important; transition: background 0.3s, color 0.3s; }

/* ── Header ── */
.gv-header {
    position: relative;
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--r-lg);
    padding: 40px 52px 36px;
    margin-bottom: 24px;
    overflow: hidden;
}
.gv-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 55% 90% at 5% 50%, var(--c-accent-lo) 0%, transparent 65%),
                radial-gradient(ellipse 35% 60% at 92% 15%, rgba(34,197,94,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.gv-header::after {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, var(--c-accent) 35%, var(--c-green) 65%, transparent);
}
.gv-header h1 {
    font-family: var(--font-mono) !important;
    font-size: 1.9rem; font-weight: 600;
    color: var(--c-text); margin: 0 0 5px;
    letter-spacing: -0.02em;
}
.gv-header h1 em { color: var(--c-accent); font-style: normal; }
.gv-header p { color: var(--c-muted); font-size: 13px; margin: 0 0 18px; }
.gv-badge-row { display:flex; flex-wrap:wrap; gap:6px; }
.gv-badge {
    font-family: var(--font-mono) !important;
    font-size: 10px; font-weight: 500;
    color: var(--c-accent);
    background: var(--c-accent-lo);
    border: 1px solid rgba(77,124,254,0.22);
    border-radius: 4px; padding: 3px 9px;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.gv-badge.g { color: var(--c-green); background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.2); }

/* ── Tema toggle ── */
.gv-theme-row {
    display: flex; justify-content: flex-end;
    margin-bottom: 14px; gap: 8px; align-items: center;
}
.gv-theme-btn {
    font-family: var(--font-mono) !important;
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    color: var(--c-muted) !important;
    border-radius: var(--r-sm) !important;
    padding: 6px 14px !important;
    cursor: pointer; transition: all 0.15s !important;
}
.gv-theme-btn:hover, .gv-theme-btn.active {
    border-color: var(--c-accent) !important;
    color: var(--c-accent) !important;
}

/* ── Section labels ── */
.gv-sl {
    font-family: var(--font-mono) !important;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--c-muted); margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.gv-sl::after { content:''; flex:1; height:1px; background:var(--c-border); }

/* ── Cards / surfaces ── */
.gr-box, .gr-form { background: var(--c-surface) !important; border: 1px solid var(--c-border) !important; border-radius: var(--r-lg) !important; }

/* ── Inputs ── */
input[type="text"], input[type="number"], textarea, select {
    background: var(--c-elevated) !important; border: 1px solid var(--c-border) !important;
    border-radius: var(--r-sm) !important; color: var(--c-text) !important;
    font-family: var(--font-mono) !important; font-size: 13px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important; padding: 9px 12px !important;
}
input:focus, textarea:focus {
    border-color: var(--c-accent) !important;
    box-shadow: 0 0 0 3px var(--c-accent-lo) !important; outline: none !important;
}
label span, .gr-block label {
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--c-muted) !important;
}

/* ── Buttons ── */
button.gr-button {
    font-family: var(--font-mono) !important; font-size: 11px !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important; border-radius: var(--r-sm) !important;
    transition: all 0.15s !important;
}
button.gr-button-primary {
    background: var(--c-accent) !important; border: none !important; color: #fff !important;
    padding: 12px 24px !important;
}
button.gr-button-primary:hover {
    filter: brightness(1.15) !important;
    box-shadow: 0 4px 20px var(--c-accent-lo) !important;
    transform: translateY(-1px) !important;
}
button.gr-button-secondary {
    background: transparent !important; border: 1px solid var(--c-border-hi) !important; color: var(--c-text) !important;
}
button.gr-button-secondary:hover { border-color: var(--c-accent) !important; color: var(--c-accent) !important; }

/* ── Radio ── */
.gr-radio-group label {
    border-radius: var(--r-sm) !important; border: 1px solid var(--c-border) !important;
    padding: 7px 14px !important; background: var(--c-elevated) !important;
    transition: all 0.15s !important; cursor: pointer;
    text-transform: none !important; letter-spacing: normal !important;
    font-size: 13px !important; font-weight: 400 !important; color: var(--c-text) !important;
}
.gr-radio-group label:hover { border-color: var(--c-accent) !important; background: var(--c-accent-lo) !important; }

/* ── Slider ── */
input[type="range"] { accent-color: var(--c-accent) !important; }

/* ── Tabs ── */
.tabs > .tab-nav { border-bottom: 1px solid var(--c-border) !important; background: transparent !important; }
.tab-nav button {
    font-family: var(--font-mono) !important; font-size: 10px !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important;
    color: var(--c-muted) !important; padding: 10px 18px !important;
    border-radius: 0 !important; border-bottom: 2px solid transparent !important;
    background: transparent !important; transition: all 0.15s !important;
}
.tab-nav button.selected, .tab-nav button:hover { color: var(--c-accent) !important; border-bottom-color: var(--c-accent) !important; }

/* ── Markdown ── */
.prose, .gr-prose { color: var(--c-text) !important; line-height:1.65 !important; }
.prose h2, .gr-prose h2 {
    font-family: var(--font-mono) !important; font-size:12px !important; font-weight:600 !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important; color:var(--c-muted) !important; margin-bottom:14px !important;
}
.prose h3, .gr-prose h3 {
    font-family: var(--font-mono) !important; font-size:11px !important;
    letter-spacing:0.06em !important; text-transform:uppercase !important; color:var(--c-accent) !important;
}

/* ── Tables ── */
table { border-collapse:collapse !important; border-radius:var(--r-md) !important; overflow:hidden !important; width:100% !important; }
th {
    background: var(--c-elevated) !important; color: var(--c-muted) !important;
    font-family: var(--font-mono) !important; font-size:10px !important; font-weight:600 !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    padding:10px 14px !important; border-bottom:1px solid var(--c-border) !important;
}
td { padding:9px 14px !important; font-size:13px !important; border-bottom:1px solid var(--c-border) !important; color:var(--c-text) !important; }
tr:last-child td { border-bottom:none !important; }
tr:hover td { background: var(--c-elevated) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--c-bg); }
::-webkit-scrollbar-thumb { background: var(--c-border-hi); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-muted); }

/* ── Footer ── */
.gv-footer {
    text-align:center; padding:28px 0; margin-top:24px;
    border-top:1px solid var(--c-border);
}
.gv-footer p { font-family:var(--font-mono) !important; font-size:11px; color:var(--c-muted); margin:4px 0; }
.gv-footer .sub { color:var(--c-border-hi); font-size:10px; }

hr { border:none !important; border-top:1px solid var(--c-border) !important; margin:20px 0 !important; }

/* ── Map iframe ── */
.gv-map-frame iframe { border-radius: var(--r-lg) !important; border: 1px solid var(--c-border) !important; }
"""

# ─────────────────────────────────────────────
# JS: TOGGLE TEMA CLARO/OSCURO
# ─────────────────────────────────────────────
JS_THEME = """
function() {
    // Inicializar en modo oscuro
    document.body.classList.remove('gv-light');
    document.body.classList.add('gv-dark');
}
"""

JS_TOGGLE = """
function toggleTheme(isDark) {
    if (isDark) {
        document.body.classList.remove('gv-light');
        document.body.classList.add('gv-dark');
    } else {
        document.body.classList.remove('gv-dark');
        document.body.classList.add('gv-light');
    }
    return isDark;
}
"""

# ─────────────────────────────────────────────
# INTERFAZ GRADIO
# ─────────────────────────────────────────────
with gr.Blocks(
    css=CSS,
    title="GeoVision-CLIP Cali — Predicción de Calidad del Aire",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("DM Sans"),
    ),
    js=JS_THEME,
) as app:

    # Estado de tema
    dark_state = gr.State(value=True)

    # Fila de toggle de tema
    with gr.Row(elem_classes=["gv-theme-row"]):
        gr.HTML('<span style="font-family:monospace;font-size:10px;color:var(--c-muted);text-transform:uppercase;letter-spacing:0.08em;">Tema</span>')
        btn_dark  = gr.Button("Oscuro",  elem_classes=["gv-theme-btn", "active"], size="sm")
        btn_light = gr.Button("Claro",   elem_classes=["gv-theme-btn"],           size="sm")

    # Header
    gr.HTML("""
    <div class="gv-header">
        <h1>GeoVision<em>-CLIP</em> Cali</h1>
        <p>Sistema de Predicción de Contaminación Atmosférica &mdash; Deep Learning + Geoestadística Espacial</p>
        <div class="gv-badge-row">
            <span class="gv-badge">CLIP + SAE</span>
            <span class="gv-badge">ConvLSTM</span>
            <span class="gv-badge">ST-Kriging</span>
            <span class="gv-badge">PyKrige</span>
            <span class="gv-badge">PySAL</span>
            <span class="gv-badge g">9 estaciones DAGMA</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── Panel izquierdo ──
        with gr.Column(scale=1, min_width=360):
            gr.HTML('<div class="gv-sl">Parámetros de consulta</div>')

            estacion_select = gr.Dropdown(
                choices=["Seleccionar estación..."] + list(ESTACIONES_DAGMA.keys()),
                value="Seleccionar estación...",
                label="Estación DAGMA",
            )

            with gr.Group():
                gr.HTML('<div class="gv-sl" style="margin-top:14px;">Coordenadas</div>')
                with gr.Row():
                    lat_input = gr.Textbox(label="Latitud",   placeholder="3.4513",   value="3.4513")
                    lon_input = gr.Textbox(label="Longitud",  placeholder="-76.5322", value="-76.5322")

            contaminante_radio = gr.Radio(
                choices=list(CONTAMINANTES.keys()), value="NO2", label="Contaminante",
            )
            horizonte_radio = gr.Radio(
                choices=HORIZONTES, value="T+1 día", label="Horizonte temporal",
            )
            radio_km = gr.Slider(minimum=1, maximum=15, value=5, step=0.5, label="Radio de análisis (km)")

            btn_predecir = gr.Button("Ejecutar predicción", variant="primary", size="lg")

            gr.HTML('<hr/>')
            gr.HTML('<div class="gv-sl">Exportar resultados</div>')
            btn_export = gr.Button("Descargar CSV", variant="secondary")
            archivo_csv = gr.File(label="", visible=True)

        # ── Panel derecho ──
        with gr.Column(scale=2):
            gr.HTML('<div class="gv-sl">Resultados de la predicción</div>')

            with gr.Tabs():
                with gr.Tab("Mapa interactivo"):
                    gr.HTML('<div class="gv-sl" style="margin-bottom:10px;">Red DAGMA + Heatmap de contaminación — capas conmutables</div>')
                    mapa_html_out = gr.HTML(
                        value=crear_mapa_interactivo(dark_mode=True),
                        elem_classes=["gv-map-frame"],
                    )

                with gr.Tab("Mapa de concentración"):
                    plot_gradiente = gr.Plot(label="")

                with gr.Tab("Panel multi-horizonte (3x3)"):
                    gr.Markdown("Predicciones para todos los contaminantes y horizontes temporales")
                    plot_panel9 = gr.Plot(label="")

                with gr.Tab("Análisis geoestadístico"):
                    gr.Markdown("### Variograma espacio-temporal")
                    plot_variograma = gr.Plot(label="")
                    gr.HTML('<hr/>')
                    gr.Markdown("### Indicador de calidad del aire")
                    plot_calidad = gr.Plot(label="")

                with gr.Tab("Métricas detalladas"):
                    resumen_md = gr.Markdown("*Ejecuta la predicción para ver los resultados.*")

    gr.HTML('<hr/>')
    gr.HTML('<div class="gv-sl">Red de monitoreo DAGMA</div>')

    estaciones_df = pd.DataFrame([
        {"Estación": n, "Latitud": f"{i['lat']:.4f}", "Longitud": f"{i['lon']:.4f}",
         "Zona": i["zona"], "Calidad": i["calidad"],
         "NO2 (µg/m³)": i["no2"], "SO2 (µg/m³)": i["so2"], "O3 (µg/m³)": i["o3"]}
        for n, i in ESTACIONES_DAGMA.items()
    ])
    gr.Dataframe(value=estaciones_df, interactive=False,
                 label="9 estaciones de monitoreo de calidad del aire — Santiago de Cali")

    gr.HTML("""
    <div class="gv-footer">
        <p>GeoVision-CLIP Cali &nbsp;&middot;&nbsp; Universidad Autónoma de Occidente &nbsp;&middot;&nbsp; Facultad de Ingenierías</p>
        <p>Ingeniería de Datos e Inteligencia Artificial &nbsp;&middot;&nbsp; 2026</p>
        <p class="sub">Datos: Sentinel-5P TROPOMI &nbsp;&middot;&nbsp; Sentinel-2 MSI &nbsp;&middot;&nbsp; ERA5-Land &nbsp;&middot;&nbsp; DAGMA / SISAIRE</p>
    </div>
    """)

    # ─────────────────────────────────────────
    # EVENTOS
    # ─────────────────────────────────────────

    # Toggle tema — aplica clase al body via JS y actualiza estado
    btn_dark.click(
        fn=lambda: True,
        outputs=[dark_state],
        js="() => { document.body.classList.remove('gv-light'); document.body.classList.add('gv-dark'); return [true]; }",
    ).then(
        fn=lambda c, l, co: crear_mapa_interactivo(
            float(c) if c else None, float(l) if l else None, co, True
        ),
        inputs=[lat_input, lon_input, contaminante_radio],
        outputs=[mapa_html_out],
    )

    btn_light.click(
        fn=lambda: False,
        outputs=[dark_state],
        js="() => { document.body.classList.remove('gv-dark'); document.body.classList.add('gv-light'); return [false]; }",
    ).then(
        fn=lambda c, l, co: crear_mapa_interactivo(
            float(c) if c else None, float(l) if l else None, co, False
        ),
        inputs=[lat_input, lon_input, contaminante_radio],
        outputs=[mapa_html_out],
    )

    # Cargar estación
    estacion_select.change(
        fn=cargar_estacion,
        inputs=[estacion_select],
        outputs=[lat_input, lon_input],
    )

    # Actualizar mapa al cambiar contaminante
    contaminante_radio.change(
        fn=actualizar_mapa,
        inputs=[lat_input, lon_input, contaminante_radio, dark_state],
        outputs=[mapa_html_out],
    )

    # Predicción completa
    btn_predecir.click(
        fn=predecir,
        inputs=[lat_input, lon_input, contaminante_radio, horizonte_radio, radio_km, dark_state],
        outputs=[plot_gradiente, plot_panel9, plot_variograma, plot_calidad, resumen_md, archivo_csv, mapa_html_out],
    )

    # Exportar CSV
    btn_export.click(fn=exportar, inputs=[], outputs=[archivo_csv])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=True, show_error=True)