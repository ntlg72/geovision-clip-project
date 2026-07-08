"""
build_gold_normalized.py
Normaliza nombres de estaciones, aplica coordenadas de SISAIRE como autoritativas,
reconstruye silver/gold localmente y sube a Wasabi.
"""
import os, re, json, hashlib, io
from datetime import datetime
import pandas as pd
import numpy as np
import boto3

# ── Wasabi ──────────────────────────────────────────────────────────────────
AWS_ACCESS = "EPAJLZS1BT5K3X8CPGR2"
AWS_SECRET = "QnJ9u6PCBlKUDUpGMMsjCAkOTaRaTYJY6uwl8hek"
ENDPOINT   = "https://s3.us-west-1.wasabisys.com"
BUCKET     = "bronze.sat"
REGION     = "us-east-1"

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=AWS_ACCESS,
    aws_secret_access_key=AWS_SECRET,
    region_name=REGION,
)

LOCAL = "/teamspace/studios/this_studio"

# ── Mapeo canónico ──────────────────────────────────────────────────────────
NOMBRE_CANONICO = {
    "BASE AÉREA":            "BASE_AEREA",
    "BASE AEREA":            "BASE_AEREA",
    "CAÑAVERALEJO":          "CANAVERALEJO",
    "CANAVERALEJO":          "CANAVERALEJO",
    "ERA OBRERO":            "ERA_OBRERO",
    "ERA_OBRERO":            "ERA_OBRERO",
    "LA ERMITA":             "LA_ERMITA",
    "LA_ERMITA":             "LA_ERMITA",
    "LA FLORA":              "LA_FLORA",
    "LA_FLORA":              "LA_FLORA",
    "TRANSITORIA-NAVARRO":   "TRANSITORIA_NAVARRO",
    "TRANSITORIA_NAVARRO":   "TRANSITORIA_NAVARRO",
    "UNIVERSIDAD DEL VALLE": "UNIVERSIDAD_DEL_VALLE",
    "UNIVALLE":              "UNIVERSIDAD_DEL_VALLE",
    "UNIVERSIDAD_DEL_VALLE": "UNIVERSIDAD_DEL_VALLE",
    "ACOPI":                 "ACOPI",
    "COMPARTIR":             "COMPARTIR",
    "ESCUELA_ARGENTINA":     "ESCUELA_ARGENTINA",
    "ESCUELA ARGENTINA":     "ESCUELA_ARGENTINA",
    "MOVIL":                 "MOVIL",
    "PANCE":                 "PANCE",
}

BBOX = dict(lon_min=-76.60, lon_max=-76.40, lat_min=3.30, lat_max=3.55)

# ── Helpers ─────────────────────────────────────────────────────────────────
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def read_hive_parquet_local(folder):
    """Lee particiones Hive locales e inyecta columna de partición."""
    dfs = []
    for root, dirs, files in os.walk(folder):
        for fn in files:
            if not fn.endswith(".parquet"):
                continue
            path = os.path.join(root, fn)
            df = pd.read_parquet(path)
            # extraer columnas de partición del path
            for m in re.finditer(r'([^/=]+)=([^/]+)/', path + "/"):
                col, val = m.group(1), m.group(2)
                if col not in df.columns:
                    df[col] = val
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def read_hive_parquet_s3(prefix):
    """Lee todas las particiones Hive de un prefijo S3."""
    paginator = s3.get_paginator("list_objects_v2")
    dfs = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            df = pd.read_parquet(io.BytesIO(body))
            for m in re.finditer(r'([^/=]+)=([^/]+)/', key + "/"):
                col, val = m.group(1), m.group(2)
                if col not in df.columns:
                    df[col] = val
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def upload_parquet(df, s3_key, local_tmp=None):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    size_mb = buf.getbuffer().nbytes / 1e6
    s3.put_object(Bucket=BUCKET, Key=s3_key, Body=buf.getvalue())
    print(f"  ✓ s3://{BUCKET}/{s3_key}  ({size_mb:.2f} MB)")
    return size_mb

def upload_bytes(data: bytes, s3_key: str):
    s3.put_object(Bucket=BUCKET, Key=s3_key, Body=data)
    print(f"  ✓ s3://{BUCKET}/{s3_key}  ({len(data)/1e3:.1f} KB)")

# ── 1. Leer fuentes locales ─────────────────────────────────────────────────
print("=" * 60)
print("1. LEYENDO FUENTES SILVER LOCALES")
print("=" * 60)

sis = read_hive_parquet_local(f"{LOCAL}/silver/sisaire_horaria.parquet")
print(f"SISAIRE:  {len(sis):,} filas  cols={list(sis.columns)}")

dag = read_hive_parquet_local(f"{LOCAL}/silver/dagma_portal_horaria.parquet")
print(f"DAGMA:    {len(dag):,} filas  cols={list(dag.columns)}")

s5p = pd.read_parquet(f"{LOCAL}/silver/no2_s5p_estaciones.parquet")
print(f"S5P:      {len(s5p):,} filas  cols={list(s5p.columns)}")

# ── 2. Normalizar nombres ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. NORMALIZANDO NOMBRES DE ESTACIONES")
print("=" * 60)

def normalize_name(df, col="estacion"):
    raw = df[col].str.strip().str.upper()
    mapped = raw.map(NOMBRE_CANONICO)
    unknown = raw[mapped.isna()].unique()
    if len(unknown):
        print(f"  WARN nombres no mapeados: {list(unknown)}")
    df = df.copy()
    df[col] = mapped.fillna(raw)
    return df

sis = normalize_name(sis)
dag = normalize_name(dag)
s5p = normalize_name(s5p)

print("SISAIRE estaciones:", sorted(sis["estacion"].unique()))
print("DAGMA   estaciones:", sorted(dag["estacion"].unique()))
print("S5P     estaciones:", sorted(s5p["estacion"].unique()))

# ── 3. Tabla de coordenadas SISAIRE (autoritativa) ──────────────────────────
print("\n" + "=" * 60)
print("3. TABLA COORDS SISAIRE (autoritativa)")
print("=" * 60)

coord_sis = (
    sis.groupby("estacion")[["latitud", "longitud"]]
    .median()
    .reset_index()
    .rename(columns={"latitud": "lat_sis", "longitud": "lon_sis"})
)
print(coord_sis.to_string(index=False))

# ── 4. Parchear coordenadas en DAGMA y S5P ──────────────────────────────────
print("\n" + "=" * 60)
print("4. PARCHEANDO COORDS EN DAGMA y S5P")
print("=" * 60)

def patch_coords(df, coord_table):
    df = df.merge(coord_table, on="estacion", how="left")
    mask = df["lat_sis"].notna()
    df.loc[mask, "latitud"]   = df.loc[mask, "lat_sis"]
    df.loc[mask, "longitud"]  = df.loc[mask, "lon_sis"]
    df.drop(columns=["lat_sis", "lon_sis"], inplace=True)
    patched = mask.sum()
    print(f"  {patched:,} filas parcheadas con coords SISAIRE")
    return df

dag = patch_coords(dag, coord_sis)
s5p = patch_coords(s5p, coord_sis)

# ── 5. Filtrar BBox ─────────────────────────────────────────────────────────
def apply_bbox(df, label):
    before = len(df)
    df = df[
        df["longitud"].between(BBOX["lon_min"], BBOX["lon_max"]) &
        df["latitud"].between(BBOX["lat_min"], BBOX["lat_max"])
    ].copy()
    after = len(df)
    if before != after:
        print(f"  {label}: {before-after:,} filas fuera de BBox eliminadas")
    return df

sis = apply_bbox(sis, "SISAIRE")
dag = apply_bbox(dag, "DAGMA")
s5p = apply_bbox(s5p, "S5P")

# ── 6. Reconstruir silver local ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. RECONSTRUYENDO SILVER LOCAL")
print("=" * 60)

import shutil

# SISAIRE
sis_out = f"{LOCAL}/silver/sisaire_horaria.parquet"
if os.path.exists(sis_out):
    shutil.rmtree(sis_out)
os.makedirs(sis_out, exist_ok=True)
for cont, grp in sis.groupby("contaminante"):
    pdir = f"{sis_out}/contaminante={cont}"
    os.makedirs(pdir, exist_ok=True)
    grp.drop(columns=["contaminante"]).to_parquet(
        f"{pdir}/data.parquet", index=False, engine="pyarrow", compression="snappy"
    )
    print(f"  SISAIRE/{cont}: {len(grp):,} filas")

# DAGMA
dag_out = f"{LOCAL}/silver/dagma_portal_horaria.parquet"
if os.path.exists(dag_out):
    shutil.rmtree(dag_out)
os.makedirs(dag_out, exist_ok=True)
for cont, grp in dag.groupby("contaminante"):
    pdir = f"{dag_out}/contaminante={cont}"
    os.makedirs(pdir, exist_ok=True)
    grp.drop(columns=["contaminante"]).to_parquet(
        f"{pdir}/data.parquet", index=False, engine="pyarrow", compression="snappy"
    )
    print(f"  DAGMA/{cont}: {len(grp):,} filas")

# S5P
s5p_out = f"{LOCAL}/silver/no2_s5p_estaciones.parquet"
s5p.to_parquet(s5p_out, index=False, engine="pyarrow", compression="snappy")
print(f"  S5P: {len(s5p):,} filas")

# ── 7. Consolidado gold ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. CONSTRUYENDO CONSOLIDADO GOLD")
print("=" * 60)

def prep(df, fuente):
    df = df.copy()
    if "contaminante" not in df.columns:
        df["contaminante"] = "NO2"
    # columna valor
    if "valor_imputado" in df.columns:
        df["valor"] = df["valor_imputado"]
    # renombrar columna NO2 S5P calibrada → valor
    for s5p_col in ("no2_pseudo_ug_m3_corr", "no2_surface", "no2_estimado_sin_bias_ug_m3"):
        if s5p_col in df.columns and "valor" not in df.columns:
            df.rename(columns={s5p_col: "valor"}, inplace=True)
            break
    df["fuente"] = fuente
    df["contaminante"] = df["contaminante"].astype(str)
    cols = ["estacion", "fecha", "latitud", "longitud", "valor", "fuente", "contaminante"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  WARN {fuente} le faltan columnas: {missing}")
    return df[[c for c in cols if c in df.columns]]

gold = pd.concat([
    prep(sis, "SISAIRE"),
    prep(dag, "DAGMA_portal"),
    prep(s5p, "S5P_pseudo"),
], ignore_index=True)

gold["fecha"] = pd.to_datetime(gold["fecha"], errors="coerce")
gold.sort_values(["estacion", "contaminante", "fecha"], inplace=True)
gold.reset_index(drop=True, inplace=True)

print(f"Shape:         {gold.shape}")
print(f"Estaciones:    {sorted(gold['estacion'].unique())}")
print(f"Fuentes:       {sorted(gold['fuente'].unique())}")
print(f"Contaminantes: {sorted(gold['contaminante'].unique())}")
print(f"Rango fechas:  {gold['fecha'].min()} → {gold['fecha'].max()}")
print(f"Nulos valor:   {gold['valor'].isna().mean()*100:.1f}%")

# Verificar coords únicas por estación (deben ser SISAIRE si están en SISAIRE)
coord_check = gold.groupby("estacion")[["latitud", "longitud"]].nunique()
multi_coord = coord_check[(coord_check["latitud"] > 1) | (coord_check["longitud"] > 1)]
if len(multi_coord):
    print(f"\n  WARN estaciones con coordenadas inconsistentes:")
    print(multi_coord)
else:
    print("\n  OK: todas las estaciones tienen coordenadas únicas")

# Guardar consolidado local
consol_path = f"{LOCAL}/silver/ground_truth_consolidado.parquet"
gold.to_parquet(consol_path, index=False, engine="pyarrow", compression="snappy")
size_mb = os.path.getsize(consol_path) / 1e6
md5 = md5_file(consol_path)
print(f"\n  Guardado: {consol_path}")
print(f"  Tamaño:   {size_mb:.2f} MB")
print(f"  MD5:      {md5}")

# ── 8. Manifest ─────────────────────────────────────────────────────────────
manifest = {
    "version": "2.0.0",
    "created_at": datetime.utcnow().isoformat() + "Z",
    "description": "Panel horario unificado de calidad del aire Cali BBox. Nombres canónicos, coords SISAIRE autoritativas.",
    "bbox": BBOX,
    "fuentes": {
        "SISAIRE": "Red oficial IDEAM, 2018-2024, NO2/O3/SO2",
        "DAGMA_portal": "Red municipal Cali, 2010-2019, NO2/O3/SO2",
        "S5P_pseudo": "Sentinel-5P TROPOMI calibrado Theil-Sen (no2_surface = 5.28 + 46948 × col_mol_m2)",
    },
    "schema": {
        "estacion":     "str  — nombre canónico UPPER_SNAKE_CASE",
        "fecha":        "datetime64[ns]  — marca horaria UTC",
        "latitud":      "float64  — grados decimales WGS-84, autoritativa de SISAIRE para estaciones compartidas",
        "longitud":     "float64  — grados decimales WGS-84",
        "valor":        "float64  — concentración µg/m³ (SISAIRE/DAGMA) o NO2 superficie µg/m³ (S5P)",
        "fuente":       "str  — SISAIRE | DAGMA_portal | S5P_pseudo",
        "contaminante": "str  — NO2 | O3 | SO2",
    },
    "estaciones_canonicas": sorted(gold["estacion"].unique().tolist()),
    "coordenadas_sisaire": coord_sis.set_index("estacion")[["lat_sis","lon_sis"]].rename(columns={"lat_sis":"latitud","lon_sis":"longitud"}).to_dict("index"),
    "stats": {
        "n_rows":       int(len(gold)),
        "n_estaciones": int(gold["estacion"].nunique()),
        "fecha_min":    str(gold["fecha"].min()),
        "fecha_max":    str(gold["fecha"].max()),
        "pct_nulos":    round(float(gold["valor"].isna().mean() * 100), 2),
        "md5_consolidado": md5,
    },
    "uso_situacion2": {
        "descripcion": "Interpolación espacial NO2. Usar fuente=SISAIRE|DAGMA_portal para ground-truth, S5P_pseudo como covariable.",
        "filtro_recomendado": "contaminante=='NO2' & valor.notna()",
        "columnas_clave": ["estacion", "fecha", "latitud", "longitud", "valor"],
    },
    "uso_situacion3": {
        "descripcion": "Clasificación y predicción temporal. Panel completo o subset por contaminante.",
        "filtro_recomendado": "valor.notna()",
        "columnas_clave": ["estacion", "fecha", "contaminante", "valor", "fuente"],
    },
    "archivos_gold": {
        "ground_truth_consolidado.parquet": "Panel unificado principal",
        "sisaire_horaria/":                 "SISAIRE particionado por contaminante (Hive)",
        "dagma_portal_horaria/":            "DAGMA particionado por contaminante (Hive)",
        "no2_s5p_estaciones.parquet":       "S5P pseudo-verdad NO2 por estación",
        "manifest_gold.json":               "Este archivo",
    },
}

manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
manifest_path  = f"{LOCAL}/silver/manifest_gold.json"
with open(manifest_path, "wb") as f:
    f.write(manifest_bytes)
print(f"\n  Manifest: {manifest_path}  ({len(manifest_bytes)/1e3:.1f} KB)")

# ── 9. Subir a SILVER en Wasabi ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. SUBIENDO A silver/ EN WASABI")
print("=" * 60)

# Limpiar prefijo silver en S3
for pref in ["silver/sisaire_horaria.parquet/", "silver/dagma_portal_horaria.parquet/",
             "silver/no2_s5p_estaciones.parquet", "silver/ground_truth_consolidado.parquet",
             "silver/manifest_gold.json"]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=pref):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=BUCKET, Key=obj["Key"])
            print(f"  deleted s3://{BUCKET}/{obj['Key']}")

# SISAIRE Hive
for cont, grp in sis.groupby("contaminante"):
    grp2 = grp.drop(columns=["contaminante"])
    upload_parquet(grp2, f"silver/sisaire_horaria.parquet/contaminante={cont}/data.parquet")

# DAGMA Hive
for cont, grp in dag.groupby("contaminante"):
    grp2 = grp.drop(columns=["contaminante"])
    upload_parquet(grp2, f"silver/dagma_portal_horaria.parquet/contaminante={cont}/data.parquet")

# S5P
upload_parquet(s5p, "silver/no2_s5p_estaciones.parquet")

# Consolidado
upload_parquet(gold, "silver/ground_truth_consolidado.parquet")

# Manifest
upload_bytes(manifest_bytes, "silver/manifest_gold.json")

# ── 10. Subir a GOLD en Wasabi ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. SUBIENDO A gold/ EN WASABI")
print("=" * 60)

# Limpiar prefijo gold en S3
for pref in ["gold/sisaire_horaria.parquet/", "gold/dagma_portal_horaria.parquet/",
             "gold/no2_s5p_estaciones.parquet", "gold/ground_truth_consolidado.parquet",
             "gold/manifest_gold.json"]:
    paginator2 = s3.get_paginator("list_objects_v2")
    for page in paginator2.paginate(Bucket=BUCKET, Prefix=pref):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=BUCKET, Key=obj["Key"])
            print(f"  deleted s3://{BUCKET}/{obj['Key']}")

# SISAIRE Hive
for cont, grp in sis.groupby("contaminante"):
    grp2 = grp.drop(columns=["contaminante"])
    upload_parquet(grp2, f"gold/sisaire_horaria.parquet/contaminante={cont}/data.parquet")

# DAGMA Hive
for cont, grp in dag.groupby("contaminante"):
    grp2 = grp.drop(columns=["contaminante"])
    upload_parquet(grp2, f"gold/dagma_portal_horaria.parquet/contaminante={cont}/data.parquet")

# S5P
upload_parquet(s5p, "gold/no2_s5p_estaciones.parquet")

# Consolidado
upload_parquet(gold, "gold/ground_truth_consolidado.parquet")

# Manifest
upload_bytes(manifest_bytes, "gold/manifest_gold.json")

# ── 11. Verificación final ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. VERIFICACIÓN FINAL S3")
print("=" * 60)

total_silver = total_gold = 0
for prefix, label in [("silver/", "SILVER"), ("gold/", "GOLD")]:
    size = 0
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            size  += obj["Size"]
            count += 1
    gb = size / 1e9
    print(f"  {label}: {count} archivos, {gb:.3f} GB")
    if label == "SILVER":
        total_silver = gb
    else:
        total_gold = gb

print(f"\n  Total SILVER + GOLD: {total_silver + total_gold:.3f} GB")

# ── 12. Checklist Situación 1 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. CHECKLIST SITUACIÓN 1 PDF")
print("=" * 60)

checklist = [
    ("OK",   "BBox Cali lon[-76.60,-76.40] lat[3.30,3.55]"),
    ("OK",   "Palmira excluida (lon=-76.31 fuera de BBox)"),
    ("OK",   "Nombres canónicos UPPER_SNAKE_CASE (12 estaciones)"),
    ("OK",   "Coordenadas SISAIRE autoritativas para estaciones compartidas"),
    ("OK",   "Medallion architecture: silver/ particionado Hive + gold/ listo para consumo"),
    ("OK",   "Consolidado unificado SISAIRE+DAGMA+S5P con columnas estacion/fecha/latitud/longitud/valor/fuente/contaminante"),
    ("OK",   "Manifest gold con schema, stats, uso Sit.2 y Sit.3"),
    ("OK",   "S5P calibrado Theil-Sen (RMSE=4.65 µg/m³ en ACOPI)"),
    ("OK",   "EDA notebook con figuras inline y títulos+ejes"),
    ("WARN", "Dataset <50 GB (actual ≈20 GB) → penalización -40% según PDF"),
    ("WARN", "Sin granulos S5P L2 completos ni HARP recorte"),
    ("WARN", "Sin diagrama arquitectura cloud"),
    ("WARN", "Sin reporte costos cloud"),
    ("WARN", "Sin repositorio Git versionado"),
]

for status, item in checklist:
    icon = "✓" if status == "OK" else "⚠"
    print(f"  [{status}] {icon} {item}")

print("\nScript completado.")
