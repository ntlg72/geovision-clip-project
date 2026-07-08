import os
import time
import json
import math
import hashlib
from pathlib import Path

import ee
import geemap
import boto3
import rasterio
from dask import delayed, compute
from google.colab import userdata
from tqdm.notebook import tqdm


# =========================================================
# 1. CONFIG
# =========================================================
PROJECT_ID = "ws-002-456504"
ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print(" Earth Engine autenticado e inicializado")

WASABI_ACCESS_KEY = userdata.get("WASABI_ACCESS_KEY")
WASABI_SECRET_KEY = userdata.get("WASABI_SECRET_KEY")

BUCKET_NAME = "bronze.sat"
ENDPOINT_URL = "https://s3.us-west-1.wasabisys.com"
REGION_NAME = "us-east-1"

LOCAL_DIR = Path("/content/s2_tmp_daily_median")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Carpeta local lista: {LOCAL_DIR}")

S3_PREFIX = "Sentinel-2"
YEARS = list(range(2020, 2025))
BATCH_SIZE = 4
MAX_RETRIES = 3
SLEEP_BETWEEN_TASKS = 1.2

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=WASABI_ACCESS_KEY,
    aws_secret_access_key=WASABI_SECRET_KEY,
    region_name=REGION_NAME
)
print(f"☁️ Cliente Wasabi listo -> bucket: {BUCKET_NAME} | prefix: {S3_PREFIX}")


# =========================================================
# 2. AOI Y TESELADO
# =========================================================
bbox = [-76.60, 3.30, -76.40, 3.55]
lon_min, lat_min, lon_max, lat_max = bbox
aoi = ee.Geometry.Rectangle(bbox)

# Puedes subir esto a 5x4 o 6x4 si el request sigue grande
N_COLS = 5
N_ROWS = 2

lon_step = (lon_max - lon_min) / N_COLS
lat_step = (lat_max - lat_min) / N_ROWS

cuadrantes = []
tile_id = 0

for i in range(N_COLS):
    for j in range(N_ROWS):
        x1 = lon_min + (i * lon_step)
        x2 = x1 + lon_step
        y1 = lat_min + (j * lat_step)
        y2 = y1 + lat_step

        cuadrantes.append({
            "id": tile_id,
            "geom": ee.Geometry.Rectangle([x1, y1, x2, y2]),
            "bbox": [x1, y1, x2, y2]
        })
        tile_id += 1

print(f"🧩 Teselas generadas: {len(cuadrantes)} ({N_COLS}x{N_ROWS})")


# =========================================================
# 3. BANDAS
# =========================================================
BANDS_ALL = [
    "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B11", "B12",
    "SCL"
]
print(f"🛰️ Bandas a exportar: {BANDS_ALL}")


# =========================================================
# 4. UTILIDADES
# =========================================================
def calcular_md5(ruta_archivo):
    h = hashlib.md5()
    with open(ruta_archivo, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def verificar_objeto_s3(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def chunk_list(seq, chunk_size):
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


# =========================================================
# 5. MÁSCARA OPCIONAL
#    Si quieres crudo, deja USE_MASK = False
#    Si quieres filtrado SCL, pon True
# =========================================================
USE_MASK = False

def mask_s2_sr(img):
    scl = img.select("SCL")
    mask = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return img.updateMask(mask).copyProperties(img, img.propertyNames())


# =========================================================
# 6. COLECCIÓN
# =========================================================
def get_s2_collection(year, region):
    print(f"   🔎 Consultando colección Sentinel-2 para {year}...")
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year+1}-01-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
    )

    if USE_MASK:
        col = col.map(mask_s2_sr)

    return col


def get_dates_from_collection(col):
    print("   📅 Extrayendo fechas únicas...")
    dates = sorted(
        col.aggregate_array("system:time_start")
           .map(lambda d: ee.Date(d).format("YYYY-MM-dd"))
           .distinct()
           .getInfo()
    )
    print(f"   ✅ Fechas detectadas: {len(dates)}")
    return dates


def build_daily_image(date_str, collection, region):
    fecha = ee.Date(date_str)
    daily = collection.filterDate(fecha, fecha.advance(1, "day"))

    n_escenas = daily.size()
    tiles = ee.List(daily.aggregate_array("MGRS_TILE")).distinct()
    orbitas = ee.List(daily.aggregate_array("SENSING_ORBIT_NUMBER")).distinct()
    satelites = ee.List(daily.aggregate_array("SPACECRAFT_NAME")).distinct()

    img_daily = daily.median().clip(region)

    img_daily = img_daily.set({
        "fecha": fecha.format("YYYY-MM-dd"),
        "system:time_start": fecha.millis(),
        "n_escenas": n_escenas,
        "tiles": tiles.join(","),
        "orbitas": orbitas.join(","),
        "satelites": satelites.join(","),
        "hacer_mosaico": n_escenas.gt(1),
        "metodo_agregacion": "median"
    })

    return img_daily


# =========================================================
# 7. EXPORTAR + SUBIR UNA TESELA DIARIA
# =========================================================
def procesar_tarea_daily(task):
    year = task["year"]
    date_str = task["date_str"]
    cuadrante_id = task["cuadrante_id"]
    region_geom = task["region_geom"]

    nombre = f"S2_Cali_{date_str}_part{cuadrante_id}.tif"
    ruta_tif = LOCAL_DIR / nombre
    s3_key = f"{S3_PREFIX}/{year}/{nombre}"
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    if verificar_objeto_s3(BUCKET_NAME, s3_key):
        return {
            "status": "exists_s3",
            "nombre": nombre,
            "fecha_adquisicion": date_str,
            "wasabi_path": s3_uri,
            "fuente": "COPERNICUS/S2_SR_HARMONIZED vía Google Earth Engine",
            "tipo_producto": "daily_median_composite",
            "metodo_agregacion": "median"
        }

    try:
        col_year = get_s2_collection(year, aoi)
        img_daily = build_daily_image(date_str, col_year, aoi)

        n_escenas = img_daily.get("n_escenas").getInfo()
        tiles = img_daily.get("tiles").getInfo()
        orbitas = img_daily.get("orbitas").getInfo()
        satelites = img_daily.get("satelites").getInfo()
        hacer_mosaico = img_daily.get("hacer_mosaico").getInfo()

        for intento in range(1, MAX_RETRIES + 1):
            try:
                if ruta_tif.exists():
                    ruta_tif.unlink()

                img_to_export = img_daily.select(BANDS_ALL).toUint16()

                geemap.ee_export_image(
                    img_to_export,
                    filename=str(ruta_tif),
                    scale=10,
                    region=region_geom,
                    file_per_band=False
                )

                if not ruta_tif.exists() or ruta_tif.stat().st_size == 0:
                    raise RuntimeError("Archivo vacío")

                with rasterio.open(ruta_tif) as src:
                    dimensiones = {"ancho": src.width, "alto": src.height}
                    bbox_out = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]

                print(f"      ⬆️ Subiendo {nombre}...")
                s3.upload_file(str(ruta_tif), BUCKET_NAME, s3_key)

                if not verificar_objeto_s3(BUCKET_NAME, s3_key):
                    raise RuntimeError("No se pudo verificar el objeto en Wasabi")

                datos_archivo = {
                    "status": "uploaded",
                    "nombre": nombre,
                    "fecha_adquisicion": date_str,
                    "hash_md5": calcular_md5(ruta_tif),
                    "dimensiones": dimensiones,
                    "bbox_epsg4326": bbox_out,
                    "wasabi_path": s3_uri,
                    "fuente": "COPERNICUS/S2_SR_HARMONIZED vía Google Earth Engine",
                    "tipo_producto": "daily_median_composite",
                    "metodo_agregacion": "median",
                    "n_escenas_dia": n_escenas,
                    "tiles": tiles,
                    "orbitas": orbitas,
                    "satelites": satelites,
                    "hacer_mosaico": bool(hacer_mosaico)
                }

                ruta_tif.unlink()
                return datos_archivo

            except Exception as e:
                if ruta_tif.exists():
                    try:
                        ruta_tif.unlink()
                    except Exception:
                        pass

                if intento == MAX_RETRIES:
                    return {
                        "status": "error",
                        "error": str(e),
                        "nombre": nombre,
                        "fecha_adquisicion": date_str,
                        "cuadrante_id": cuadrante_id
                    }

                time.sleep(2 * intento)
    finally:
        time.sleep(SLEEP_BETWEEN_TASKS)


# =========================================================
# 8. MANIFIESTO
# =========================================================
def guardar_y_subir_manifiesto(manifiesto, year):
    ruta_manifest = LOCAL_DIR / f"manifest_daily_median_{year}.json"
    with open(ruta_manifest, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)

    key = f"{S3_PREFIX}/{year}/{ruta_manifest.name}"
    s3.upload_file(str(ruta_manifest), BUCKET_NAME, key)

    if ruta_manifest.exists():
        ruta_manifest.unlink()

    print(f"   📄 Manifiesto subido: s3://{BUCKET_NAME}/{key}")


# =========================================================
# 9. PROCESAMIENTO POR AÑO
# =========================================================
def procesar_anio(year):
    print(f"\n{'='*90}")
    print(f"🚀 PROCESANDO AÑO {year}")
    print(f"{'='*90}")

    col = get_s2_collection(year, aoi)
    dates = get_dates_from_collection(col)
    tareas = []

    print("   🧱 Construyendo lista de tareas...")
    for date_idx, date_str in enumerate(dates, start=1):
        daily = col.filterDate(ee.Date(date_str), ee.Date(date_str).advance(1, "day"))
        n_escenas = daily.size().getInfo()
        print(f"      [{date_idx}/{len(dates)}] {date_str} -> {n_escenas} escenas")

        for c in cuadrantes:
            tareas.append({
                "year": year,
                "date_str": date_str,
                "cuadrante_id": c["id"],
                "region_geom": c["geom"]
            })

    print(f"   ✅ Tareas construidas: {len(tareas)}")

    manifiesto = {
        "metadata_proyecto": {
            "sensor": "Sentinel-2 L2A",
            "dataset": "COPERNICUS/S2_SR_HARMONIZED",
            "area": "Cali Metropolitana",
            "bbox_general_epsg4326": bbox,
            "bucket": BUCKET_NAME,
            "year": year,
            "cloud_filter": "CLOUDY_PIXEL_PERCENTAGE < 60",
            "scl_mask_aplicada": USE_MASK,
            "scl_incluida_como_banda": True,
            "bandas_exportadas": BANDS_ALL,
            "teselas": len(cuadrantes),
            "grid_shape": [N_COLS, N_ROWS],
            "tipo_producto": "daily_median_composite",
            "metodo_agregacion": "median"
        },
        "archivos": [],
        "errores": []
    }

    total_batches = math.ceil(len(tareas) / BATCH_SIZE)
    print(f"   📦 Total batches: {total_batches} | tamaño batch: {BATCH_SIZE}")

    pbar = tqdm(total=total_batches, desc=f"Año {year}", unit="batch")

    for batch_idx, batch in enumerate(chunk_list(tareas, BATCH_SIZE), start=1):
        print(f"\n   🔄 Ejecutando batch {batch_idx}/{total_batches} con {len(batch)} tareas")

        trabajos = [delayed(procesar_tarea_daily)(task) for task in batch]
        resultados = compute(*trabajos, scheduler="threads")

        n_ok, n_exist, n_err = 0, 0, 0

        for r in resultados:
            if r.get("status") == "error":
                manifiesto["errores"].append(r)
                n_err += 1
            else:
                manifiesto["archivos"].append(r)
                if r.get("status") == "exists_s3":
                    n_exist += 1
                else:
                    n_ok += 1

        print(f"   ✅ Batch {batch_idx}: subidos={n_ok}, existentes={n_exist}, errores={n_err}")
        pbar.update(1)

    pbar.close()

    print(f"   📁 Archivos en manifiesto: {len(manifiesto['archivos'])}")
    print(f"   ⚠️ Errores en manifiesto: {len(manifiesto['errores'])}")

    guardar_y_subir_manifiesto(manifiesto, year)
    print(f"✅ Año {year} finalizado")


# =========================================================
# 10. EJECUCIÓN
# =========================================================
print(f"📂 Verificando carpeta local: {LOCAL_DIR.exists()} -> {LOCAL_DIR}")

for year in YEARS:
    procesar_anio(year)

print("\n Proceso terminado")