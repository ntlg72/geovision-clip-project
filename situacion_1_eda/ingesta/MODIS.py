import ee, geemap, os, time, json, hashlib, boto3, rasterio
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask
from dask.diagnostics import ProgressBar
from itertools import islice

# --- 1. AUTENTICACIÓN E INICIALIZACIÓN ---
ee.Authenticate()
ee.Initialize(project="ws-002-456504")

# --- 2. CONFIGURACIÓN DE GEOMETRÍA (ejemplo: Cali) ---
bbox = [-76.60, 3.30, -76.40, 3.55]
aoi = ee.Geometry.Rectangle(bbox)

# --- 3. CONFIGURACIÓN WASABI ---
BUCKET_NAME = "bronze.sat"
ENDPOINT_URL = 'https://s3.us-west-1.wasabisys.com'
REGION_NAME = 'us-east-1'

s3 = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=WASABI_ACCESS_KEY,
    aws_secret_access_key=WASABI_SECRET_KEY,
    region_name=REGION_NAME
)

def calcular_md5(ruta_archivo):
    h = hashlib.md5()
    with open(ruta_archivo, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

# --- 4. FUNCIÓN PARA CONSTRUIR IMAGEN DIARIA ---
def build_daily_image(date_str, collection, region):
    fecha = ee.Date(date_str)
    daily = collection.filterDate(fecha, fecha.advance(1, "day"))
    n_escenas = daily.size()

    img_daily = daily.median().clip(region)
    img_daily = img_daily.set({
        "fecha": fecha.format("YYYY-MM-dd"),
        "system:time_start": fecha.millis(),
        "n_escenas": n_escenas,
        "hacer_mosaico": n_escenas.gt(1)
    })
    return img_daily

# --- 5. EXPORTACIÓN CON DASK ---
@dask.delayed
def export_daily(date_str, col, year):
    try:
        img_daily = build_daily_image(date_str, col, aoi)
        fecha = img_daily.get("fecha").getInfo()
        nombre_archivo = f"/content/modis_mcd19a2_v2/MODIS_MAIAC_MEDIAN_{year}_{fecha}.tif"

        if os.path.exists(nombre_archivo):
            print(f"⏭️ Ya existía: {nombre_archivo}")
            return nombre_archivo

        geemap.ee_export_image(
            img_daily,
            filename=nombre_archivo,
            scale=1000,
            region=aoi,
            file_per_band=False
        )

        if os.path.exists(nombre_archivo) and os.path.getsize(nombre_archivo) > 0:
            print(f"✅ Descargado: {nombre_archivo}")
        else:
            if os.path.exists(nombre_archivo):
                os.remove(nombre_archivo)
            print(f"⚠️ Archivo vacío eliminado: {nombre_archivo}")

        return nombre_archivo

    except Exception as e:
        print(f"⚠️ Error en {date_str}: {e}")
        return None

# --- 6. SUBIDA A WASABI ---
def subir_archivo(ruta_tif):
    nombre = Path(ruta_tif).name
    try:
        with rasterio.open(ruta_tif) as src:
            dimensiones = {"ancho": src.width, "alto": src.height}
            bbox = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            try:
                fecha = nombre.split('_')[3].replace(".tif","")
                anio = fecha.split('-')[0]
            except:
                fecha, anio = "unknown", "2020"

        s3_key = f"modis_mcd19a2_v2/{anio}/{nombre}"
        s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

        s3.upload_file(str(ruta_tif), BUCKET_NAME, s3_key)
        print(f"⬆️ Subido {nombre} a {s3_uri}")

        return {
            "nombre": nombre,
            "fecha_adquisicion": fecha,
            "hash_md5": calcular_md5(ruta_tif),
            "dimensiones": dimensiones,
            "bbox_epsg4326": bbox,
            "wasabi_path": s3_uri,
            "fuente": "MODIS/061/MCD19A2_GRANULES vía Google Earth Engine"
        }
    except Exception as e:
        print(f"⚠️ Error subiendo {nombre}: {e}")
        return None

def procesar_y_subir(batch_size=20, sleep_time=2):
    archivos_tif = list(Path("/content/modis_mcd19a2_v2").glob("*.tif"))
    print(f"📂 Encontrados {len(archivos_tif)} archivos para subir...")

    manifiesto = {
        "metadata_proyecto": {
            "sensor": "MODIS MCD19A2 (MAIAC AOD)",
            "producto_origen": "MODIS/061/MCD19A2_GRANULES",
            "area": "Cali Metropolitana",
            "bucket": BUCKET_NAME,
            "carpeta_destino": "modis_mcd19a2_v2",
            "generado_en": datetime.now().isoformat()
        },
        "archivos": []
    }

    for i in range(0, len(archivos_tif), batch_size):
        batch = archivos_tif[i:i+batch_size]
        print(f"\n🚀 Procesando lote {i//batch_size+1} con {len(batch)} archivos...")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(subir_archivo, ruta) for ruta in batch]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    manifiesto["archivos"].append(result)

        print(f"⏸️ Esperando {sleep_time} segundos antes del siguiente lote...")
        time.sleep(sleep_time)

    # Guardar y subir manifiesto único
    anio = datetime.now().year
    nombre_manifiesto = f"manifest_modis_mcd19a2_{anio}.json"
    with open(nombre_manifiesto, "w") as f:
        json.dump(manifiesto, f, indent=4)

    key_manifiesto = f"modis_mcd19a2_v2/{anio}/{nombre_manifiesto}"
    s3.upload_file(nombre_manifiesto, BUCKET_NAME, key_manifiesto)

    print(f"\n✅ PROCESO FINALIZADO")
    print(f"📄 Manifiesto único disponible en: s3://{BUCKET_NAME}/{key_manifiesto}")

# --- 7. LOOP MULTIANUAL ---
os.makedirs("/content/modis_mcd19a2_v2", exist_ok=True)

for year in range(2020, 2025):  # 2020–2024 inclusive
    print(f"\n🚀 Procesando año {year}...\n")

    col = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year+1}-01-01")
        .select(["Optical_Depth_055", "AOD_QA"])
    )

    dates = (
        col.aggregate_array("system:time_start")
           .map(lambda d: ee.Date(d).format("YYYY-MM-dd"))
           .distinct()
           .getInfo()
    )

    print(f"📅 Detectadas {len(dates)} fechas útiles en {year}.\n")

    batch_size = 10 if len(dates) <= 200 else 20 if len(dates) <= 500 else 30
    print(f"⚙️ Usando batch_size={batch_size}")

    def chunked(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, size))
            if not batch:
                break
            yield batch

    for batch in chunked(dates, batch_size):
        tasks = [export_daily(date_str, col, year) for date_str in batch]
        with ProgressBar():
            dask.compute(*tasks, scheduler="threads")

# --- 8. SUBIDA FINAL A WASABI ---
procesar_y_subir(batch_size=20, sleep_time=3)
