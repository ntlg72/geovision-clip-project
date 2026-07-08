# Situación 1 — Construcción del Panel Espacio-Temporal (lado DAGMA + SISAIRE)

Cierre del componente in-situ del proyecto GeoVision-CLIP Cali. Esta lectura
cubre todo lo que el PDF exige para Sit. 1 desde el ground truth y resuelve
las limitaciones identificadas en los análisis previos.

## 1. Artefactos materializados
### Cambio v1.2: Coordenadas ACOPI corregidas (GPS en campo)
La estación SISAIRE etiquetada "ESTACION YUMBO" fue renombrada a **ACOPI** en v1.1.
En v1.2 se corrigieron las coordenadas a las medidas GPS en campo:
**3°30'06.0"N 76°30'27.2"W = (3.501678, -76.507554)**.
Todos los silvers, manifests y artefactos S5P fueron recalculados. El cambio está documentado
en los manifests bajo `cambio_v1.1_a_v1.2`.

Tres silvers + un silver consolidado, todos con MD5 reproducible:

| Silver                                  | Filas      | Tamaño   | MD5                                |
|-----------------------------------------|-----------|---------|--------------------------------------|
| `silver/sisaire_horaria.parquet`        | 456,616   | 6.53 MB | 8b05f294e85bb578cefb0cf046968098     |
| `silver/dagma_portal_horaria.parquet`   | 815,110   | 9.68 MB | 7e20724f37e8d68b2efd788bf686446a     |
| `silver/no2_s5p_estaciones.parquet`     | 7,922     | 0.18 MB | 2e89d2133a84d3202ad9290c06a36665     |
| `silver/ground_truth_consolidado.parquet` | 1,279,648 | 5.68 MB | b5610d4d645e6722dc47e068728a9555   |

Manifests JSON en `silver/manifest_*.json` con schema, particiones, umbrales,
calibraciones, rango temporal y MD5.

El silver consolidado integra:
- DAGMA portal (pre-2020-01) — 7 estaciones, históricos 2010-2019.
- SISAIRE (2020-2024) — 12 estaciones DAGMA + CVC.
- S5P pseudo-truth NO2 (2020-2024) — 12 estaciones, columna troposférica
  calibrada a superficie.

## 2. Cobertura del requisito PDF Sit. 1

| Requisito PDF                                         | Estado | Evidencia                                              |
|-------------------------------------------------------|--------|--------------------------------------------------------|
| Manifest JSON con MD5                                 | Cumple | 4 manifests en `silver/manifest_*.json`                |
| ETL distribuido                                       | Parcial | Pipeline modular; falta paralelizar con Dask para S5P/S2 cuando se descarguen |
| Recorte HARP S5P                                      | N/A para in-situ | Aplica a Sit. 1 satelital, no a DAGMA/SISAIRE |
| Zarr/Parquet particionado                             | Cumple | Parquet particionado por contaminante + fuente         |
| Almacenamiento cloud                                  | Cumple | Bronze en Wasabi `s3://bronze.sat/`                    |
| ≥ 8 visualizaciones del panel                         | Cumple | 19 figuras en `eda_sisaire/`                           |
| Distribución temporal de adquisiciones                | Cumple | Figura 18                                              |
| Mapa de cobertura espacial efectiva                   | Cumple | Figura 17                                              |
| Serie temporal media por estación de cada contaminante | Cumple | Figura 16 (con IC 95% por mes)                        |
| Discusión de cuellos de botella                       | Cumple | Sección 5 de este README                              |

## 3. Limitaciones resueltas

### 3.1 NO2 con solo 3 estaciones in-situ (Univalle 2013-19, ACOPI+Movil+Palmira 2020-24)

**Resuelta** con pseudo-ground-truth NO2 desde Sentinel-5P sobre las 12
estaciones (×10 en volumen para LOO-CV).

- Calibración Theil-Sen robusta sobre 212 días-estación de overlap.
- `no2_surface = 5.28 + 46,948 · no2_col_mol_m2` (coords corregidas v1.2).
- Validación cruzada ACOPI: RMSE = 4.65 µg/m³ (cumple KPI mínimo PDF ≤ 8).
- Validación cruzada MOVIL: RMSE = 1.93 µg/m³.
- Limitación residual: 4 estaciones de Cali central caen en la misma celda
  S5P (resolución 7 km); tratar como cluster en LOO-CV.

Detalles en [eda_sisaire/README_no2_s5p.md](eda_sisaire/README_no2_s5p.md).

### 3.2 Ventana 2020-2024 sub-óptima

**Resuelta** con análisis de ventanas deslizantes 5-year. La ventana
**2019-2023** es ~71% mejor en cobertura O3 LOO-CV que 2020-2024. El
silver actual cubre 2020-2024 + DAGMA portal 2010-2019, dando flexibilidad
para seleccionar 2019-2023 en Sit. 3 sin re-descarga.

Detalles en [eda_sisaire/README_ventanas_5anios.md](eda_sisaire/README_ventanas_5anios.md).

### 3.3 DAGMA portal stale (2010-2019)

**Resuelta** integrándolo al silver consolidado como fuente histórica. Las
12 series del DAGMA portal pasaron por el mismo pipeline (Hampel +
stuck + dedup + imputación corta) con 13 series aceptadas:
ESCUELA_ARGENTINA, LA_FLORA, COMPARTIR, UNIVALLE, PANCE, LA_ERMITA,
CAÑAVERALEJO (algunas con varios contaminantes).

### 3.4 Inconsistencia DAGMA portal vs SISAIRE

**Identificada y documentada** (no resoluble sin acceso a documentación de
calibración). Análisis distribucional revela diferencias sistemáticas:

| Estación      | Contaminante | Mediana DAGMA portal | Mediana SISAIRE | Ratio |
|---------------|--------------|----------------------|-----------------|-------|
| PANCE         | O3           | 19.04 µg/m³          | 4.60            | 4.14  |
| LA FLORA      | SO2          | 11.49                | 4.14            | 2.78  |
| UNIV. VALLE   | O3           | 19.64                | 7.67            | 2.56  |
| COMPARTIR     | O3           | 18.05                | 8.97            | 2.01  |
| LA FLORA      | O3           | 6.80                 | 5.00            | 1.36  |
| LA ERMITA     | SO2          | 3.50                 | 4.03            | 0.87  |
| CAÑAVERALEJO  | SO2          | 1.15                 | 2.70            | 0.43  |

Causa probable: cambio de sensor / recalibración en el cambio 2019→2020,
o cambio de unidad reportada (ppb vs µg/m³ en algunos canales).

**Mitigación**: usar las dos eras como series independientes en Sit. 3
(no concatenar como serie continua), y reportar la inconsistencia como
hallazgo metodológico. Ningún modelo debería asumir la misma calibración
entre ambas eras.

Tablas: `eda_sisaire/tabla_solape_dagma_sisaire.csv` y
`eda_sisaire/tabla_comparativa_dagma_sisaire.csv`.

## 4. Análisis avanzados ejecutados

### 4.1 Clustering K-Means de estaciones (k=2 óptimo, silhouette = 0.44)

Sobre features estadísticos: media, mediana, std, p95, p99, amplitud del
ciclo diurno, hora del pico, autocorrelación lag-24h. Dos clusters
tipológicos:

- **Cluster 0**: perfiles de baja-media intensidad y bajo ciclo diurno
  (estaciones residenciales, periféricas).
- **Cluster 1**: perfiles de alta intensidad y ciclo diurno fuerte
  (estaciones urbanas/industriales).

Detalles: [eda_sisaire/19_kmeans_clusters_estaciones.png](eda_sisaire/19_kmeans_clusters_estaciones.png),
[eda_sisaire/tabla_kmeans_clusters.csv](eda_sisaire/tabla_kmeans_clusters.csv).

### 4.2 Tests de tendencia Mann-Kendall + change-point Pelt (RBF)

Hallazgos clave:

- **NO2 disminuye** en el período DAGMA portal (UNIV. VALLE -0.9 µg/m³/año,
  ESCUELA_ARGENTINA -4.8 µg/m³/año).
- **NO2 aumenta post-COVID** (ACOPI +9.1 µg/m³/año, MOVIL +4.8) — recuperación
  del tráfico vehicular.
- **O3 disminuye en DAGMA portal**, **aumenta en SISAIRE** — comportamiento
  consistente con química troposférica reducida NOx (titration effect).
- **SO2 disminuye** en estaciones antiguas (CAÑAVERALEJO -1.2, LA_ERMITA -0.7
  µg/m³/año), **aumenta en post-2020** (ACOPI +8.3, BASE_AÉREA +2.7) — apunta
  a reactivación industrial Yumbo-Acopi.
- Múltiples cambios estructurales detectados; los más relevantes coinciden
  con 2020-2022 (post-COVID).

Tabla completa: [eda_sisaire/tabla_tendencias_mk.csv](eda_sisaire/tabla_tendencias_mk.csv).

### 4.3 Validación variográfica + Moran I exploratorio

Ya documentado en [eda_sisaire/reporte_limitantes.md](eda_sisaire/reporte_limitantes.md).

## 5. Cuellos de botella detectados

| Cuello de botella | Impacto | Mitigación |
|-------------------|---------|------------|
| Calibración entre eras (ratio mediana 2-4×) | LOO-CV no puede tratar las dos eras como un continuum | Entrenar y validar por era; reportar como limitación |
| 4 estaciones centrales en misma celda S5P | LOO-CV con pseudo-truth de baja independencia para esas 4 | Agrupar como cluster, reportar n=9 efectivos en vez de 12 |
| Cobertura NO2 in-situ desigual (Univalle 2013-19; CVC 2020+) | Cero solapamiento temporal entre eras | TSCV + S5P pseudo-truth (ambas mitigaciones aplicadas) |
| Series con stuck values >100h (BASE AÉREA, CAÑAVERALEJO) | Posible bias en estadística | Flag_stuck activo, datos descartables vía mascara |
| Bronze DAGMA portal stale (2020-01-01 último día) | No aporta a 2020-2024 | Confirmado vía SISAIRE que DAGMA reportó a IDEAM hasta 2023-09 |

## 6. Inventario de visualizaciones (19 figuras)

| #  | Archivo                                       | Sit. PDF |
|----|------------------------------------------------|----------|
| 01 | 01_cobertura_temporal.png                      | 1        |
| 02 | 02_mapa_estaciones.png                         | 1        |
| 03 | 03_distribuciones.png                          | 1, 2     |
| 04 | 04_ciclo_diurno.png                            | 1        |
| 05 | 05_estacionalidad.png                          | 1        |
| 06 | 06_correlaciones.png                           | 3        |
| 07 | 07_solapamiento_temporal.png                   | 3        |
| 08 | 08_outliers.png                                | 1        |
| 09 | 09_acf_pacf.png                                | 3        |
| 10 | 10_variograma_exploratorio.png                 | 3        |
| 11 | 11_moran_scatterplot.png                       | 3        |
| 12 | 12_ventanas_5anios.png                         | 1        |
| 13 | 13_calibracion_s5p_no2.png                     | 3        |
| 14 | 14_cobertura_no2_s5p.png                       | 1, 3     |
| 15 | 15_timeseries_no2_acopi.png                    | 3        |
| 16 | 16_series_temporales_estaciones.png            | 1 (req)  |
| 17 | 17_cobertura_espacial_efectiva.png             | 1 (req)  |
| 18 | 18_distribucion_temporal_adquisiciones.png     | 1 (req)  |
| 19 | 19_kmeans_clusters_estaciones.png              | 3        |

## 7. Inventario de tablas (17 CSV)

`eda_sisaire/tabla_*.csv` + `log_limpieza_*.csv`. Las nuevas en este cierre:

- `tabla_solape_dagma_sisaire.csv`
- `tabla_comparativa_dagma_sisaire.csv`
- `tabla_kmeans_clusters.csv`
- `tabla_tendencias_mk.csv`
- `log_limpieza_dagma_portal.csv`

## 8. Lo que queda para cerrar Sit. 1 completa

Lo que pertenece al lado satelital y NO a DAGMA/SISAIRE (fuera del alcance
de este cierre, según solicitud del usuario):

1. Descargar S5P L2 con HARP y persistir a Zarr.
2. Descargar Sentinel-2 L2A 13 bandas para 4 tiles.
3. Descargar MAIAC AOD y ERA5-Land.
4. Implementar ETL distribuido con Dask.
5. Verificar que el panel completo supera 50 GB.
6. Documentar costos cloud.

Todo lo anterior depende de credenciales de GEE/CDSE/NASA Earthdata que el
equipo debe configurar.

## 9. Reportes complementarios

- [eda_sisaire/reporte_limitantes.md](eda_sisaire/reporte_limitantes.md) —
  reporte autogenerado sobre limitantes vs requisitos PDF.
- [eda_sisaire/README_ventanas_5anios.md](eda_sisaire/README_ventanas_5anios.md) —
  selección de la franja temporal óptima.
- [eda_sisaire/README_no2_s5p.md](eda_sisaire/README_no2_s5p.md) —
  pseudo-ground-truth NO2 desde Sentinel-5P.
