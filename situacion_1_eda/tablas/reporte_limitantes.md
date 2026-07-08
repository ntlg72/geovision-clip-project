# Reporte EDA SISAIRE — Hallazgos y conexión con el proyecto

_Generado automáticamente desde Wasabi `bronze.sat/IDEAM/`_

## 1. Resumen general

- **Estaciones únicas**: 12
- **Autoridades**: ['CVC', 'DAGMA']
- **Contaminantes**: ['NO2', 'O3', 'SO2']
- **Mediciones totales (bronze)**: 164,602
- **Mediciones post-S5P (>= 2018-04-30)**: 164,602 (100.0%)
- **Rango temporal**: 2020-01-01 00:00:00 a 2024-12-31 23:00:00
- **Filas en silver layer**: 456,616 (frecuencia horaria estricta)
- **Series aceptadas para LOO-CV**: 21 de 21

## 2. Cobertura por contaminante (post-Sentinel-5P)

| Contaminante | N estaciones in_bbox_propuesto | N externas | Mediciones post-S5P |
|---|---|---|---|
| NO2 | 1 | 2 | 13,496 |
| SO2 | 6 | 2 | 51,483 |
| O3 | 8 | 2 | 99,623 |

## 3. Conexión con requisitos del proyecto

### 3.1 LOO-CV espacial (Situación 3)

- **NO2**: [IMPOSIBLE] N=1 — requiere data adicional
- **SO2**: [VIABLE robusto] N=6
- **O3**: [VIABLE robusto] N=8

### 3.2 Pseudo-labels para GeoVision-CLIP (Situación 2)

| Contaminante | p25 | p50 | p75 | p90 | p99 | Unidad |
|---|---|---|---|---|---|---|
| NO2 | 4.4 | 6.8 | 10.3 | 14.5 | 31.2 | ug/m3 |
| SO2 | 0.7 | 3.4 | 6.9 | 13.3 | 58.7 | ug/m3 |
| O3 | 3.0 | 7.8 | 17.9 | 36.8 | 94.9 | ug/m3 |

### 3.3 Kriging Espacio-Temporal (Situación 3)

- **NO2**: N=3, distancias inter-estación 3.1-23.3 km (mediana 22.2 km) → variograma estimable hasta ~12 km de rango.
- **SO2**: N=8, distancias inter-estación 1.9-30.7 km (mediana 9.1 km) → variograma estimable hasta ~15 km de rango.
- **O3**: N=10, distancias inter-estación 0.5-36.4 km (mediana 13.2 km) → variograma estimable hasta ~18 km de rango.

#### 3.3.1 Variograma exploratorio — modelo de menor RMSE

| Contaminante | Modelo | RMSE | Nugget | Sill | Rango |
|---|---|---|---|---|---|
| SO2 | spherical | 6.100 | 0.000 | 9.078 | 0.007 |
| O3 | spherical | 15.068 | 0.000 | 19.792 | 0.032 |

#### 3.3.2 Moran I global (KPI Situación 3 = > 0.30 con p<0.05)

| Contaminante | n | k | Moran I | p_sim | HH sig | LL sig |
|---|---|---|---|---|---|---|
| SO2 | 8 | 4 | -0.1843 | 0.365 | 0 | 0 |
| O3 | 10 | 4 | -0.3608 | 0.003 | 0 | 0 |

### 3.4 Estacionariedad temporal (insumo ConvLSTM)

p-valores ADF y KPSS sobre las 6 series con mayor cobertura limpia. ADF rechaza H0 (no estacionaria) si p<0.05; KPSS rechaza H0 (estacionaria) si p<0.05. Coincidencia ADF<0.05 y KPSS>0.05 = serie estacionaria.

| Estación | Contaminante | Transformación | n | ADF p | KPSS p |
|---|---|---|---|---|---|
| PALMIRA | O3 | original | 26942 | 0.0 | 0.01 |
| PALMIRA | O3 | diff_1 | 26941 | 0.0 | 0.1 |
| ACOPI | O3 | original | 19283 | 0.0 | 0.01 |
| ACOPI | O3 | diff_1 | 19282 | 0.0 | 0.1 |
| MOVIL | O3 | original | 16555 | 0.0 | 0.01 |
| MOVIL | O3 | diff_1 | 16554 | 0.0 | 0.1 |
| ACOPI | SO2 | original | 14350 | 0.0 | 0.01 |
| ACOPI | SO2 | diff_1 | 14349 | 0.0 | 0.1 |
| COMPARTIR | O3 | original | 8872 | 0.0 | 0.01 |
| COMPARTIR | O3 | diff_1 | 8871 | 0.0 | 0.1 |
| ERA OBRERO | O3 | original | 8808 | 0.0 | 0.01 |
| ERA OBRERO | O3 | diff_1 | 8807 | 0.0 | 0.1 |

## 4. Hallazgos del pre-procesamiento (silver layer)

- **Series excluidas** (no_aceptadas): 0 de 21 por `n_validos_finales < 100` o `dias_cobertura < 90`.
- **% obs. flageadas como outlier o stuck** (sobre el bruto): 15.50%
- **% obs. imputadas** (gaps < 6h, sobre el final): 23.14%
- **Series con stuck values relevantes (>100 horas)**:
  - ACOPI (SO2): 313 horas pegadas → revisar sensor.
  - BASE AÉREA (O3): 1086 horas pegadas → revisar sensor.
  - BASE AÉREA (SO2): 1371 horas pegadas → revisar sensor.
  - CAÑAVERALEJO (SO2): 2713 horas pegadas → revisar sensor.
  - COMPARTIR (O3): 103 horas pegadas → revisar sensor.
  - ERA OBRERO (O3): 262 horas pegadas → revisar sensor.
  - LA ERMITA (SO2): 1631 horas pegadas → revisar sensor.
  - MOVIL (O3): 624 horas pegadas → revisar sensor.
  - PALMIRA (O3): 299 horas pegadas → revisar sensor.
  - PANCE (O3): 124 horas pegadas → revisar sensor.
  - UNIVERSIDAD DEL VALLE (O3): 124 horas pegadas → revisar sensor.

## 5. Limitantes identificadas

### 5.1 NO2 con cobertura insuficiente para LOO-CV puro
- Solo **1** estación(es) SISAIRE post-S5P en el BBox propuesto.
- 2 estación(es) externas disponibles para validación adicional.
- DAGMA Cali no reporta NO2 a SISAIRE (cobertura desde portal DAGMA hasta 2019-12).
- **Mitigación**: TSCV temporal sobre Yumbo + validación OOD en Univalle archivada (DAGMA portal).

### 5.2 Series cortas (cobertura < 1 año post-S5P)

| Estación | Contaminante | Días cobertura | n post-S5P |
|---|---|---|---|
| TRANSITORIA-NAVARRO | O3 | 313 | 4,138 |
| PALMIRA | SO2 | 330 | 5,634 |
| MOVIL | SO2 | 261 | 4,902 |
| LA FLORA | SO2 | 326 | 4,165 |
| TRANSITORIA-NAVARRO | SO2 | 209 | 3,297 |

### 5.3 Heterogeneidad de ventanas temporales entre series
- **CVC**: cierres entre 2021-07-20 y 2024-12-31
- **DAGMA**: cierres entre 2020-11-22 y 2023-09-14
- **Implicación**: el LOO-CV puro solo es defendible cuando cada estación contribuye en su propia ventana de observación; no se requiere intersección global.

## 6. Plan de uso recomendado en el pipeline

1. **Consumir el silver layer** desde `silver/sisaire_horaria.parquet` (NO releer del bronze) — el manifest expone el MD5 verificable.
2. **Trabajar sobre `valor_imputado`** en Sit. 2 y Sit. 3; preservar `valor` y los flags para análisis de sensibilidad.
3. **Etiquetar tiles Sentinel-2** con los percentiles de la sección 3.2.
4. **LOO-CV diferenciado por contaminante**: espacial puro para SO2/O3, TSCV+OOD para NO2.
5. **Variograma**: ajustar sobre los residuos del ConvLSTM, separable por contaminante. Los parámetros de la sección 3.3.1 son referencia inicial.
6. **Moran I de los residuos**: debe ser cero (sin estructura) tras el ConvLSTM+Kriging — comparar contra los valores de la sección 3.3.2 que vienen de la concentración cruda.
7. **Documentar** en el informe la decisión arquitectónica DAGMA-portal + SISAIRE como hallazgo metodológico, no como limitación.
8. **Excluir MOVIL** del Kriging si la auditoría detectó múltiples coordenadas — usable solo como validación OOD.
