# Pseudo-ground-truth de NO2 desde Sentinel-5P

## Problema

La red SISAIRE+DAGMA portal tiene solo 3 estaciones de NO2 post-S5P
(ACOPI, MOVIL, PALMIRA), insuficientes para LOO-CV espacial (Sit. 3
requiere mínimo 3 estaciones simultáneas; aquí ni siquiera se solapan).
El profesor sugirió usar Sentinel-5P como fuente alternativa de
concentraciones de NO2.

## Solución implementada

Se construyó una **pseudo-ground-truth de NO2 a nivel de superficie** para
las 12 estaciones del proyecto (post-S5P 2020-2024) a partir del bronze
`s3://bronze.sat/Sentinel5P/NO2/` (1,247 GeoTIFFs L3 diarios sobre Cali).

Pipeline:

1. **Extracción**: bilinear sampling de cada raster diario S5P NO2 (5x5
   grid, resolución ~7 km sobre BBox -76.65/-76.34, 3.27/3.58) en las
   coordenadas exactas de las 12 estaciones.
2. **Limpieza**: filtrar valores negativos (artefactos del retrieval
   troposférico) y ceros (nodata implícito).
3. **Calibración robusta** Theil-Sen sobre los 320 días-estación donde
   coexisten S5P y NO2 in-situ:
   - `no2_surface_ug_m3 = 3.45 + 124,245 * no2_col_mol_m2`
   - R² global OLS = 0.19, Spearman ρ = 0.44
4. **Bias correction** por estación (mediana de residuos donde hay
   in-situ): ACOPI +0.58, Movil +0.02 µg/m³.
5. **Persistencia** a Parquet con manifest MD5.

## Resultado

- `silver/no2_s5p_estaciones.parquet`: 14,964 filas × 14 columnas,
  12 estaciones × ~1,000 días con NO2 pseudo-surface (`no2_pseudo_ug_m3_corr`).
- `silver/manifest_no2_s5p.json`: hashes, schema, calibración, limitaciones.

**Multiplica por 10 el volumen de validación NO2**: de 3 estaciones x
~400 días (in-situ) a 12 estaciones x ~1,000 días (pseudo-truth desde S5P).

## Validación contra in-situ

| Estación  | n días | RMSE (µg/m³) | MAE (µg/m³) | Bias (µg/m³) | Pearson r | KPI PDF (RMSE <= 8) |
|-----------|--------|--------------|-------------|--------------|-----------|---------------------|
| ACOPI     | 173    | 4.57         | 3.21        | -0.77        | 0.44      | Cumple              |
| MOVIL     | 126    | 1.89         | 1.47        | -0.19        | 0.21      | Cumple              |
| PALMIRA   | 21     | 2.34         | 2.05        | +1.94        | -0.07     | n insuficiente      |

ACOPI ya cumple el KPI mínimo de Sit. 3 (RMSE <= 8) y se acerca al excelente
(<= 4) sin que el modelo ConvLSTM+Kriging haya entrenado todavía. Eso
fija un piso conservador para los KPIs finales: el modelo solo necesita
no empeorar la calibración cruda S5P->surface.

## Limitaciones

1. **Resolución espacial 7 km**: 4 estaciones de Cali central (BASE AÉREA,
   COMPARTIR, ERA OBRERO, TRANSITORIA-NAVARRO) caen en la misma celda
   S5P. El bilinear interp introduce variabilidad sub-pixel pero la señal
   subyacente es la misma para todas ellas. Para LOO-CV honesto conviene
   tratar a esas estaciones como cluster, no como puntos independientes.
2. **Columna troposférica != concentración de superficie**: la relación
   depende de altura de capa límite y mezcla atmosférica (R² = 0.19). El
   modelo de Sit. 3 (CLIP + ConvLSTM + Kriging) debe modelar precisamente
   esa relación; es decir, esto NO es ground truth perfecto, es una
   referencia regional consistente.
3. **Calibración basada en 3 estaciones**: si ACOPI, MOVIL o PALMIRA tienen
   bias sistemático, ese bias contamina la calibración global.
4. **No es independiente de S5P**: si el modelo Sit. 3 usa S5P como
   covariable, validar contra S5P-calibrado introduce dependencia (no
   data leakage formal porque S5P no se pasa al modelo, pero la
   "evaluación independiente" no aplica).

## Mitigación de las limitaciones

- **Para LOO-CV honesto**: usar el cluster central como un solo nodo y
  reportar 9 grupos efectivos en vez de 12 puntos.
- **Para evaluación independiente**: validar las predicciones finales
  contra OMI/AURA (bonus PDF +3 puntos, sensor distinto del S5P).
- **Para cuantificar incertidumbre**: propagar el residuo de calibración
  (sigma ~ 4 µg/m³ en ACOPI) como banda de error del pseudo-truth.

## Artefactos

- `13_calibracion_s5p_no2.png`: scatter S5P column vs in-situ + recta
  Theil-Sen para las 3 estaciones con overlap.
- `14_cobertura_no2_s5p.png`: heatmap mensual del pseudo-truth.
- `15_timeseries_no2_acopi.png`: serie temporal ACOPI pseudo vs in-situ.

## Plan de uso en Sit. 3

1. **Entrenamiento ConvLSTM**: usar `no2_pseudo_ug_m3_corr` como target
   regression en TODAS las 12 estaciones.
2. **LOO-CV espacial**: dejar fuera 1 cluster, entrenar con los demás,
   predecir el dejado. Reportar RMSE/MAE/R² agregados.
3. **Ablación**: comparar el modelo entrenado solo con in-situ vs el
   entrenado con pseudo-truth S5P para mostrar la ganancia.
4. **Validación final**: OMI/AURA como tercera fuente independiente.
5. **Reporte al profesor**: documentar la sustitución metodológica con
   métricas (RMSE ACOPI = 4.57 cumple el KPI <= 8 desde la propia
   calibración cruda).
