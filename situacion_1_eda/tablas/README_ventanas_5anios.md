# Selección de la ventana de 5 años para el proyecto

## Pregunta

El PDF pide 5 años de cobertura. ¿Cuál ventana de 5 años maximiza la densidad
del ground truth (DAGMA + SISAIRE) cumpliendo el requisito de Sentinel-5P
(operacional desde 2018-04-30)?

## Método

Se combinó el bronze del DAGMA portal (`s3://bronze.sat/DAGMA/`, 2010-01 a 2020-01,
limpieza por rango físico) con el silver SISAIRE (`silver/sisaire_horaria.parquet`,
2020-01 a 2024-12, post-Hampel + dedup) y se barrió mes por mes una ventana de
60 meses. Se computaron tres métricas por ventana y por contaminante:

- estaciones con cobertura robusta (>= 30 meses con >= 100 obs/mes)
- meses con >= 3 estaciones simultáneas activas (umbral mínimo para LOO-CV puro)
- observaciones totales

## Resultado

| Ventana            | Post-S5P | NO2 obs | SO2 meses >=3 | O3 meses >=3 | O3 obs |
|--------------------|----------|---------|---------------|--------------|--------|
| 2018-04 a 2023-03  | Sí       | 20,089  | 14            | 48           | 140,912|
| 2019-01 a 2023-12  | Sí       | 17,149  | 14            | 40           | 117,864|
| 2020-01 a 2024-12  | Sí       | 15,434  | 14            | 28           | 111,869|
| 2016-01 a 2020-12  | No       | 20,946  | 10            | 60           | 165,582|

La ventana actual del proyecto (2020-2024) es la peor de las tres opciones
post-S5P. La ventana 2018-04 a 2023-03 gana en todo: +30% observaciones de
O3 y NO2, +71% meses LOO-CV viables para O3, mismo SO2.

## Recomendación

**Adoptar 2019-01 a 2023-12** como ventana del proyecto: 5 años naturales
completos, 100% post-S5P, 14 meses LOO-CV viables para SO2 y 40 para O3.
Frente a 2018-2023 pierde 8 meses de O3 a cambio de una narrativa de "5 años
calendario" más limpia para el informe.

## Limitación estructural

Ningún 5-year window logra >= 3 estaciones de NO2 simultáneas. Univalle
(DAGMA portal) reporta NO2 hasta 2019-12; CVC (SISAIRE) arranca en 2020 con
ACOPI y Palmira. Las dos eras no se solapan. Para NO2, TSCV temporal es la
única opción independientemente de la ventana elegida.

## Plan de acción

1. Extender el bronze SISAIRE para incluir 2019 y re-correr el pipeline silver.
2. Integrar el DAGMA portal limpio como segundo silver
   (`silver/dagma_horaria.parquet`) aplicando el mismo Hampel + stuck + dedup.
3. Concatenar ambos silvers con columna `fuente` para trazabilidad. En
   solapamientos preferir SISAIRE silver (frecuencia horaria estricta).
4. Descargar S5P, S2, ERA5 y MAIAC para 2019-01 a 2023-12.
5. Documentar la elección en la sección 2 del informe con la Figura 12 y
   el reporte de NO2 como limitación conocida (mitigada con TSCV + bonus
   OMI/AURA del PDF).

## Artefactos

- `12_ventanas_5anios.png`: coverage map combinado 2010-2024 con las 3
  ventanas candidatas sombreadas.
- `tabla_ventanas_candidatas.csv`: tabla numérica detrás del veredicto.
