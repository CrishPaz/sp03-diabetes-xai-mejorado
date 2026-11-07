# Benchmark Científico Comparativo de Modelos sobre Cohortes Reales

Resultados obtenidos mediante **5-Fold Stratified Cross-Validation** con preprocesamiento anti-fuga dentro de cada pliegue.

| Cohorte / Dataset | Muestra ($N$) | Tipo de Variables | Algoritmo | ROC-AUC | PR-AUC | F1-Score | Sensibilidad | Especificidad | Brier Score |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CDC NHANES 2017–2018**<br>*(Prev: 16.0%)* | 4829 | Biomarcadores de Laboratorio | **Random Forest** | 0.9422 ± 0.004 | 0.7943 | 0.7261 | 0.8511 | 0.9056 | 0.0832 |
|  |  |  | **XGBoost** | 0.9412 ± 0.005 | 0.8003 | 0.7140 | 0.8381 | 0.9026 | 0.0810 |
|  |  |  | **Regresion Logistica** | 0.9401 ± 0.005 | 0.7738 | 0.6933 | 0.8550 | 0.8832 | 0.0896 |
|  |  |  | **Stacking Hibrido** | 0.9435 ± 0.004 | 0.7947 | 0.7492 | 0.7474 | 0.9527 | 0.0612 |
| **NIDDK Pima Indians**<br>*(Prev: 34.9%)* | 768 | Benchmark Histórico | **Random Forest** | 0.8378 ± 0.018 | 0.7103 | 0.7068 | 0.7874 | 0.7620 | 0.1635 |
|  |  |  | **XGBoost** | 0.8281 ± 0.021 | 0.7156 | 0.6773 | 0.7575 | 0.7420 | 0.1691 |
|  |  |  | **Regresion Logistica** | 0.8362 ± 0.020 | 0.7250 | 0.6688 | 0.7274 | 0.7580 | 0.1673 |
|  |  |  | **Stacking Hibrido** | 0.8350 ± 0.021 | 0.7185 | 0.6583 | 0.6531 | 0.8240 | 0.1569 |
| **CDC BRFSS 2015**<br>*(Prev: 50.0%)* | 70692 | Salud Poblacional y Estilo de Vida | **Random Forest** | 0.8213 ± 0.002 | 0.7962 | 0.7556 | 0.7864 | 0.7048 | 0.1744 |
|  |  |  | **XGBoost** | 0.8296 ± 0.002 | 0.8045 | 0.7619 | 0.7968 | 0.7053 | 0.1674 |
|  |  |  | **Regresion Logistica** | 0.8247 ± 0.002 | 0.7947 | 0.7528 | 0.7675 | 0.7286 | 0.1699 |
|  |  |  | **Stacking Hibrido** | 0.8246 ± 0.002 | 0.7996 | 0.7561 | 0.7823 | 0.7132 | 0.1703 |
| **Diabetes 130-US Hospitals (UCI 296)**<br>*(Prev: 8.8%)* | 71515 | Complicaciones y Reingreso Hospitalario | **Random Forest** | 0.6065 ± 0.006 | 0.1322 | 0.1946 | 0.5557 | 0.5992 | 0.2384 |
|  |  |  | **XGBoost** | 0.6058 ± 0.007 | 0.1341 | 0.1930 | 0.5709 | 0.5808 | 0.2363 |
|  |  |  | **Regresion Logistica** | 0.6042 ± 0.006 | 0.1334 | 0.1944 | 0.5265 | 0.6247 | 0.2401 |
|  |  |  | **Stacking Hibrido** | 0.6070 ± 0.005 | 0.1349 | 0.0000 | 0.0000 | 1.0000 | 0.0792 |
| **Early Stage Diabetes (UCI 529 - Sylhet)**<br>*(Prev: 61.5%)* | 520 | Sintomatología Prodrómica | **Random Forest** | 0.9970 ± 0.001 | 0.9982 | 0.9729 | 0.9594 | 0.9800 | 0.0313 |
|  |  |  | **XGBoost** | 0.9934 ± 0.007 | 0.9959 | 0.9679 | 0.9500 | 0.9800 | 0.0360 |
|  |  |  | **Regresion Logistica** | 0.9753 ± 0.010 | 0.9871 | 0.9347 | 0.9187 | 0.9250 | 0.0575 |
|  |  |  | **Stacking Hibrido** | 0.9920 ± 0.006 | 0.9954 | 0.9654 | 0.9594 | 0.9550 | 0.0365 |
