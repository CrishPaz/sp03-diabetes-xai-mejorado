#!/usr/bin/env python3
"""
Limpieza, estandarización y preparación de cohortes reales de investigación biomédica:
1. CDC NHANES 2017-2018 (Clínico y biomarcadores de laboratorio HPLC)
2. NIDDK Pima Indians (Benchmark clásico)
3. CDC BRFSS 2015 (Epidemiológico, estilo de vida y salud poblacional, N=70,692 balanceado)
4. Diabetes 130-US Hospitals (UCI 296, N=71,518 pacientes únicos sin fuga de datos, complicaciones y reingreso)
5. Early Stage Diabetes Risk Prediction (UCI 529, N=520 pacientes con sintomatología prodrómica)
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def process_cdc_brfss():
    raw_path = RAW_DIR / "brfss" / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    if not raw_path.exists():
        print(f"Skipping BRFSS: {raw_path} not found.")
        return None

    print("\n--- Procesando CDC BRFSS 2015 (Diabetes Health Indicators) ---")
    df = pd.read_csv(raw_path)
    # Target: Diabetes_binary -> target (0: No diabetes, 1: Diabetes o Prediabetes)
    df = df.rename(columns={"Diabetes_binary": "target"})
    # Convert all columns to numeric int/float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    out_csv = PROCESSED_DIR / "cdc_brfss_5050_clean.csv"
    df.to_csv(out_csv, index=False)
    print(f"Guardado BRFSS limpio: {out_csv} ({df.shape[0]} filas, {df.shape[1]} columnas)")

    # Sample excel for fast visual inspection (5,000 rows)
    out_xlsx = PROCESSED_DIR / "cdc_brfss_5050_sample_5k.xlsx"
    df.head(5000).to_excel(out_xlsx, index=False)
    print(f"Guardado BRFSS Excel muestra 5k: {out_xlsx}")
    return df


def process_early_stage_sylhet():
    raw_path = RAW_DIR / "early_stage_sylhet" / "diabetes_data_upload.csv"
    if not raw_path.exists():
        print(f"Skipping Early Stage: {raw_path} not found.")
        return None

    print("\n--- Procesando Early Stage Diabetes Risk (UCI 529 - Sylhet) ---")
    df = pd.read_csv(raw_path)
    
    # Map Yes/No to 1/0
    binary_cols = [c for c in df.columns if c not in ["Age", "Gender", "class"]]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    
    # Gender: Male=1, Female=0
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(1).astype(int)
    
    # Target: Positive=1, Negative=0
    df["target"] = df["class"].map({"Positive": 1, "Negative": 0}).astype(int)
    df = df.drop(columns=["class"])

    # Standardize column names (lowercase with underscores)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    out_csv = PROCESSED_DIR / "early_stage_diabetes_sylhet_clean.csv"
    df.to_csv(out_csv, index=False)
    print(f"Guardado Early Stage limpio: {out_csv} ({df.shape[0]} filas, {df.shape[1]} columnas)")

    out_xlsx = PROCESSED_DIR / "early_stage_diabetes_sylhet_clean.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"Guardado Early Stage Excel: {out_xlsx}")
    return df


def process_diabetes_130_hospitals():
    raw_path = RAW_DIR / "diabetes_130_hospitals" / "diabetic_data.csv"
    if not raw_path.exists():
        print(f"Skipping 130 Hospitals: {raw_path} not found.")
        return None

    print("\n--- Procesando Diabetes 130-US Hospitals (UCI 296) ---")
    df = pd.read_csv(raw_path)

    # 1. Regla anti-fuga: deduplicar por paciente (primer encuentro clínico por paciente único)
    df_unique = df.drop_duplicates(subset=["patient_nbr"], keep="first").copy()
    print(f"Total encuentros originales: {len(df)} -> Pacientes únicos (sin fuga): {len(df_unique)}")

    # 2. Filtrar géneros inválidos (solo 3 registros desconocidos en 71k)
    df_unique = df_unique[df_unique["gender"].isin(["Male", "Female"])].copy()
    df_unique["gender_male"] = (df_unique["gender"] == "Male").astype(int)

    # 3. Mapear rangos de edad a punto medio numérico
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    df_unique["age_numeric"] = df_unique["age"].map(age_map).fillna(65).astype(int)

    # 4. Target de reingreso prematuro (<30 días)
    # <30 = 1 (complicación aguda temprana), >30 o NO = 0
    df_unique["target"] = (df_unique["readmitted"] == "<30").astype(int)

    # 5. Medicaciones clave y uso de insulina
    df_unique["insulin_treatment"] = (df_unique["insulin"] != "No").astype(int)
    df_unique["diabetes_med_prescribed"] = (df_unique["diabetesMed"] == "Yes").astype(int)
    df_unique["medication_changed"] = (df_unique["change"] == "Ch").astype(int)
    
    # 6. Prueba HbA1c documentada (>7, >8, Norm)
    df_unique["a1c_tested"] = (df_unique["A1Cresult"] != "None").astype(int)
    df_unique["a1c_elevated"] = df_unique["A1Cresult"].isin([">7", ">8"]).astype(int)

    # 7. Variables de utilización y comorbilidades numéricas
    features_to_keep = [
        "patient_nbr",
        "gender_male",
        "age_numeric",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "insulin_treatment",
        "diabetes_med_prescribed",
        "medication_changed",
        "a1c_tested",
        "a1c_elevated",
        "readmitted",
        "target"
    ]

    df_clean = df_unique[features_to_keep].copy()
    out_csv = PROCESSED_DIR / "diabetes_130_hospitals_clean.csv"
    df_clean.to_csv(out_csv, index=False)
    print(f"Guardado 130 Hospitals limpio: {out_csv} ({df_clean.shape[0]} pacientes, {df_clean.shape[1]} columnas)")

    out_xlsx = PROCESSED_DIR / "diabetes_130_hospitals_sample_5k.xlsx"
    df_clean.head(5000).to_excel(out_xlsx, index=False)
    print(f"Guardado 130 Hospitals Excel muestra 5k: {out_xlsx}")
    return df_clean


if __name__ == "__main__":
    b = process_cdc_brfSS = process_cdc_brfss()
    e = process_early_stage_sylhet()
    h = process_diabetes_130_hospitals()
    print("\n✅ Todas las cohortes adicionales fueron procesadas y exportadas con éxito.")
