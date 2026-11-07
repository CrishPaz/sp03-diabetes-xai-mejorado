"""
Base de conocimiento clínico estructurada para el módulo RAG (LangChain / LangFlow).
Contiene fragmentos normativos y directrices de práctica clínica verificables:
- American Diabetes Association (ADA) Standards of Care in Diabetes (2024)
- Organización Mundial de la Salud (OMS) - Directrices de Diagnóstico y Tamizaje
- Centros para el Control y la Prevención de Enfermedades (CDC) - Vigilancia Epidemiológica
"""

from __future__ import annotations

from typing import Any, Dict, List

CLINICAL_GUIDELINES_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "ADA-2024-DIAG-01",
        "source": "ADA Standards of Care in Diabetes 2024 (Capítulo 2: Clasificación y Diagnóstico)",
        "topic": "Criterios Diagnósticos de Diabetes Mellitus",
        "content": (
            "El diagnóstico de diabetes mellitus tipo 2 se establece con cualquiera de los siguientes criterios: "
            "1. Hemoglobina glicosilada (HbA1c) ≥ 6.5% (48 mmol/mol), realizado en laboratorio con método certificado NGSP. "
            "2. Glucemia plasmática en ayunas (FPG) ≥ 126 mg/dL (7.0 mmol/L), tras un ayuno de al menos 8 horas. "
            "3. Glucemia plasmática a las 2 horas ≥ 200 mg/dL (11.1 mmol/L) durante una prueba de tolerancia oral a la glucosa (OGTT con 75 g). "
            "4. En un paciente con síntomas clásicos de hiperglucemia o crisis hiperglucémica, una glucemia plasmática al azar ≥ 200 mg/dL (11.1 mmol/L). "
            "En ausencia de hiperglucemia inequívoca, el diagnóstico requiere dos pruebas anormales de la misma muestra o en dos muestras separadas."
        ),
        "tags": ["diagnostico", "hba1c", "glucosa", "ayuno", "ogtt", "ada_2024"]
    },
    {
        "id": "ADA-2024-PREDIAB-02",
        "source": "ADA Standards of Care in Diabetes 2024 (Capítulo 3: Prevención de Diabetes Tipo 2)",
        "topic": "Criterios de Prediabetes y Riesgo Intermedio",
        "content": (
            "La prediabetes se define por cualquiera de los siguientes parámetros: "
            "1. Glucemia en ayunas alterada (IFG): 100 mg/dL a 125 mg/dL (5.6 a 6.9 mmol/L). "
            "2. Tolerancia a la glucosa alterada (IGT): glucemia a las 2h en OGTT de 140 mg/dL a 199 mg/dL (7.8 a 11.0 mmol/L). "
            "3. HbA1c entre 5.7% y 6.4% (39 a 47 mmol/mol). "
            "Los pacientes con prediabetes tienen un riesgo anual de progresión a diabetes del 5% al 10% y presentan un riesgo cardiovascular elevado. "
            "Se recomienda intervención intensiva en el estilo de vida (pérdida de al menos 7% del peso corporal y ≥ 150 min/semana de actividad física)."
        ),
        "tags": ["prediabetes", "prevencion", "estilo_de_vida", "hba1c", "ada_2024"]
    },
    {
        "id": "ADA-2024-CARDIO-03",
        "source": "ADA Standards of Care in Diabetes 2024 (Capítulo 10: Enfermedad Cardiovascular y Manejo de Riesgo)",
        "topic": "Presión Arterial y Manejo Hemodinámico",
        "content": (
            "Para personas con diabetes e hipertensión arterial, el objetivo de presión arterial en la mayoría de adultos es < 130/80 mmHg, "
            "siempre que se pueda alcanzar de forma segura. "
            "La confirmación de hipertensión requiere mediciones repetidas en múltiples consultas. "
            "El tratamiento inicial debe incluir modificaciones del estilo de vida y farmacoterapia con inhibidores de la ECA, ARA-II, "
            "diuréticos tiazídicos o bloqueadores de los canales de calcio. Para pacientes con albuminuria establecida (relación albúmina/creatinina ≥ 30 mg/g), "
            "los IECA o ARA-II son el tratamiento de primera línea."
        ),
        "tags": ["hipertension", "presion_arterial", "sistolica", "diastolica", "cardiovascular", "ada_2024"]
    },
    {
        "id": "ADA-2024-LIPID-04",
        "source": "ADA Standards of Care in Diabetes 2024 (Capítulo 10: Dislipidemia y Control Lipídico)",
        "topic": "Perfil Lipídico, Triglicéridos y Colesterol HDL",
        "content": (
            "El perfil lipídico aterogénico típico en personas con resistencia a la insulina y DM2 incluye: "
            "hipertrigliceridemia (triglicéridos ≥ 150 mg/dL), colesterol HDL bajo (< 40 mg/dL en hombres, < 50 mg/dL en mujeres) "
            "y predominio de partículas de LDL pequeñas y densas. "
            "La dislipidemia diabética acelera la aterosclerosis macrovascular. Se recomienda tamizaje lipídico al diagnóstico y al menos "
            "anualmente, con terapia con estatinas de moderada o alta intensidad según la estratificación del riesgo cardiovascular global a 10 años."
        ),
        "tags": ["lipidos", "colesterol", "trigliceridos", "hdl", "estatinas", "ada_2024"]
    },
    {
        "id": "ADA-2024-COMPLIC-05",
        "source": "ADA Standards of Care in Diabetes 2024 (Capítulo 11 y 12: Complicaciones Microvasculares)",
        "topic": "Complicaciones Crónicas Micro y Macrovasculares",
        "content": (
            "Las complicaciones crónicas de la diabetes se dividen en: "
            "1. Nefropatía diabética: evaluada mediante tasa de filtración glomerular estimada (eGFR < 60 mL/min/1.73m²) y albuminuria (ACR ≥ 30 mg/g). "
            "2. Retinopatía diabética: requiere fondo de ojo anual dilatado para detectar microaneurismas, exudados y neovascularización. "
            "3. Neuropatía periférica: tamizaje anual con monofilamento de 10 g y evaluación de sensibilidad vibratoria. "
            "4. Complicaciones macrovasculares (MACE): infarto agudo de miocardio, accidente cerebrovascular y enfermedad arterial periférica. "
            "El control intensivo de la glucemia (HbA1c < 7.0%), la presión arterial (< 130/80) y los lípidos reduce drásticamente la tasa de eventos a largo plazo."
        ),
        "tags": ["complicaciones", "nefropatia", "retinopatia", "neuropatia", "mace", "supervivencia", "ada_2024"]
    },
    {
        "id": "OMS-2023-SCREEN-06",
        "source": "OMS Directrices para el Diagnóstico y Manejo de la Diabetes Tipo 2 (2023)",
        "topic": "Tamizaje Poblacional y Factores de Riesgo No Modificables",
        "content": (
            "El tamizaje oportuno en población asintomática está indicado en todos los adultos a partir de los 35-45 años, o antes si presentan: "
            "sobrepeso u obesidad (IMC ≥ 25 kg/m² o ≥ 23 kg/m² en poblaciones asiáticas), antecedentes familiares de primer grado con diabetes, "
            "antecedente de diabetes gestacional, hipertensión diagnosticada o sedentarismo. "
            "Las herramientas predictivas automatizadas basadas en inteligencia artificial deben servir como filtro de estratificación para derivar a "
            "pruebas confirmatorias de laboratorio, sin sustituir el criterio diagnóstico del profesional médico."
        ),
        "tags": ["oms", "tamizaje", "factores_riesgo", "edad", "imc", "antecedente_familiar"]
    },
    {
        "id": "CDC-2023-SURVEIL-07",
        "source": "CDC National Diabetes Statistics Report & Guías de Vigilancia Epidemiológica",
        "topic": "Vigilancia Epidemiológica de Factores Cardiometabólicos",
        "content": (
            "La vigilancia epidemiológica en salud pública clasifica el riesgo metabólico en cuatro estratos operacionales: "
            "Bajo (<25% probabilidad de desbalance), Moderado (25-50%: ventana óptima para prevención primaria), "
            "Alto (50-75%: alta sospecha, requiere confirmación diagnóstica con glucosa y HbA1c) y "
            "Muy Alto (>75%: sospecha crítica, prioridad para confirmación inmediata, evaluación de daño orgánico y prevención de complicaciones). "
            "Cualquier sistema de alerta automatizada debe emitir advertencias de supervisión profesional y trazabilidad del algoritmo empleado."
        ),
        "tags": ["cdc", "vigilancia", "estratos_riesgo", "salud_publica", "alertas"]
    }
]
