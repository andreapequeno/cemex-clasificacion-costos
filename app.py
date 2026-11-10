"""
Reclasificaicon automatica de costos CEMEX - By UDEM

Clasifica archivos SAP con el ensamble Soft Voting previamente entrenado.
No entrena el modelo ni aplica SMOTE a los archivos nuevos.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_TITLE = "Reclasificación automática de costos CEMEX - By UDEM"

MODEL_PATH = Path("modelo/modelo_cemex.joblib")
ARTIFACTS_PATH = Path("modelo/codificadores_cemex.joblib")

MAX_FILE_SIZE_MB = 50

# Son las cuatro variables que aparecen en X en el notebook _5modelos_CEMEX.
ORIGINAL_FEATURES = [
    "Denom.clase de coste",
    "Cl.coste",
    "Costo Estandar",
    "Descrip.clases coste",
]

MODEL_FEATURES = [
    "Denom.clase de coste_num",
    "Cl.coste_num",
    "Costo Estandar_num",
    "Descrip.clases coste_num",
]

TARGET_NAME = "Categoría de costo"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
)


def stop_error(message: str) -> None:
    st.error(message)
    st.stop()


def check_access() -> None:
    """
    El acceso a la aplicación se administra mediante
    Streamlit Community Cloud - Sharing.
    """
    return

def show_confidentiality_notice() -> bool:
    st.subheader("Confidencialidad y protección de la información")

    st.warning(
        """
        La información cargada en esta aplicación es de carácter
        confidencial y está destinada exclusivamente al desarrollo
        de las actividades autorizadas del proyecto.
    
        - **Acceso restringido:** únicamente las personas incluidas
          en la lista de usuarios autorizados pueden acceder a la aplicación.
    
        - **Uso de la información:** los archivos se utilizan
          exclusivamente para la clasificación y análisis de los costos
          del área de transporte propio de CEMEX.
    
        - **Confidencialidad:** la información no se comparte, distribuye ni divulga a personas no autorizadas..
    
        - **Responsabilidad:** cada usuario debe contar con autorización
          para cargar y procesar los archivos.
    
        - **Resguardo:** no se deben utilizar los archivos para fines
          distintos a los establecidos en el proyecto.
        """
    )

    return st.checkbox(
        "He leído y acepto el aviso. Confirmo que tengo autorización para "
        "utilizar el archivo que voy a cargar."
    )


@st.cache_resource(show_spinner="Cargando el modelo y sus codificadores...")
def load_model_and_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Verifica que modelo_cemex.joblib esté dentro de la carpeta modelo."
        )

    if not ARTIFACTS_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró {ARTIFACTS_PATH}. "
            "Sube codificadores_cemex.joblib a la carpeta modelo."
        )

    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)

    # El archivo definitivo debe ser el diccionario integrado desde Colab.
    required = {
        "feature_encoders",
        "original_target_encoder",
        "model_target_encoder",
        "feature_order",
    }
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError(
            "El archivo codificadores_cemex.joblib no tiene la estructura "
            "definitiva. Debe contener feature_encoders, "
            "original_target_encoder, model_target_encoder y feature_order. "
            "Vuelve a generarlo con el bloque integrado del notebook 5."
        )

    feature_encoders = artifacts["feature_encoders"]

    missing_encoders = [
        col for col in ORIGINAL_FEATURES if col not in feature_encoders
    ]
    if missing_encoders:
        raise ValueError(
            "Faltan codificadores para las variables predictoras: "
            + ", ".join(missing_encoders)
        )

    if TARGET_NAME not in feature_encoders:
        raise ValueError(
            "No se encontró el codificador original de la variable objetivo "
            "'Categoría de costo'."
        )

    if list(artifacts["feature_order"]) != MODEL_FEATURES:
        raise ValueError(
            "El orden de variables guardado no coincide con el utilizado "
            "por el modelo."
        )

    return model, artifacts


def read_file(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()

    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"El archivo supera el límite de {MAX_FILE_SIZE_MB} MB.")

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".xlsx":
        df = pd.read_excel(BytesIO(content), engine="openpyxl")
    elif suffix == ".csv":
        last_error = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(
                    BytesIO(content),
                    encoding=encoding,
                    low_memory=False,
                )
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(f"No se pudo leer el CSV: {last_error}")
    else:
        raise ValueError("Formato no permitido. Utiliza Excel (.xlsx) o CSV.")

    if df.empty:
        raise ValueError("El archivo no contiene registros.")

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"El archivo contiene columnas duplicadas: {duplicated}")

    return df


def encode_predictors(df: pd.DataFrame, artifacts: dict):
    missing_columns = [col for col in ORIGINAL_FEATURES if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas por el modelo: "
            + ", ".join(missing_columns)
        )

    encoders = artifacts["feature_encoders"]
    encoded = pd.DataFrame(index=df.index)
    valid_mask = pd.Series(True, index=df.index)
    unknown_summary = {}

    for original_col, numeric_col in zip(ORIGINAL_FEATURES, MODEL_FEATURES):
        encoder = encoders[original_col]

        # Reproduce astype(str) usado en el notebook de transformación.
        values = df[original_col].astype(str)

        # LabelEncoder.classes_ conserva el orden de las categorías aprendidas.
        mapping = {
            str(category): int(code)
            for code, category in enumerate(encoder.classes_)
        }

        mapped = values.map(mapping)
        unknown = mapped.isna()

        unknown_summary[original_col] = int(unknown.sum())
        valid_mask &= ~unknown
        encoded[numeric_col] = mapped

    return encoded, valid_mask, unknown_summary


def decode_predictions(predicted_codes, artifacts: dict):
    """
    Conversión en dos etapas:
    1) código del modelo -> etiqueta numérica original mediante le.inverse_transform
    2) código numérico original -> texto mediante el LabelEncoder del notebook 3
    """
    model_target_encoder = artifacts["model_target_encoder"]
    original_target_encoder = artifacts["original_target_encoder"]

    predicted_codes = np.asarray(predicted_codes).astype(int)

    original_numeric_codes = model_target_encoder.inverse_transform(
        predicted_codes
    )

    original_numeric_codes = np.asarray(original_numeric_codes).astype(int)

    category_names = original_target_encoder.inverse_transform(
        original_numeric_codes
    )

    return category_names


def classify(df_original: pd.DataFrame, model, artifacts: dict):
    encoded, valid_mask, unknown_summary = encode_predictors(
        df_original,
        artifacts,
    )

    # Copia exacta de las columnas originales; no se reordena ni elimina filas.
    result = df_original.copy()

    # Evitar sobrescribir una columna que ya venga en el archivo de SAP.
    if TARGET_NAME in result.columns:
        backup_col = "Categoría de costo original (SAP)"
        if backup_col in result.columns:
            raise ValueError(
                "El archivo ya contiene 'Categoría de costo' y "
                "'Categoría de costo original (SAP)'. Revisa los encabezados."
            )
        result = result.rename(columns={TARGET_NAME: backup_col})

    predictions = pd.Series(pd.NA, index=result.index, dtype="object")

    valid_index = valid_mask[valid_mask].index
    if len(valid_index) > 0:
        X_new = encoded.loc[valid_index, MODEL_FEATURES].astype("int64")
        model_predictions = model.predict(X_new)
        predictions.loc[valid_index] = decode_predictions(
            model_predictions,
            artifacts,
        )

    result[TARGET_NAME] = predictions
    result["Estado de clasificación"] = np.where(
        valid_mask,
        "Clasificado",
        "Pendiente: valor no reconocido en una variable predictora",
    )

    stats = {
        "total": len(result),
        "clasificados": int(valid_mask.sum()),
        "pendientes": int((~valid_mask).sum()),
        "unknown_summary": unknown_summary,
    }

    return result, stats


def make_excel(result: pd.DataFrame) -> bytes:
    output = BytesIO()

    summary = (
        result[TARGET_NAME]
        .fillna("Pendiente de clasificación")
        .value_counts(dropna=False)
        .rename_axis(TARGET_NAME)
        .reset_index(name="Cantidad de registros")
    )
    summary["Porcentaje sobre total"] = (
        summary["Cantidad de registros"] / max(len(result), 1) * 100
    ).round(2)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Clasificacion", index=False)
        summary.to_excel(writer, sheet_name="Resumen", index=False)

    output.seek(0)
    return output.getvalue()


def show_dashboard(result: pd.DataFrame, stats: dict):
    st.subheader("Dashboard de clasificación")

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros originales", f"{stats['total']:,}")
    c2.metric("Registros clasificados", f"{stats['clasificados']:,}")
    c3.metric("Registros pendientes", f"{stats['pendientes']:,}")

    classified = result[result[TARGET_NAME].notna()].copy()

    if classified.empty:
        st.info("No hay registros clasificados para mostrar.")
        return

    classified["Actual"] = pd.to_numeric(
        classified["Actual"], errors="coerce"
    ).fillna(0)

    summary = (
        classified
        .groupby(TARGET_NAME)
        .agg(
            **{
                "Cantidad de registros": (TARGET_NAME, "size"),
                "Monto de la categoría": ("Actual", "sum")
            }
        )
        .reset_index()
    )

    monto_total = summary["Monto de la categoría"].sum()

    summary["Porcentaje sobre monto total"] = (
        summary["Monto de la categoría"] / monto_total * 100
        if monto_total != 0 else 0
    )

    summary = summary.sort_values(
        "Monto de la categoría", ascending=False
    )

    st.dataframe(
        summary.style.format({
            "Cantidad de registros": "{:,.0f}",
            "Monto de la categoría": "${:,.2f}",
            "Porcentaje sobre monto total": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)

    with left:
        fig_bar = px.bar(
            summary.sort_values("Cantidad", ascending=False),
            x=TARGET_NAME,
            y="Cantidad",
            title="Cantidad de registros por categoría",
            text="Cantidad",
        )
        fig_bar.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        fig_pie = px.pie(
            summary,
            names=TARGET_NAME,
            values="Cantidad",
            title="Distribución de categorías",
        )
        st.plotly_chart(fig_pie, use_container_width=True)


def main():
    check_access()

    st.title(APP_TITLE)
    st.write(
        "Aplicación para clasificar registros de costos 'Otros' del área de transporte propio, descargados de SAP, mediante el modelo Ensamble previamente entrenado con datos históricos del 2025."
    )

    accepted = show_confidentiality_notice()
    if not accepted:
        st.info("Acepta el aviso de confidencialidad para habilitar la carga.")
        st.stop()

    try:
        model, artifacts = load_model_and_artifacts()
    except Exception as exc:
        stop_error(f"No fue posible cargar el modelo o sus codificadores: {exc}")

    st.subheader("1. Cargar base de datos SAP")
    uploaded = st.file_uploader(
        "Selecciona un archivo Excel (.xlsx) o CSV",
        type=["xlsx", "csv"],
        help=f"Tamaño máximo permitido: {MAX_FILE_SIZE_MB} MB.",
    )

    if uploaded is None:
        st.caption(
            "El archivo debe contener estas cuatro columnas: "
            + ", ".join(ORIGINAL_FEATURES)
        )
        return

    try:
        df_original = read_file(uploaded)
        st.success(
            f"Archivo leído: {len(df_original):,} registros y "
            f"{len(df_original.columns):,} columnas."
        )

        with st.expander("Vista previa del archivo original"):
            st.dataframe(df_original.head(20), use_container_width=True)

        if st.button("Clasificar registros", type="primary"):
            with st.spinner("Transformando variables y clasificando registros..."):
                result, stats = classify(df_original, model, artifacts)

            st.session_state["cemex_result"] = result
            st.session_state["cemex_stats"] = stats
            st.session_state["cemex_filename"] = uploaded.name

    except Exception as exc:
        stop_error(f"No fue posible procesar el archivo: {exc}")

    if "cemex_result" not in st.session_state:
        return

    result = st.session_state["cemex_result"]
    stats = st.session_state["cemex_stats"]

    st.divider()
    show_dashboard(result, stats)

    if stats["pendientes"] > 0:
        st.warning(
            f"{stats['pendientes']:,} registros quedaron pendientes porque "
            "contienen valores que no aparecieron en el entrenamiento. "
            "No se inventaron códigos ni se asignaron categorías."
        )
        st.write("Valores no reconocidos por variable:")
        st.json(stats["unknown_summary"])

    st.subheader("Resultado de la clasificación")
    st.dataframe(result.head(100), use_container_width=True)

    excel_bytes = make_excel(result)
    original_stem = Path(
        st.session_state.get("cemex_filename", "SAP")
    ).stem
    download_name = f"Reclasificacion_Costos_CEMEX_{original_stem}.xlsx"

    st.download_button(
        "Descargar Excel clasificado",
        data=excel_bytes,
        file_name=download_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


if __name__ == "__main__":
    main()
