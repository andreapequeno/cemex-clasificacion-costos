"""
Reclasificaicon automatica de costos CEMEX - By UDEM

La aplicación NO entrena el modelo. Carga el ensamble previamente entrenado
y los codificadores exportados desde el notebook de Google Colab.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

APP_TITLE = "Reclasificaicon automatica de costos CEMEX - By UDEM"
MODEL_PATH = Path("modelo/modelo_cemex.joblib")
ENCODERS_PATH = Path("modelo/codificadores_cemex.joblib")
MAX_FILE_SIZE_MB = 50

# Orden exacto usado para entrenar el ensamble en el notebook _5modelos_CEMEX.
FEATURES = [
    "Denom.clase de coste",
    "Cl.coste",
    "Costo Estandar",
    "Descrip.clases coste",
]

# Nombres de las columnas numéricas que recibió el modelo.
ENCODED_FEATURES = [
    "Denom.clase de coste_num",
    "Cl.coste_num",
    "Costo Estandar_num",
    "Descrip.clases coste_num",
]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def stop_with_error(message: str) -> None:
    st.error(message)
    st.stop()


def require_authorized_user() -> None:
    """
    Autenticación OIDC de Streamlit. Se cierra el acceso si no está configurada.
    Para pruebas locales exclusivamente puede activarse CEMEX_LOCAL_DEV=1.
    No activar esa variable en un despliegue público.
    """
    local_dev = os.getenv("CEMEX_LOCAL_DEV", "").strip() == "1"
    if local_dev:
        st.sidebar.warning(
            "Modo de desarrollo local activo. No utilices este modo en producción "
            "ni cargues información empresarial real."
        )
        return

    try:
        auth_config = st.secrets.get("auth", {})
        allowed_emails = [
            str(email).strip().lower()
            for email in auth_config.get("allowed_emails", [])
            if str(email).strip()
        ]
    except Exception:
        allowed_emails = []

    if not allowed_emails:
        stop_with_error(
            "La aplicación no tiene una lista privada de correos autorizados. "
            "El acceso está bloqueado por seguridad. El administrador debe "
            "configurar Streamlit Secrets antes de habilitar el uso."
        )

    try:
        logged_in = bool(st.user.is_logged_in)
    except Exception:
        logged_in = False

    if not logged_in:
        st.warning("Debes iniciar sesión con una cuenta previamente autorizada.")
        st.button(
            "Iniciar sesión",
            on_click=st.login,
            args=("google",),
            type="primary",
        )
        st.stop()

    email = str(getattr(st.user, "email", "") or "").strip().lower()
    if not email or email not in allowed_emails:
        st.error(
            "Tu cuenta no está autorizada para utilizar esta aplicación. "
            "Si consideras que se trata de un error, contacta al administrador."
        )
        st.button("Cerrar sesión", on_click=st.logout)
        st.stop()

    with st.sidebar:
        st.caption(f"Sesión iniciada: {email}")
        st.button("Cerrar sesión", on_click=st.logout)


def show_confidentiality_notice() -> bool:
    st.subheader("Aviso de confidencialidad y resguardo de información")
    st.warning(
        """
**La información procesada mediante esta aplicación es confidencial y de uso
exclusivamente empresarial.**

- Solo deben cargarse archivos para los que el usuario tenga autorización.
- Los datos se utilizarán únicamente para la clasificación y análisis de costos.
- No se permite compartir, descargar para fines ajenos, distribuir ni divulgar
  información a personas o terceros no autorizados.
- El usuario es responsable de verificar que cuenta con autorización para
  utilizar la información.
- No deben cargarse contraseñas, datos personales innecesarios ni archivos
  ajenos al propósito del proyecto.
- La información no debe publicarse en repositorios, sitios web o servicios
  externos sin autorización expresa de CEMEX.
- La aplicación procesa los archivos en memoria y no implementa almacenamiento
  permanente de los archivos cargados. Esto no equivale a una garantía de
  eliminación de registros técnicos, copias de infraestructura o datos retenidos
  por el proveedor de alojamiento.
"""
    )
    accepted = st.checkbox(
        "He leído y acepto el aviso. Confirmo que estoy autorizado(a) para "
        "utilizar la información que voy a cargar.",
        value=False,
    )
    return accepted


@st.cache_resource(show_spinner="Cargando el modelo previamente entrenado...")
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {MODEL_PATH}. Coloca ahí el archivo exportado "
            "desde Google Colab."
        )
    if not ENCODERS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {ENCODERS_PATH}. Debes generar los codificadores "
            "desde el notebook original y colocarlos en la carpeta modelo/."
        )

    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ENCODERS_PATH)

    required_keys = {"feature_encoders", "target_encoder", "feature_order"}
    if not required_keys.issubset(artifacts):
        raise ValueError(
            "El archivo de codificadores no contiene la estructura esperada. "
            "Vuelve a generarlo con el bloque incluido en las instrucciones."
        )

    if list(artifacts["feature_order"]) != ENCODED_FEATURES:
        raise ValueError(
            "El orden de variables guardado no coincide con el orden esperado "
            "por la aplicación. No se ejecutarán predicciones."
        )

    for col in FEATURES:
        if col not in artifacts["feature_encoders"]:
            raise ValueError(f"Falta el codificador de la variable: {col}")

    return model, artifacts


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"El archivo supera el límite de {MAX_FILE_SIZE_MB} MB."
        )

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(BytesIO(raw), engine="openpyxl")
    elif suffix == ".csv":
        # Se prueban codificaciones habituales de exportaciones de SAP.
        last_error = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(BytesIO(raw), encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(f"No se pudo leer el CSV: {last_error}")
    else:
        raise ValueError("Formato no permitido. Carga un archivo .xlsx o .csv.")

    if df.empty:
        raise ValueError("El archivo no contiene registros.")

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Hay nombres de columnas duplicados: {duplicated}")

    return df


def encode_new_data(
    original_df: pd.DataFrame,
    artifacts: dict,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Aplica los LabelEncoder originales sin volver a ajustarlos.
    Los valores no vistos durante el entrenamiento quedan pendientes.
    """
    missing = [col for col in FEATURES if col not in original_df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas necesarias para el modelo: "
            + ", ".join(missing)
        )

    encoders = artifacts["feature_encoders"]
    encoded = pd.DataFrame(index=original_df.index)
    valid_mask = pd.Series(True, index=original_df.index)
    unknown_counts = {}

    for source_col, model_col in zip(FEATURES, ENCODED_FEATURES):
        encoder = encoders[source_col]
        values = original_df[source_col].astype(str)
        mapping = {str(label): int(code) for code, label in enumerate(encoder.classes_)}
        mapped = values.map(mapping)

        unknown_mask = mapped.isna()
        unknown_counts[source_col] = int(unknown_mask.sum())
        valid_mask &= ~unknown_mask
        encoded[model_col] = mapped

    # Solo se predicen filas para las que existen equivalencias aprendidas.
    return encoded, valid_mask, unknown_counts


def classify_dataframe(
    original_df: pd.DataFrame,
    model,
    artifacts: dict,
) -> tuple[pd.DataFrame, dict]:
    encoded, valid_mask, unknown_counts = encode_new_data(original_df, artifacts)

    result = original_df.copy()
    target_col = "Categoría de costo"

    # Si SAP ya contiene una columna con ese nombre, se conserva bajo otro nombre.
    if target_col in result.columns:
        backup_name = "Categoría de costo original"
        if backup_name in result.columns:
            backup_name = "Categoría de costo original (SAP)"
        result = result.rename(columns={target_col: backup_name})

    predictions_text = pd.Series(pd.NA, index=result.index, dtype="object")
    valid_positions = valid_mask[valid_mask].index

    if len(valid_positions) > 0:
        X_new = encoded.loc[valid_positions, ENCODED_FEATURES].astype("int64")
        predicted_codes = model.predict(X_new)
        target_encoder = artifacts["target_encoder"]
        predicted_labels = target_encoder.inverse_transform(
            pd.Series(predicted_codes).astype(int).to_numpy()
        )
        predictions_text.loc[valid_positions] = predicted_labels

    result[target_col] = predictions_text
    result["Estado de clasificación"] = "Clasificado"
    result.loc[~valid_mask, "Estado de clasificación"] = (
        "Pendiente: categoría de entrada no vista en entrenamiento"
    )

    stats = {
        "total": len(result),
        "clasificados": int(valid_mask.sum()),
        "pendientes": int((~valid_mask).sum()),
        "unknown_counts": unknown_counts,
    }
    return result, stats


def build_excel(result: pd.DataFrame) -> bytes:
    output = BytesIO()
    summary = (
        result["Categoría de costo"]
        .fillna("Pendiente de clasificación")
        .value_counts(dropna=False)
        .rename_axis("Categoría de costo")
        .reset_index(name="Cantidad de registros")
    )
    summary["Porcentaje"] = (
        summary["Cantidad de registros"] / max(len(result), 1) * 100
    ).round(2)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Clasificacion", index=False)
        summary.to_excel(writer, sheet_name="Resumen", index=False)

    output.seek(0)
    return output.getvalue()


def show_dashboard(result: pd.DataFrame, stats: dict) -> None:
    st.subheader("Dashboard de clasificación")

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros originales", f"{stats['total']:,}")
    c2.metric("Registros clasificados", f"{stats['clasificados']:,}")
    c3.metric("Registros pendientes", f"{stats['pendientes']:,}")

    classified = result["Categoría de costo"].dropna()
    if classified.empty:
        st.info("No hay registros clasificados para mostrar en las gráficas.")
        return

    summary = (
        classified.value_counts()
        .rename_axis("Categoría de costo")
        .reset_index(name="Cantidad")
    )
    summary["Porcentaje"] = summary["Cantidad"] / stats["total"] * 100

    st.dataframe(
        summary.style.format({"Porcentaje": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        fig_bar = px.bar(
            summary.sort_values("Cantidad", ascending=False),
            x="Categoría de costo",
            y="Cantidad",
            title="Cantidad de registros por categoría",
            text="Cantidad",
        )
        fig_bar.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        fig_pie = px.pie(
            summary,
            names="Categoría de costo",
            values="Cantidad",
            title="Distribución porcentual de categorías",
        )
        st.plotly_chart(fig_pie, use_container_width=True)


def main():
    require_authorized_user()

    st.title(APP_TITLE)
    st.write(
        "Herramienta para clasificar registros de costos operativos de SAP "
        "mediante el modelo de Machine Learning Ensamble Soft Voting "
        "previamente entrenado."
    )

    st.info(
        "La aplicación utiliza un modelo ya entrenado. No realiza entrenamiento "
        "ni balanceo SMOTE al cargar archivos nuevos."
    )

    accepted = show_confidentiality_notice()
    if not accepted:
        st.info("Acepta el aviso de confidencialidad para habilitar la carga.")
        st.stop()

    try:
        model, artifacts = load_artifacts()
    except Exception as exc:
        stop_with_error(f"No fue posible cargar los artefactos del modelo: {exc}")

    st.subheader("1. Carga de base de datos SAP")
    uploaded = st.file_uploader(
        "Selecciona un archivo Excel (.xlsx) o CSV",
        type=["xlsx", "csv"],
        help=f"Tamaño máximo: {MAX_FILE_SIZE_MB} MB.",
    )

    if uploaded is None:
        st.caption(
            "La aplicación necesita las columnas originales correspondientes "
            "a las cuatro variables explicativas utilizadas durante el entrenamiento."
        )
        return

    try:
        original_df = read_uploaded_file(uploaded)
        st.success(
            f"Archivo leído correctamente: {len(original_df):,} registros "
            f"y {len(original_df.columns):,} columnas."
        )
        with st.expander("Vista previa del archivo original"):
            st.dataframe(original_df.head(20), use_container_width=True)

        if st.button("Clasificar registros", type="primary"):
            with st.spinner("Aplicando codificadores y generando predicciones..."):
                result, stats = classify_dataframe(original_df, model, artifacts)

            st.session_state["cemex_result"] = result
            st.session_state["cemex_stats"] = stats
            st.session_state["cemex_filename"] = uploaded.name

    except Exception as exc:
        stop_with_error(f"No fue posible procesar el archivo: {exc}")

    if "cemex_result" in st.session_state:
        result = st.session_state["cemex_result"]
        stats = st.session_state["cemex_stats"]

        st.divider()
        show_dashboard(result, stats)

        if stats["pendientes"] > 0:
            st.warning(
                f"{stats['pendientes']:,} registros quedaron pendientes porque "
                "una o más variables contienen valores que no aparecieron en "
                "el conjunto de entrenamiento. No se asignó una categoría "
                "inventada. Revisa los valores y actualiza los codificadores "
                "solo mediante un proceso controlado."
            )
            st.write("Cantidad de valores no reconocidos por variable:")
            st.json(stats["unknown_counts"])

        st.subheader("Resultado de la clasificación")
        st.dataframe(result.head(100), use_container_width=True)

        output_bytes = build_excel(result)
        original_name = Path(st.session_state.get("cemex_filename", "SAP")).stem
        output_name = f"Reclasificacion_Costos_CEMEX_{original_name}.xlsx"

        st.download_button(
            label="Descargar Excel clasificado",
            data=output_bytes,
            file_name=output_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        st.caption(
            "El Excel incluye la hoja 'Clasificacion' con las columnas originales "
            "y el resultado, y la hoja 'Resumen' con el conteo por categoría."
        )


if __name__ == "__main__":
    main()
