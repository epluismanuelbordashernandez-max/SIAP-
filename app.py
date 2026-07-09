"""
app.py — SIAP (Sistema Inteligente de Adaptación Pedagógica)
===============================================================

Aplicación en Streamlit que permite a un docente registrar el perfil de
un estudiante (necesidades auditivas, visuales y ritmo de aprendizaje) y
recibir una sugerencia de adaptación curricular generada por un modelo
de clasificación entrenado con scikit-learn.

NOTA IMPORTANTE PARA EL ESTUDIANTE / DOCENTE DEL CURSO:
--------------------------------------------------------
Este archivo entrena el modelo con datos SINTÉTICOS (function
`generar_datos_sinteticos`) para que la demo funcione de inmediato, sin
depender de un dataset externo. Cuando tengas listo tu propio modelo
entrenado en el notebook (`SIAP_modelo.ipynb`), reemplaza la función
`entrenar_modelo()` por la carga de tu modelo real, por ejemplo:

    import joblib
    modelo = joblib.load("modelo_siap.pkl")

Así el resto de la interfaz (formulario, predicción, explicación) no
tiene que cambiar.

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SIAP — Adaptación Curricular Inteligente",
    page_icon="🎓",
    layout="centered",
)

RITMOS = ["Lento", "Normal", "Rápido"]
ADAPTACIONES = [
    "Apoyo auditivo (subtítulos / lengua de señas)",
    "Material visual ampliado / alto contraste",
    "Refuerzo con ritmo extendido",
    "Retos adicionales con ritmo acelerado",
    "Sin adaptación específica requerida",
]


# ---------------------------------------------------------------------------
# 1. Datos sintéticos de ejemplo
# ---------------------------------------------------------------------------
def generar_datos_sinteticos(n: int = 600, semilla: int = 42) -> pd.DataFrame:
    """Genera un dataset sintético de perfiles estudiantiles.

    Cada fila representa un estudiante con:
    - necesidad_auditiva (0/1)
    - necesidad_visual (0/1)
    - ritmo_aprendizaje (0=Lento, 1=Normal, 2=Rápido)
    - desempeno_promedio (0-100)
    - adaptacion_sugerida (etiqueta objetivo, generada con una regla base
      + ruido, solo para fines demostrativos)

    Parameters
    ----------
    n : int
        Número de estudiantes a simular.
    semilla : int
        Semilla para reproducibilidad.

    Returns
    -------
    pd.DataFrame
        Dataset sintético listo para entrenar el modelo de ejemplo.
    """
    rng = np.random.default_rng(semilla)

    necesidad_auditiva = rng.integers(0, 2, size=n)
    necesidad_visual = rng.integers(0, 2, size=n)
    ritmo_aprendizaje = rng.integers(0, 3, size=n)  # 0 lento, 1 normal, 2 rápido
    desempeno_promedio = rng.normal(70, 15, size=n).clip(0, 100)

    etiquetas = []
    for aud, vis, ritmo in zip(necesidad_auditiva, necesidad_visual, ritmo_aprendizaje):
        if aud == 1:
            etiquetas.append(0)  # Apoyo auditivo
        elif vis == 1:
            etiquetas.append(1)  # Apoyo visual
        elif ritmo == 0:
            etiquetas.append(2)  # Refuerzo, ritmo lento
        elif ritmo == 2:
            etiquetas.append(3)  # Retos adicionales, ritmo rápido
        else:
            etiquetas.append(4)  # Sin adaptación específica

    df = pd.DataFrame(
        {
            "necesidad_auditiva": necesidad_auditiva,
            "necesidad_visual": necesidad_visual,
            "ritmo_aprendizaje": ritmo_aprendizaje,
            "desempeno_promedio": desempeno_promedio,
            "adaptacion_sugerida": etiquetas,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. Entrenamiento del modelo (cacheado para no reentrenar en cada click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando modelo de ejemplo de SIAP...")
def entrenar_modelo():
    """Entrena y devuelve el modelo de clasificación junto con su accuracy.

    En este ejemplo se usa un RandomForestClassifier de scikit-learn sobre
    datos sintéticos. Sustituir esta función por la carga del modelo real
    entrenado en el notebook del proyecto cuando esté disponible.

    Returns
    -------
    tuple[RandomForestClassifier, float]
        El modelo entrenado y el accuracy obtenido en el set de prueba.
    """
    df = generar_datos_sinteticos()
    X = df[["necesidad_auditiva", "necesidad_visual", "ritmo_aprendizaje", "desempeno_promedio"]]
    y = df["adaptacion_sugerida"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, modelo.predict(X_test))
    return modelo, accuracy


# ---------------------------------------------------------------------------
# 3. Interfaz de usuario
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🎓 SIAP — Sistema Inteligente de Adaptación Pedagógica")
    st.write(
        "Registra el perfil de un estudiante y SIAP sugerirá la adaptación "
        "curricular más adecuada para el docente."
    )

    modelo, accuracy = entrenar_modelo()

    with st.expander("ℹ️ Sobre este modelo de demostración"):
        st.markdown(
            f"""
            - Este modelo se entrena **en cada despliegue** con datos
              **sintéticos**, únicamente para fines de demostración.
            - Accuracy en el set de prueba sintético: **{accuracy:.0%}**
            - Para producción, reemplázalo por el modelo entrenado con
              datos reales en `SIAP_modelo.ipynb`.
            """
        )

    st.subheader("Perfil del estudiante")
    col1, col2 = st.columns(2)

    with col1:
        nombre = st.text_input("Nombre del estudiante", placeholder="Ej. Ana Pérez")
        necesidad_auditiva = st.checkbox("Presenta necesidad auditiva")
        necesidad_visual = st.checkbox("Presenta necesidad visual")

    with col2:
        ritmo_label = st.select_slider("Ritmo de aprendizaje", options=RITMOS, value="Normal")
        desempeno = st.slider("Desempeño académico promedio (0-100)", 0, 100, 70)

    ritmo_valor = RITMOS.index(ritmo_label)

    if st.button("Generar sugerencia", type="primary"):
        entrada = pd.DataFrame(
            [[int(necesidad_auditiva), int(necesidad_visual), ritmo_valor, desempeno]],
            columns=["necesidad_auditiva", "necesidad_visual", "ritmo_aprendizaje", "desempeno_promedio"],
        )

        prediccion = modelo.predict(entrada)[0]
        probabilidades = modelo.predict_proba(entrada)[0]

        sugerencia = ADAPTACIONES[prediccion]
        confianza = probabilidades[prediccion]

        st.success(f"**Adaptación sugerida:** {sugerencia}")
        st.caption(f"Confianza del modelo: {confianza:.0%}")

        if nombre:
            st.write(f"Sugerencia generada para **{nombre}**.")

        st.subheader("Distribución de probabilidad por adaptación")
        tabla_prob = pd.DataFrame(
            {"Adaptación": ADAPTACIONES, "Probabilidad": probabilidades}
        ).sort_values("Probabilidad", ascending=False)
        st.bar_chart(tabla_prob.set_index("Adaptación"))

    st.divider()
    st.caption(
        "SIAP · Proyecto académico. Modelo de demostración entrenado con "
        "datos sintéticos, sin información real de estudiantes."
    )


if __name__ == "__main__":
    main()
