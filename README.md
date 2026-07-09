# SIAP — Sistema Inteligente de Adaptación Pedagógica

SIAP registra a los estudiantes y su perfil de diversidad (necesidades
auditivas, visuales, ritmo de aprendizaje lento o rápido, entre otras) y
sugiere al docente adaptaciones curriculares concretas, apoyándose en un
modelo de clasificación entrenado con `scikit-learn` y expuesto mediante
una aplicación en `Streamlit`.

## Contenido del repositorio

```
├── landing.html          # Landing page del proyecto (ciclo de vida CRISP-DM)
├── app.py                # Aplicación Streamlit con el modelo de SIAP
├── SIAP_modelo.ipynb      # Notebook con el análisis y entrenamiento (CRISP-DM)
├── requirements.txt       # Dependencias del proyecto
├── build_notebook.py      # Script que genera SIAP_modelo.ipynb (uso interno)
└── README.md              # Este archivo
```

> **Estado de los datos:** el notebook y la app funcionan con datos
> **sintéticos** generados automáticamente (`generar_datos_sinteticos()`),
> para que todo el proyecto sea reproducible sin depender de un dataset
> externo. Al integrar datos reales de estudiantes, reemplaza esa función
> por la carga de tu propio dataset (anonimizado) y vuelve a entrenar.

## Requisitos

- Python 3.10 o superior
- pip

Dependencias principales (ver `requirements.txt`):

- `streamlit` — interfaz para el docente
- `scikit-learn` — modelo de clasificación
- `pandas`, `numpy` — manejo de datos
- `matplotlib` — visualizaciones en el notebook
- `joblib` — guardar/cargar el modelo entrenado

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/siap.git
cd siap

# 2. Crear y activar un entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

### 1. Landing page

Abre `landing.html` directamente en el navegador, o publícala con
GitHub Pages (Settings → Pages → Deploy from branch → `main` / `/root`).

### 2. Notebook del modelo

```bash
jupyter notebook SIAP_modelo.ipynb
```

Sigue las seis secciones del notebook (comprensión del negocio,
comprensión de los datos, preparación, modelado, evaluación y
despliegue). Al final se guarda el modelo entrenado en `modelo_siap.pkl`.

### 3. Aplicación Streamlit

```bash
streamlit run app.py
```

Esto abre la app en `http://localhost:8501`. Desde ahí el docente puede:

1. Registrar el perfil del estudiante (necesidades auditiva/visual, ritmo
   de aprendizaje, desempeño promedio).
2. Presionar **"Generar sugerencia"**.
3. Ver la adaptación curricular sugerida y la confianza del modelo.

Para usar el modelo entrenado en el notebook en lugar del modelo de
demostración, reemplaza en `app.py` la función `entrenar_modelo()` por:

```python
import joblib
modelo = joblib.load("modelo_siap.pkl")
```

## Ciclo de vida del proyecto (CRISP-DM)

| Fase | Descripción |
|---|---|
| 1. Comprensión del negocio | Docentes sin herramientas para adaptar el currículo a la diversidad del aula. |
| 2. Comprensión de los datos | Variables de diversidad, ritmo de aprendizaje y desempeño del estudiante. |
| 3. Preparación de datos | Limpieza, codificación y partición en entrenamiento/prueba. |
| 4. Modelado | `RandomForestClassifier` que sugiere el tipo de adaptación curricular. |
| 5. Evaluación | Accuracy, reporte de clasificación y matriz de confusión. |
| 6. Despliegue | App en Streamlit + repositorio publicado en GitHub. |

## Publicar el proyecto en GitHub

Si aún no tienes el repositorio creado:

```bash
git init
git add .
git commit -m "Primera versión de SIAP: landing page, notebook y app Streamlit"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/siap.git
git push -u origin main
```

## Imágenes de la landing page

Las ilustraciones de la landing page (`landing.html`) son gráficos
vectoriales (SVG) propios, con estética de "visor de realidad aumentada",
para evitar depender de imágenes externas que puedan romperse o tener
restricciones de licencia. Si prefieres imágenes generadas por IA,
consulta la sección de prompts sugeridos que Claude te compartió en el
chat para generarlas con tu herramienta de preferencia (Midjourney, DALL·E,
Firefly, etc.) y reemplazar el bloque `<svg>` del hero por una etiqueta
`<img src="tu-imagen.png" alt="...">`.

## Autoría

Proyecto académico — Sistema Inteligente de Adaptación Pedagógica (SIAP), 2026.
