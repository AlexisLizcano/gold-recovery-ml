# ⛏️ Gold Recovery ML — Predicción de Recuperación de Oro

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Proyecto de Machine Learning para predecir el **coeficiente de recuperación de oro** en dos etapas del proceso de purificación mineral (rougher y final), usando datos reales de una planta de flotación.

Desarrollado como parte del **Sprint 13 de TripleTen Data Science Bootcamp**.

---

## 📋 Tabla de Contenidos

- [Descripción del Problema](#-descripción-del-problema)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados](#-resultados)
- [Instalación y Uso](#-instalación-y-uso)
- [App Interactiva](#-app-interactiva-streamlit)
- [Tecnologías](#-tecnologías)

---

## 🎯 Descripción del Problema

En el proceso de extracción de oro, el mineral pasa por varias etapas de flotación y limpieza. El objetivo es construir un modelo que prediga la **tasa de recuperación** en:

- **Etapa Rougher** — Primera flotación del mineral
- **Etapa Final** — Concentrado limpio listo para exportar

La métrica de evaluación es el **sMAPE Final** (ponderado):

```
sMAPE final = 25% × sMAPE(rougher) + 75% × sMAPE(final)
```

---

## 📁 Estructura del Proyecto

```
gold-recovery-ml/
│
├── 📓 notebooks/
│   └── sprint13_project.ipynb       # Análisis completo paso a paso
│
├── 📊 Datasets/
│   ├── gold_recovery_train.csv      # Datos de entrenamiento (16,860 registros)
│   ├── gold_recovery_test.csv       # Datos de prueba (5,856 registros)
│   └── gold_recovery_full.csv       # Dataset completo (referencia)
│
├── 🖥️ app.py                        # App interactiva en Streamlit
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Este archivo
```

---

## 📈 Resultados

| Modelo | sMAPE Final | vs Baseline |
|---|---|---|
| Baseline (Media) | 9.787 | — |
| Regresión Lineal | ~9.5 | -3% |
| **Random Forest** | **9.199** | **-6.0% ✅** |
| Gradient Boosting | ~9.3 | -5% |

**Conclusiones clave:**
- La concentración de Au aumenta de **7.7 g/t → 42.7 g/t** a lo largo del proceso ✅
- **Random Forest** es el modelo ganador con sMAPE de 9.199
- Se detectaron diferencias significativas entre train y test (KS test p < 0.05)
- Se evitó data leakage alineando features correctamente con el conjunto de prueba

---

## 🚀 Instalación y Uso

### 1. Clona el repositorio

```bash
git clone https://github.com/TU_USUARIO/gold-recovery-ml.git
cd gold-recovery-ml
```

### 2. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 3. Explora el notebook

Abre `notebooks/sprint13_project.ipynb` en Jupyter o VS Code.

### 4. Corre la app interactiva

```bash
streamlit run app.py
```

Se abrirá en tu navegador en `http://localhost:8501`

---

## 🖥️ App Interactiva Streamlit

La app incluye 5 secciones:

| Sección | Descripción |
|---|---|
| 🏠 Inicio | KPIs del proyecto, diagrama Sankey del proceso, recuperación en el tiempo |
| 🔬 Exploración de Datos | Distribuciones, análisis de nulos, comparación Train vs Test |
| ⚗️ Proceso de Purificación | Concentración de Au/Ag/Pb por etapa, detección de anomalías |
| 🤖 Modelos ML | Comparación de 4 modelos con CV 5-fold |
| 📊 Resultados Finales | Predicciones vs reales, distribución del error, conclusiones |

---

## 🛠️ Tecnologías

- **Python 3.10+**
- **pandas** — Manipulación de datos
- **scikit-learn** — Modelos ML (Random Forest, Gradient Boosting, Linear Regression)
- **Streamlit** — App interactiva
- **Plotly** — Visualizaciones interactivas
- **scipy** — Test estadísticos (KS test)

---

## 👩‍💻 Autora

**Lizca** — TripleTen Data Science Bootcamp, Sprint 13

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/TU_PERFIL)
