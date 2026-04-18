"""
Gold Recovery ML Dashboard —
Alexis Lizcano | Data Science Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Recovery ML",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo principal */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #0f1117 100%);
        border-right: 1px solid #2d2f3f;
    }

    /* Tarjetas de métricas */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2030 0%, #252840 100%);
        border: 1px solid #3a3d5c;
        border-radius: 12px;
        padding: 16px !important;
    }

    /* Headers */
    h1 { color: #f5c842 !important; }
    h2, h3 { color: #e0e0e0 !important; }

    /* Separadores */
    hr { border-color: #2d2f3f; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d2e;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9a9cb0;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f5c842 !important;
        color: #0f1117 !important;
        font-weight: 600;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-gold { background: #f5c842; color: #0f1117; }
    .badge-green { background: #22c55e; color: #fff; }
    .badge-blue { background: #3b82f6; color: #fff; }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #1e2030 0%, #1a2540 100%);
        border-left: 4px solid #f5c842;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #d0d0e0;
        font-size: 14px;
        line-height: 1.7;
    }

    /* Stagger animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeIn 0.5s ease forwards; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
GOLD_PALETTE = ["#f5c842", "#e8973a", "#c0392b", "#3b82f6", "#22c55e", "#a78bfa"]
PLOTLY_TEMPLATE = "plotly_dark"


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    smape_values = np.where(denominator != 0, diff / denominator, 0)
    return np.mean(smape_values) * 100


def final_smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r = smape(y_true[:, 0], y_pred[:, 0])
    f = smape(y_true[:, 1], y_pred[:, 1])
    return 0.25 * r + 0.75 * f


smape_scorer = make_scorer(final_smape, greater_is_better=False)


# ─────────────────────────────────────────────
# CARGA DE DATOS (con caché)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos...")
def load_data():
    import os

    # Buscar los CSVs en varias ubicaciones
    search_dirs = [
        "Datasets",
        "datasets",
        ".",
        os.path.join(os.path.dirname(__file__), "Datasets"),
        os.path.join(os.path.dirname(__file__), "datasets"),
    ]

    def find_file(name):
        for d in search_dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"No se encontró '{name}'. Asegúrate de que la carpeta Datasets/ "
            f"esté en el mismo directorio que app.py."
        )

    train = pd.read_csv(find_file("gold_recovery_train.csv"))
    test  = pd.read_csv(find_file("gold_recovery_test.csv"))
    full  = pd.read_csv(find_file("gold_recovery_full.csv"))

    # Fechas como índice
    for df in [train, test, full]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

    # Eliminar duplicados
    train = train.drop_duplicates()
    test  = test.drop_duplicates()
    full  = full.drop_duplicates()

    # Concentraciones totales y limpieza
    stages = {
        "alimentación": ["rougher.input.feed_au","rougher.input.feed_ag","rougher.input.feed_pb"],
        "rougher":      ["rougher.output.concentrate_au","rougher.output.concentrate_ag","rougher.output.concentrate_pb"],
        "final":        ["final.output.concentrate_au","final.output.concentrate_ag","final.output.concentrate_pb"],
    }

    for df in [train, full]:
        for name, cols in stages.items():
            df[f"total_{name}"] = df[cols].sum(axis=1)

    for df_name, df in [("train", train), ("full", full)]:
        mask = (df["total_alimentación"] > 0) & (df["total_rougher"] > 0) & (df["total_final"] > 0)
        if df_name == "train":
            train = df[mask].copy()
        else:
            full = df[mask].copy()

    return train, test, full


# ─────────────────────────────────────────────
# ENTRENAMIENTO (con caché)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Entrenando modelos... (puede tardar ~1 min)")
def train_models(train_df):
    import os
    from sklearn.impute import SimpleImputer

    target_cols = ["rougher.output.recovery", "final.output.recovery"]

    # Encontrar columnas que NO están en test
    for p in ["Datasets/gold_recovery_test.csv", "datasets/gold_recovery_test.csv", "gold_recovery_test.csv"]:
        if os.path.exists(p):
            test_cols = set(pd.read_csv(p).columns)
            break

    missing = set(train_df.columns) - test_cols
    features = train_df.drop(columns=list(missing), errors="ignore")
    features = features.drop(columns=[c for c in target_cols if c in features.columns], errors="ignore")

    target = train_df[target_cols].copy()
    target = target.dropna()
    features = features.loc[target.index]

    # Imputar NaN con la mediana
    imputer = SimpleImputer(strategy="median")
    features_imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns,
        index=features.index
    )

    models = {
        "Baseline (Media)": DummyRegressor(strategy="mean"),
        "Regresión Lineal": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MultiOutputRegressor(LinearRegression()))
        ]),
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        ),
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, features_imputed, target, scoring=smape_scorer, cv=5)
        results[name] = {
            "smape_mean": -scores.mean(),
            "smape_std":  scores.std(),
            "scores":     -scores,
        }

    best_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )
    best_model.fit(features_imputed, target)

    return results, best_model, features_imputed, target


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <span style='font-size:48px;'>⛏️</span>
        <h2 style='color:#f5c842; margin:8px 0 2px; font-size:18px;'>Gold Recovery ML</h2>
        <p style='color:#6b7280; font-size:12px; margin:0;'>Sprint 13 · TripleTen</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#9ca3af; font-size:12px; font-weight:600; letter-spacing:1px;'>NAVEGACIÓN</p>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠 Inicio", "🔬 Exploración de Datos", "⚗️ Proceso de Purificación", "🤖 Modelos ML", "📊 Resultados Finales"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#6b7280; font-size:11px; line-height:1.8;'>
        <b style='color:#9ca3af;'>Métrica</b><br>
        sMAPE Final = 25% × sMAPE<sub>rougher</sub> + 75% × sMAPE<sub>final</sub>
        <br><br>
        <b style='color:#9ca3af;'>Dataset</b><br>
        Train: 15,096 registros<br>
        Test: 5,856 registros<br>
        Features: 87 columnas
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────
try:
    train_df, test_df, full_df = load_data()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()


# ═════════════════════════════════════════════
# PÁGINA 1: INICIO
# ═════════════════════════════════════════════
if page == "🏠 Inicio":
    st.markdown("""
    <div class='fade-in'>
    <h1 style='font-size:38px; margin-bottom:6px;'>⛏️ Recuperación de Oro</h1>
    <p style='color:#9ca3af; font-size:16px; margin-bottom:30px;'>
        Predicción del coeficiente de recuperación en el proceso de flotación y purificación de mineral
    </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Registros de Entrenamiento", f"{train_df.shape[0]:,}")
    with col2:
        st.metric("Features Disponibles", f"{train_df.shape[1] - 2}")
    with col3:
        st.metric("Rec. Rougher Promedio", f"{train_df['rougher.output.recovery'].mean():.1f}%")
    with col4:
        st.metric("Rec. Final Promedio", f"{train_df['final.output.recovery'].mean():.1f}%")

    st.markdown("---")

    # Descripción del proceso
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("### 🏭 ¿Qué es la flotación de oro?")
        st.markdown("""
        <div class='info-box'>
        El proceso de extracción de oro implica varias etapas de purificación:
        <br><br>
        <b style='color:#f5c842;'>1. Alimentación (Feed)</b> — El mineral crudo entra al sistema con baja concentración de oro (~7.7 g/t).<br><br>
        <b style='color:#f5c842;'>2. Etapa Rougher</b> — Primera flotación: el oro sube al concentrado mientras las colas se descartan. La recuperación promedio es <b>84.3%</b>.<br><br>
        <b style='color:#f5c842;'>3. Etapa Final (Cleaner)</b> — Limpieza adicional que eleva la concentración de Au hasta ~42.7 g/t. La recuperación final promedio es <b>67.7%</b>.<br><br>
        El objetivo del modelo es <b>predecir ambas tasas de recuperación</b> a partir de las condiciones del proceso.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        # Diagrama de flujo del proceso
        fig = go.Figure()
        fig.add_trace(go.Sankey(
            node=dict(
                pad=20, thickness=25,
                label=["Mineral Crudo", "Etapa Rougher", "Concentrado Rougher", "Etapa Final", "Concentrado Final (Au ↑)", "Colas Rougher"],
                color=["#6b7280", "#3b82f6", "#f5c842", "#a78bfa", "#f5c842", "#ef4444"],
                x=[0.0, 0.35, 0.55, 0.7, 1.0, 0.6],
                y=[0.5, 0.5, 0.3, 0.3, 0.3, 0.8],
            ),
            link=dict(
                source=[0, 1, 2, 1],
                target=[1, 2, 3, 5],
                value=[100, 84, 68, 16],
                color=["rgba(59,130,246,0.3)", "rgba(245,200,66,0.4)", "rgba(167,139,250,0.3)", "rgba(239,68,68,0.3)"]
            )
        ))
        fig.update_layout(
            title=dict(text="Flujo del Proceso de Purificación", font=dict(color="#e0e0e0")),
            paper_bgcolor="#1e2030", font_color="#e0e0e0",
            height=320, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Línea de tiempo de recuperación
    st.markdown("### 📅 Recuperación a lo largo del tiempo")
    time_df = train_df[["rougher.output.recovery", "final.output.recovery"]].copy()
    time_df = time_df.dropna().resample("W").mean()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=time_df.index, y=time_df["rougher.output.recovery"],
        name="Rougher", line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
    ))
    fig2.add_trace(go.Scatter(
        x=time_df.index, y=time_df["final.output.recovery"],
        name="Final", line=dict(color="#f5c842", width=2),
        fill="tozeroy", fillcolor="rgba(245,200,66,0.1)"
    ))
    fig2.update_layout(
        template=PLOTLY_TEMPLATE, height=300,
        xaxis_title="Fecha", yaxis_title="Recuperación (%)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════
# PÁGINA 2: EXPLORACIÓN DE DATOS
# ═════════════════════════════════════════════
elif page == "🔬 Exploración de Datos":
    st.markdown("## 🔬 Exploración de Datos")

    tab1, tab2, tab3 = st.tabs(["📊 Estadísticas", "🔍 Valores Nulos", "📐 Train vs Test"])

    with tab1:
        st.markdown("### Distribución de las Variables Objetivo")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                train_df, x="rougher.output.recovery",
                nbins=50, title="Recuperación Rougher",
                color_discrete_sequence=["#3b82f6"],
                template=PLOTLY_TEMPLATE
            )
            fig.add_vline(x=train_df["rougher.output.recovery"].mean(),
                          line_dash="dash", line_color="#f5c842",
                          annotation_text=f"Media: {train_df['rougher.output.recovery'].mean():.1f}%")
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                train_df, x="final.output.recovery",
                nbins=50, title="Recuperación Final",
                color_discrete_sequence=["#f5c842"],
                template=PLOTLY_TEMPLATE
            )
            fig.add_vline(x=train_df["final.output.recovery"].mean(),
                          line_dash="dash", line_color="#3b82f6",
                          annotation_text=f"Media: {train_df['final.output.recovery'].mean():.1f}%")
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Scatter plot entre targets
        st.markdown("### Relación entre Recuperación Rougher y Final")
        sample = train_df[["rougher.output.recovery", "final.output.recovery"]].dropna().sample(
            min(3000, len(train_df)), random_state=42
        )
        fig = px.scatter(
            sample,
            x="rougher.output.recovery", y="final.output.recovery",
            opacity=0.4, template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#f5c842"],
            trendline="ols", trendline_color_override="#3b82f6",
            labels={"rougher.output.recovery": "Recuperación Rougher (%)",
                    "final.output.recovery": "Recuperación Final (%)"}
        )
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Análisis de Valores Nulos")
        null_pct = (train_df.isnull().mean() * 100).sort_values(ascending=False)
        null_pct = null_pct[null_pct > 0].head(30)

        if null_pct.empty:
            st.success("✅ No se detectaron valores nulos significativos en el dataset de entrenamiento.")
        else:
            fig = px.bar(
                x=null_pct.values, y=null_pct.index,
                orientation="h", template=PLOTLY_TEMPLATE,
                title="Top columnas con valores nulos (%)",
                color=null_pct.values,
                color_continuous_scale=["#22c55e", "#f5c842", "#ef4444"]
            )
            fig.update_layout(height=500, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nulos en Train", f"{train_df.isnull().sum().sum():,}")
        with col2:
            st.metric("Nulos en Test", f"{test_df.isnull().sum().sum():,}")
        with col3:
            st.metric("Duplicados eliminados", "0")

    with tab3:
        st.markdown("### Comparación de Distribuciones — Train vs Test")
        st.markdown("""
        <div class='info-box'>
        Se comparan las distribuciones del tamaño de partículas entre train y test.
        Si difieren significativamente (p-value < 0.05 en el test KS), el modelo puede tener dificultades para generalizar.
        </div>
        """, unsafe_allow_html=True)

        features_to_compare = [
            "rougher.input.feed_size",
            "primary_cleaner.input.feed_size",
            "rougher.input.feed_au",
        ]

        for feature in features_to_compare:
            if feature in train_df.columns and feature in test_df.columns:
                train_vals = train_df[feature].dropna()
                test_vals  = test_df[feature].dropna()
                ks_stat, p_value = ks_2samp(train_vals, test_vals)

                col1, col2 = st.columns([3, 1])
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=train_vals, name="Train",
                        opacity=0.6, histnorm="probability density",
                        marker_color="#3b82f6"
                    ))
                    fig.add_trace(go.Histogram(
                        x=test_vals, name="Test",
                        opacity=0.6, histnorm="probability density",
                        marker_color="#f5c842"
                    ))
                    fig.update_layout(
                        barmode="overlay", template=PLOTLY_TEMPLATE,
                        title=feature, height=250,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.metric("KS Statistic", f"{ks_stat:.4f}")
                    color = "🔴" if p_value < 0.05 else "🟢"
                    st.metric("p-value", f"{color} {p_value:.4f}")
                    if p_value < 0.05:
                        st.warning("Distribuciones significativamente diferentes")
                    else:
                        st.success("Distribuciones similares")


# ═════════════════════════════════════════════
# PÁGINA 3: PROCESO DE PURIFICACIÓN
# ═════════════════════════════════════════════
elif page == "⚗️ Proceso de Purificación":
    st.markdown("## ⚗️ Proceso de Purificación de Mineral")

    # Concentración de metales por etapa
    st.markdown("### Concentración de Metales por Etapa")
    st.markdown("""
    <div class='info-box'>
    Se espera que el <b style='color:#f5c842;'>oro (Au) aumente</b> progresivamente a lo largo del proceso,
    mientras que las impurezas (<b>Ag, Pb</b>) deberían reducirse en el concentrado final.
    </div>
    """, unsafe_allow_html=True)

    metal_tabs = st.tabs(["🥇 Oro (Au)", "🥈 Plata (Ag)", "🔩 Plomo (Pb)"])
    metals = {"🥇 Oro (Au)": "au", "🥈 Plata (Ag)": "ag", "🔩 Plomo (Pb)": "pb"}
    stages_map = {
        "Alimentación": "rougher.input.feed_{}",
        "Concentrado Rougher": "rougher.output.concentrate_{}",
        "Concentrado Final": "final.output.concentrate_{}",
    }
    colors = {"Alimentación": "#6b7280", "Concentrado Rougher": "#3b82f6", "Concentrado Final": "#f5c842"}

    def hex_to_rgba(hex_color, alpha=0.4):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for tab, (label, metal) in zip(metal_tabs, metals.items()):
        with tab:
            fig = go.Figure()
            means = []
            for stage_name, pattern in stages_map.items():
                col = pattern.format(metal)
                if col in train_df.columns:
                    vals = train_df[col].dropna()
                    fig.add_trace(go.Violin(
                        y=vals, name=stage_name,
                        fillcolor=hex_to_rgba(colors[stage_name], 0.4),
                        line_color=colors[stage_name],
                        box_visible=True, meanline_visible=True,
                        opacity=0.8
                ))
                    means.append({"Etapa": stage_name, "Media (g/t)": vals.mean(), "Mediana (g/t)": vals.median()})

            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=400,
                yaxis_title="Concentración (g/t)",
                showlegend=True,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            if means:
                df_means = pd.DataFrame(means)
                df_means["Media (g/t)"] = df_means["Media (g/t)"].round(2)
                df_means["Mediana (g/t)"] = df_means["Mediana (g/t)"].round(2)
                st.dataframe(df_means, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Concentración total y anomalías
    st.markdown("### Detección de Anomalías — Concentración Total")
    stage_cols = {
        "Alimentación": ["rougher.input.feed_au","rougher.input.feed_ag","rougher.input.feed_pb"],
        "Rougher": ["rougher.output.concentrate_au","rougher.output.concentrate_ag","rougher.output.concentrate_pb"],
        "Final": ["final.output.concentrate_au","final.output.concentrate_ag","final.output.concentrate_pb"],
    }

    col1, col2, col3 = st.columns(3)
    cols_ui = [col1, col2, col3]

    for (stage, stage_c), ui_col in zip(stage_cols.items(), cols_ui):
        total_col = f"total_{stage.lower()}"
        if total_col in train_df.columns:
            zeros = (train_df[total_col] == 0).sum()
            with ui_col:
                fig = px.histogram(
                    train_df, x=total_col, nbins=60,
                    title=f"Concentración Total — {stage}",
                    template=PLOTLY_TEMPLATE,
                    color_discrete_sequence=[GOLD_PALETTE[list(stage_cols.keys()).index(stage)]]
                )
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.metric(f"Valores = 0 ({stage})", f"{zeros:,}", delta=f"Eliminados ✓" if zeros > 0 else "Sin anomalías")


# ═════════════════════════════════════════════
# PÁGINA 4: MODELOS ML
# ═════════════════════════════════════════════
elif page == "🤖 Modelos ML":
    st.markdown("## 🤖 Modelos de Machine Learning")

    st.markdown("""
    <div class='info-box'>
    Se comparan 4 modelos con <b>validación cruzada de 5 folds</b>.
    La métrica es el <b style='color:#f5c842;'>sMAPE Final</b> (menor es mejor):<br>
    sMAPE final = <b>25%</b> × sMAPE<sub>rougher</sub> + <b>75%</b> × sMAPE<sub>final</sub>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Entrenando modelos con cross-validation... esto puede tardar ~1 minuto."):
        try:
            results, best_model, features, target = train_models(train_df)
        except Exception as e:
            st.error(f"Error al entrenar modelos: {e}")
            st.stop()

    # Tabla de resultados
    res_df = pd.DataFrame([
        {
            "Modelo": name,
            "sMAPE Promedio": round(v["smape_mean"], 3),
            "± Std": round(v["smape_std"], 3),
        }
        for name, v in results.items()
    ]).sort_values("sMAPE Promedio")

    # Gráfico de barras
    colors_bar = ["#22c55e" if i == 0 else "#3b82f6" if i == 1 else "#9ca3af"
                  for i in range(len(res_df))]
    fig = go.Figure(go.Bar(
        x=res_df["Modelo"], y=res_df["sMAPE Promedio"],
        error_y=dict(type="data", array=res_df["± Std"].values, visible=True),
        marker_color=colors_bar,
        text=res_df["sMAPE Promedio"].round(2),
        textposition="outside"
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=380,
        title="Comparación de Modelos — sMAPE Final (CV 5-Fold)",
        yaxis_title="sMAPE (↓ mejor)",
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla
    st.dataframe(
        res_df.style.highlight_min(subset=["sMAPE Promedio"], color="#1a3a1a"),
        use_container_width=True, hide_index=True
    )

    # Box plot por fold
    st.markdown("### Distribución del Error por Fold")
    fig2 = go.Figure()
    for i, (name, v) in enumerate(results.items()):
        fig2.add_trace(go.Box(
            y=v["scores"], name=name,
            marker_color=GOLD_PALETTE[i],
            boxmean=True
        ))
    fig2.update_layout(
        template=PLOTLY_TEMPLATE, height=350,
        yaxis_title="sMAPE por Fold",
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Ganador
    winner = res_df.iloc[0]["Modelo"]
    winner_score = res_df.iloc[0]["sMAPE Promedio"]
    baseline_score = results["Baseline (Media)"]["smape_mean"]
    improvement = ((baseline_score - winner_score) / baseline_score) * 100

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a3a1a 0%, #1e2030 100%);
                border: 1px solid #22c55e; border-radius: 12px; padding: 20px; margin-top: 20px;'>
        <h3 style='color:#22c55e; margin:0 0 8px;'>🏆 Mejor Modelo: {winner}</h3>
        <p style='color:#d0d0e0; margin:0;'>
            sMAPE Final: <b style='color:#f5c842;'>{winner_score:.3f}</b> &nbsp;|&nbsp;
            Mejora sobre baseline: <b style='color:#22c55e;'>{improvement:.1f}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PÁGINA 5: RESULTADOS FINALES
# ═════════════════════════════════════════════
elif page == "📊 Resultados Finales":
    st.markdown("## 📊 Resultados Finales del Proyecto")

    with st.spinner("Cargando resultados del modelo final..."):
        try:
            results, best_model, features, target = train_models(train_df)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    best_smape = results["Random Forest"]["smape_mean"]
    baseline_smape = results["Baseline (Media)"]["smape_mean"]

    # KPIs finales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mejor Modelo", "Random Forest")
    with col2:
        st.metric("sMAPE Final", f"{best_smape:.3f}", delta=f"-{baseline_smape - best_smape:.2f} vs baseline", delta_color="inverse")
    with col3:
        st.metric("sMAPE Baseline", f"{baseline_smape:.3f}")
    with col4:
        improvement = ((baseline_smape - best_smape) / baseline_smape) * 100
        st.metric("Mejora vs Baseline", f"{improvement:.1f}%")

    st.markdown("---")

    # Predicciones en conjunto de validación
    st.markdown("### 🎯 Predicciones vs Valores Reales")

    # Dividir manualmente para mostrar resultados
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
    val_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    val_model.fit(X_tr, y_tr)
    preds = val_model.predict(X_val)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            x=y_val["rougher.output.recovery"].values[:500],
            y=preds[:500, 0],
            template=PLOTLY_TEMPLATE,
            title="Rougher — Real vs Predicho",
            labels={"x": "Real (%)", "y": "Predicho (%)"},
            opacity=0.5, color_discrete_sequence=["#3b82f6"]
        )
        fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=100,
                      line=dict(color="#f5c842", dash="dash"))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            x=y_val["final.output.recovery"].values[:500],
            y=preds[:500, 1],
            template=PLOTLY_TEMPLATE,
            title="Final — Real vs Predicho",
            labels={"x": "Real (%)", "y": "Predicho (%)"},
            opacity=0.5, color_discrete_sequence=["#f5c842"]
        )
        fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=100,
                      line=dict(color="#3b82f6", dash="dash"))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Distribución del error
    st.markdown("### Distribución del Error de Predicción")
    errors_rougher = np.abs(y_val["rougher.output.recovery"].values - preds[:, 0])
    errors_final   = np.abs(y_val["final.output.recovery"].values - preds[:, 1])

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=errors_rougher, name="Error Rougher",
                               opacity=0.7, marker_color="#3b82f6", nbinsx=50))
    fig.add_trace(go.Histogram(x=errors_final, name="Error Final",
                               opacity=0.7, marker_color="#f5c842", nbinsx=50))
    fig.update_layout(
        barmode="overlay", template=PLOTLY_TEMPLATE, height=300,
        xaxis_title="Error Absoluto (%)", yaxis_title="Frecuencia",
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Conclusiones
    st.markdown("### 📝 Conclusiones del Proyecto")
    st.markdown("""
    <div class='info-box'>
    <b style='color:#f5c842; font-size:15px;'>1. Proceso de purificación funciona correctamente</b><br>
    La concentración de oro aumenta de ~7.7 g/t en la alimentación a ~42.7 g/t en el concentrado final, confirmando que el proceso extrae el metal de forma eficiente.<br><br>

    <b style='color:#f5c842; font-size:15px;'>2. Random Forest es el modelo ganador</b><br>
    Con un sMAPE final de <b>{:.3f}</b>, supera al baseline ({:.3f}) en un <b>{:.1f}%</b>.
    Su capacidad para capturar relaciones no lineales entre variables del proceso lo hace superior a la regresión lineal.<br><br>

    <b style='color:#f5c842; font-size:15px;'>3. Diferencias en distribuciones Train vs Test</b><br>
    El test KS detectó diferencias significativas en el tamaño de partículas, lo que sugiere que el modelo podría enfrentar desafíos de generalización en producción.<br><br>

    <b style='color:#f5c842; font-size:15px;'>4. Alineación de features es crítica</b><br>
    El conjunto de test no contiene variables de output del proceso, lo que requirió una selección cuidadosa de features para evitar data leakage.
    </div>
    """.format(best_smape, baseline_smape, improvement), unsafe_allow_html=True)
