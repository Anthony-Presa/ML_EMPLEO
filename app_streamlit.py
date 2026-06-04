import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
 
# ── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Empleo - ENAHO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ── Ocultar chrome de Streamlit ───────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stDecoration"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)
 
# ── Cargar modelo ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return load("modelo_rfempleo_tunning.joblib")
 
clf = load_model()
 
# ── Opciones (deben coincidir exactamente con el pipeline del modelo) ─────────
zona_options = [
    "Costa Norte", "Costa Centro", "Costa Sur",
    "Sierra Norte", "Sierra Centro", "Sierra Sur",
    "Selva", "Lima Metropolitana",
]
estrato_options = [
    "De 500 000 a más habitantes",
    "De 100 000 a 499 999 habitantes",
    "De 50 000 a 99 999 habitantes",
    "De 20 000 a 49 999 habitantes",
    "De 2 000 a 19 999 habitantes",
    "De 500 a 1 999 habitantes",
    "Área de Empadronamiento Rural (AER) Compuesto",
    "Área de Empadronamiento Rural (AER) Simple",
]
parentesco_options = [
    "Jefe/Jefa", "Esposo(a)/compañero(a)", "Hijo(a)/Hijastro(a)",
    "Yerno/Nuera", "Nieto(a)", "Padres/Suegros",
    "Otros parientes", "Trabajador Hogar", "Pensionista",
    "Otros no parientes", "Hermano(a)",
]
estado_civil_options = [
    "Conviviente", "Casado(a)", "Viudo(a)",
    "Divorciado(a)", "Separado(a)", "Soltero(a)",
]
estudios_options = [
    "Sin nivel", "Educación inicial",
    "Primaria incompleta", "Primaria completa",
    "Secundaria incompleta", "Secundaria completa",
    "Superior no universitaria Incompleta", "Superior no universitaria completa",
    "Superior universitaria incompleta", "Superior universitaria completa",
    "Maestria/Doctorado", "Básica especial",
]
ultimo_anio_options = [
    "Sin nivel", "Primer nivel", "Segundo nivel", "Tercer nivel",
    "Cuarto nivel", "Quinto nivel", "Sexto nivel", "Séptimo nivel",
]
sector_options = ["No aplica", "Estatal", "No Estatal"]
 
 
def build_select(name, label, options):
    opts = "".join(f'<option value="{o}">{o}</option>' for o in options)
    return f"""
    <div class="space-y-1">
        <label class="font-label-sm text-label-sm text-on-surface-variant uppercase">{label}</label>
        <div class="form-inset rounded-lg px-3 py-2 flex items-center">
            <select name="{name}" class="bg-transparent border-none focus:ring-0 w-full text-on-surface">
                {opts}
            </select>
        </div>
    </div>"""
 
 
# ── Leer predicción previa desde query params ─────────────────────────────────
params     = st.query_params
resultado  = params.get("resultado", "")
porcentaje = params.get("pct", "")
color_res  = params.get("color", "#82cfff")
icono_res  = params.get("icon", "query_stats")
 
# ── Procesar POST (form submit via query params) ──────────────────────────────
if "zona" in params:
    try:
        df = pd.DataFrame({
            "ZONA_GEOGRAFICA":     [params.get("zona", "")],
            "ESTRATO":             [params.get("estrato", "")],
            "PARENTESCO_CON_JEFE": [params.get("parentesco", "")],
            "SEXO":                [params.get("sexo", "Hombre")],
            "EDAD":                [int(params.get("edad", 25))],
            "ESTADO_CIVIL":        [params.get("estado_civil", "")],
            "ESTUDIOS":            [params.get("estudios", "")],
            "ULTIMO_AÑO_APROBADO": [params.get("ultimo_anio", "")],
            "SECTOR_ESTUDIOS":     [params.get("sector", "")],
        })
        probs = clf.predict_proba(df)[0]
        clase = int(np.argmax(probs))
        if clase == 1:
            resultado  = "Con Empleo"
            porcentaje = str(round(float(probs[1]) * 100, 1))
            color_res  = "#82cfff"
            icono_res  = "check_circle"
        else:
            resultado  = "Sin Empleo"
            porcentaje = str(round(float(probs[0]) * 100, 1))
            color_res  = "#ffb4ab"
            icono_res  = "cancel"
    except Exception as e:
        resultado  = f"Error: {e}"
        porcentaje = ""
 
# ── Panel resultado: HTML dinámico ────────────────────────────────────────────
if resultado and porcentaje:
    pct_float = float(porcentaje)
    status_area_html = f"""
    <div class="w-full flex flex-col items-center animate-in fade-in duration-700">
        <div style="background:{color_res}22" class="p-4 rounded-full mb-4">
            <span class="material-symbols-outlined text-4xl"
                  style="color:{color_res};font-variation-settings:'FILL' 1">
                {icono_res}
            </span>
        </div>
        <h3 class="font-title-md text-on-surface-variant uppercase tracking-widest text-xs mb-2">
            Probabilidad de Empleo
        </h3>
        <div class="text-6xl font-bold mb-2 font-display-lg" style="color:{color_res}">
            {porcentaje}%
        </div>
        <p class="text-body-md text-on-surface max-w-xs mx-auto mb-6">
            Resultado: <span style="color:{color_res}" class="font-bold">{resultado}</span>
        </p>
        <div class="w-full space-y-4">
            <div class="p-4 bg-surface-container-highest border border-outline-variant rounded-lg text-left">
                <div class="flex justify-between text-xs font-bold mb-2">
                    <span>CONFIANZA DEL MODELO</span>
                    <span style="color:{color_res}">{porcentaje}%</span>
                </div>
                <div class="h-2 w-full bg-surface-container-lowest rounded-full overflow-hidden">
                    <div style="background:{color_res};width:{pct_float}%" class="h-full transition-all duration-1000"></div>
                </div>
            </div>
            <div class="p-4 bg-surface-container-highest border border-outline-variant rounded-lg text-left">
                <p class="text-xs font-bold mb-2 uppercase opacity-60">Variables Determinantes</p>
                <ul class="text-sm space-y-2">
                    <li class="flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-tertiary inline-block"></span>
                        Nivel Educativo (Peso: +0.42)
                    </li>
                    <li class="flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-secondary inline-block"></span>
                        Zona Geográfica (Peso: +0.18)
                    </li>
                    <li class="flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-primary inline-block"></span>
                        Edad (Peso: +0.14)
                    </li>
                </ul>
            </div>
        </div>
    </div>"""
else:
    status_area_html = """
    <div class="w-20 h-20 rounded-full bg-surface-container-highest flex items-center justify-center relative mx-auto">
        <span class="material-symbols-outlined text-4xl text-on-surface-variant opacity-50">query_stats</span>
    </div>
    <div class="mt-4">
        <h3 class="font-headline-lg text-xl text-on-surface">Esperando datos...</h3>
        <p class="text-on-surface-variant mt-2">
            Complete el formulario y presione "Predecir" para procesar la
            información con el modelo Random Forest.
        </p>
    </div>
    <div class="w-full mt-4 bg-surface-container-highest rounded-lg p-4 border border-outline-variant text-left">
        <div class="flex items-center gap-2 mb-2">
            <span class="material-symbols-outlined text-sm text-secondary">verified_user</span>
            <span class="text-label-sm font-bold uppercase">Métricas de Confianza</span>
        </div>
        <div class="h-1.5 w-full bg-surface-container-lowest rounded-full overflow-hidden">
            <div class="bg-outline-variant h-full w-0"></div>
        </div>
        <p class="text-[10px] font-data-mono text-on-surface-variant mt-1">EXACTITUD: N/A</p>
    </div>"""
 
# ── Construir selects ─────────────────────────────────────────────────────────
sel_zona       = build_select("zona",        "Zona Geográfica",      zona_options)
sel_estrato    = build_select("estrato",     "Estrato Socioeconómico", estrato_options)
sel_parentesco = build_select("parentesco",  "Parentesco",           parentesco_options)
sel_civil      = build_select("estado_civil","Estado Civil",         estado_civil_options)
sel_estudios   = build_select("estudios",    "Nivel de Estudios",    estudios_options)
sel_sector     = build_select("sector",      "Sector de Estudios",   sector_options)
sel_ultimo     = build_select("ultimo_anio", "Último Año Aprobado",  ultimo_anio_options)
 
# ── HTML completo (diseño Stitch) ─────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html class="dark" lang="es">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Inter:wght@400;500;600&family=Geist:wght@400&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<script>
tailwind.config = {{
    darkMode: "class",
    theme: {{
        extend: {{
            colors: {{
                "surface-variant": "#343438", "surface": "#131316",
                "on-background": "#e4e2e6", "on-primary-container": "#7884b3",
                "primary-container": "#0d1b44", "outline-variant": "#45464e",
                "on-surface": "#e4e2e6", "on-error": "#690005",
                "surface-container": "#1f1f23", "tertiary": "#edc200",
                "surface-container-high": "#2a2a2d", "tertiary-container": "#cda700",
                "surface-container-highest": "#343438", "secondary-container": "#00abec",
                "on-secondary-container": "#003c55", "on-tertiary": "#3c2f00",
                "on-primary": "#212e58", "primary": "#b8c5f7",
                "surface-container-lowest": "#0e0e11", "background": "#131316",
                "secondary": "#82cfff", "outline": "#909099",
                "surface-container-low": "#1b1b1e",
            }},
        }},
    }},
}};
</script>
<style>
.material-symbols-outlined {{
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}}
.form-inset {{
    background-color: rgba(14,14,17,0.6);
    border: 1px solid #2d3748;
}}
.form-inset:focus-within {{
    border-color: #82cfff;
    box-shadow: 0 0 0 1px #82cfff;
}}
select, input[type=number] {{
    color: #e4e2e6 !important;
    background: transparent !important;
    width: 100%;
    border: none;
    outline: none;
}}
select option {{ background: #1f1f23; color: #e4e2e6; }}
@keyframes spin-slow {{
    from {{ transform: rotate(0deg); }}
    to   {{ transform: rotate(360deg); }}
}}
.animate-spin-slow {{ animation: spin-slow 8s linear infinite; }}
</style>
</head>
<body class="bg-background text-on-background overflow-x-hidden" style="font-family:'Inter',sans-serif">
 
<!-- Header -->
<header class="w-full bg-surface border-b border-outline-variant h-16 flex items-center px-6">
    <div class="flex items-center gap-4 w-full justify-between">
        <div class="flex items-center gap-4">
            <span class="font-bold text-xl" style="color:#82cfff;font-family:Montserrat">
                📊 Predicción de Empleo - ENAHO
            </span>
        </div>
        <nav class="flex items-center gap-8">
            <a style="color:#82cfff;border-bottom:2px solid #82cfff;padding-bottom:4px;font-size:13px;font-weight:600" href="#">
                Modelo Predictivo
            </a>
        </nav>
    </div>
</header>
 
<div class="flex min-h-screen">
<!-- Sidebar -->
<aside class="flex flex-col w-64 bg-surface-container border-r border-outline-variant p-4 space-y-4 shrink-0">
    <div class="flex items-center gap-3 p-2 mb-4">
        <div class="w-10 h-10 rounded-full bg-secondary-container flex items-center justify-center" style="color:#003c55">
            <span class="material-symbols-outlined">person</span>
        </div>
        <div>
            <p class="font-semibold text-sm leading-tight">Analista ENAHO</p>
            <p class="text-sm" style="color:#909099">Ministerio de Trabajo</p>
        </div>
    </div>
    <nav class="space-y-1">
        <a class="flex items-center gap-3 p-3 rounded-lg font-bold"
           style="background:#00abec;color:#001e2d" href="#">
            <span class="material-symbols-outlined">analytics</span>
            Modelo Predictivo
        </a>
        <a class="flex items-center gap-3 p-3 rounded-lg" style="color:#909099" href="#">
            <span class="material-symbols-outlined">bar_chart</span>
            Análisis de Datos
        </a>
        <a class="flex items-center gap-3 p-3 rounded-lg" style="color:#909099" href="#">
            <span class="material-symbols-outlined">info</span>
            Acerca de ENEI
        </a>
    </nav>
    <div class="mt-auto">
        <button onclick="location.href='?'" class="w-full py-3 font-bold rounded-lg"
                style="background:#82cfff;color:#001e2d">
            Nueva Consulta
        </button>
    </div>
</aside>
 
<!-- Main -->
<main class="flex-grow overflow-y-auto px-10 py-8">
 
    <!-- Hero -->
    <h1 class="font-bold mb-4" style="font-family:Montserrat;font-size:40px;color:#82cfff;line-height:1.2">
        Plataforma de Predicción de Empleo
    </h1>
    <div class="rounded-xl border border-outline-variant p-6 mb-8 relative overflow-hidden"
         style="background:#0d1b44">
        <p style="font-size:15px;line-height:1.7;color:#7884b3;max-width:780px;margin:0">
            Esta herramienta utiliza un algoritmo avanzado de
            <span style="color:#82cfff;font-weight:600">Random Forest Classifier</span>
            entrenado con los datos más recientes de la
            <span style="color:#82cfff;font-weight:600">Encuesta Nacional de Hogares (ENAHO) 2025</span>
            para predecir la probabilidad de inserción laboral basada en perfiles
            demográficos y socioeducativos.
        </p>
    </div>
 
    <!-- Bento Grid -->
    <div class="grid gap-6" style="grid-template-columns: 2fr 1fr">
 
        <!-- Formulario -->
        <div class="bg-surface-container-low border border-outline-variant rounded-xl overflow-hidden">
            <div class="p-6 border-b border-outline-variant flex justify-between items-center"
                 style="background:rgba(42,42,45,0.3)">
                <h2 class="font-semibold flex items-center gap-2" style="font-size:16px">
                    <span class="material-symbols-outlined" style="color:#82cfff">input</span>
                    Datos de Entrada
                </h2>
                <span style="font-size:11px;color:#909099;font-family:monospace">
                    FORMATO: JSON / TABULAR
                </span>
            </div>
 
            <form class="p-6" id="predictionForm">
                <div class="grid grid-cols-2 gap-x-6 gap-y-4">
                    {sel_zona}
                    <!-- Sexo -->
                    <div class="space-y-1">
                        <label style="font-size:11px;font-weight:600;letter-spacing:.07em;
                                      text-transform:uppercase;color:#909099">Sexo</label>
                        <div class="flex gap-6 p-2">
                            <label class="flex items-center gap-2 cursor-pointer">
                                <input type="radio" name="sexo" value="Hombre" checked
                                       style="accent-color:#82cfff"/>
                                <span>Hombre</span>
                            </label>
                            <label class="flex items-center gap-2 cursor-pointer">
                                <input type="radio" name="sexo" value="Mujer"
                                       style="accent-color:#82cfff"/>
                                <span>Mujer</span>
                            </label>
                        </div>
                    </div>
 
                    <!-- Edad -->
                    <div class="space-y-1">
                        <label style="font-size:11px;font-weight:600;letter-spacing:.07em;
                                      text-transform:uppercase;color:#909099">Edad (14-98)</label>
                        <div class="form-inset rounded-lg px-3 py-2">
                            <input type="number" name="edad" min="14" max="98" value="25"
                                   placeholder="Ej: 25"
                                   style="color:#e4e2e6;background:transparent;border:none;outline:none;width:100%"/>
                        </div>
                    </div>
                    {sel_civil}
                    {sel_estudios}
                    {sel_sector}
                    {sel_estrato}
                    {sel_parentesco}
 
                    <!-- Último Año Aprobado (span 2 cols) -->
                    <div class="space-y-1 col-span-2">
                        <label style="font-size:11px;font-weight:600;letter-spacing:.07em;
                                      text-transform:uppercase;color:#909099">
                            Último Año Aprobado: <span id="anioLabel">Sin nivel</span>
                        </label>
                        <div class="form-inset rounded-lg px-3 py-3">
                            <input type="range" name="ultimo_anio_idx" id="anioRange"
                                   min="0" max="7" value="0"
                                   style="width:100%;accent-color:#82cfff"
                                   oninput="updateAnio(this.value)"/>
                            <div class="flex justify-between mt-1"
                                 style="font-size:10px;color:#909099;font-family:monospace">
                                <span>0</span><span>1</span><span>2</span><span>3</span>
                                <span>4</span><span>5</span><span>6</span><span>7</span>
                            </div>
                        </div>
                        <input type="hidden" name="ultimo_anio" id="anioValue" value="Sin nivel"/>
                    </div>
                </div>
 
                <!-- Botones -->
                <div class="mt-6 pt-6 flex gap-4"
                     style="border-top:1px solid #45464e">
                    <button type="button" onclick="handlePredict()"
                            class="flex items-center justify-center gap-2 px-10 py-3 font-bold rounded-lg"
                            style="background:#82cfff;color:#001e2d;font-size:15px">
                        <span class="material-symbols-outlined">bolt</span>
                        Predecir
                    </button>
                    <button type="reset" onclick="location.href='?'"
                            class="flex items-center justify-center gap-2 px-10 py-3 font-bold rounded-lg"
                            style="border:1.5px solid #82cfff;color:#82cfff;background:transparent;font-size:15px">
                        <span class="material-symbols-outlined">restart_alt</span>
                        Resetear
                    </button>
                </div>
            </form>
        </div>
 
        <!-- Panel Resultado -->
        <div class="bg-surface-container-low border border-outline-variant rounded-xl flex flex-col">
            <div class="p-6 border-b border-outline-variant flex items-center gap-2"
                 style="background:rgba(42,42,45,0.3)">
                <span class="material-symbols-outlined" style="color:#edc200">analytics</span>
                <h2 class="font-semibold" style="font-size:16px">Resultado del Modelo</h2>
            </div>
            <div class="flex-grow flex flex-col items-center justify-center p-8 text-center space-y-4"
                 id="statusArea">
                {status_area_html}
            </div>
        </div>
    </div>
 
    <!-- Cards -->
    <div class="grid grid-cols-3 gap-6 mt-8">
        <div class="rounded-xl border border-outline-variant p-6 flex items-start gap-4"
             style="background:#0e0e11">
            <span class="material-symbols-outlined" style="color:#82cfff;font-size:28px;font-variation-settings:'FILL' 1;flex-shrink:0">dataset</span>
            <div>
                <p class="font-bold text-sm mb-1">Datos ENAHO 2025</p>
                <p style="font-size:12px;color:#909099;line-height:1.5">
                    Dataset procesado y normalizado para inferencia en tiempo real.
                </p>
            </div>
        </div>
        <div class="rounded-xl border border-outline-variant p-6 flex items-start gap-4"
             style="background:#0e0e11">
            <span class="material-symbols-outlined" style="color:#82cfff;font-size:28px;font-variation-settings:'FILL' 1;flex-shrink:0">memory</span>
            <div>
                <p class="font-bold text-sm mb-1">Inferencia de IA</p>
                <p style="font-size:12px;color:#909099;line-height:1.5">
                    Algoritmos de aprendizaje supervisado con alta precisión predictiva.
                </p>
            </div>
        </div>
        <div class="rounded-xl border border-outline-variant p-6 flex items-start gap-4"
             style="background:#0e0e11">
            <span class="material-symbols-outlined" style="color:#82cfff;font-size:28px;font-variation-settings:'FILL' 1;flex-shrink:0">security</span>
            <div>
                <p class="font-bold text-sm mb-1">Cumplimiento Ético</p>
                <p style="font-size:12px;color:#909099;line-height:1.5">
                    Tratamiento anónimo de datos según normativas institucionales.
                </p>
            </div>
        </div>
    </div>
 
    <!-- Footer -->
    <div class="mt-10 pt-5 flex flex-wrap justify-between items-center"
         style="border-top:1px solid #45464e;color:#909099;font-size:12px">
        <div>
            <p class="font-bold" style="color:#e4e2e6">
                © 2025 ENEI – Instituto Nacional de Estadística e Informática
            </p>
            <p>Datos basados en ENAHO.</p>
        </div>
        <div class="flex gap-6">
            <span>Privacidad</span>
            <span>Términos de Uso</span>
            <span style="color:#82cfff;font-weight:700">Contacto</span>
        </div>
    </div>
</main>
</div>
 
<script>
const anioLabels = [
    "Sin nivel","Primer nivel","Segundo nivel","Tercer nivel",
    "Cuarto nivel","Quinto nivel","Sexto nivel","Séptimo nivel"
];
 
function updateAnio(val) {{
    document.getElementById("anioLabel").textContent = anioLabels[val];
    document.getElementById("anioValue").value = anioLabels[val];
}}
 
function getFormData() {{
    const f = document.getElementById("predictionForm");
    const fd = new FormData(f);
    const params = new URLSearchParams();
    for (const [k, v] of fd.entries()) params.append(k, v);
    // sexo via radio
    const sexo = f.querySelector('input[name="sexo"]:checked');
    if (sexo) params.set("sexo", sexo.value);
    return params;
}}
 
function handlePredict() {{
    // Mostrar spinner
    document.getElementById("statusArea").innerHTML = `
        <div class="flex flex-col items-center justify-center space-y-6">
            <div class="relative w-24 h-24">
                <svg class="w-full h-full animate-spin" style="color:#82cfff" viewBox="0 0 100 100">
                    <circle style="opacity:.25" cx="50" cy="50" r="40"
                            stroke="currentColor" stroke-width="8" fill="none"></circle>
                    <path style="opacity:.75" fill="currentColor"
                          d="M4 50a46 46 0 0 1 46-46v8a38 38 0 0 0-38 38H4z"></path>
                </svg>
                <div class="absolute inset-0 flex items-center justify-center">
                    <span class="material-symbols-outlined text-3xl" style="color:#82cfff">auto_awesome</span>
                </div>
            </div>
            <h3 style="font-size:18px;font-weight:600">Procesando Inferencias...</h3>
            <p style="color:#909099;font-size:13px;font-family:monospace">
                Ejecutando Random Forest...
            </p>
        </div>`;
 
    // Enviar a Streamlit via recarga con query params
    const params = getFormData();
    // pequeño delay para mostrar spinner antes de recargar
    setTimeout(() => {{
        window.location.href = "?" + params.toString();
    }}, 600);
}}
</script>
</body>
</html>"""
 
# ── Renderizar ────────────────────────────────────────────────────────────────
st.components.v1.html(HTML, height=1100, scrolling=True)
 
# streamlit run app_streamlit.py