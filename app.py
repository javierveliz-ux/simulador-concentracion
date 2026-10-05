"""
Simulador interactivo de indicadores de concentración de mercado (Monte Carlo)
Organización Industrial - Taller práctico
Ejecutar con:  streamlit run app.py
"""
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# 1. INDICADORES DE CONCENTRACIÓN
#    Todas las funciones reciben una matriz S (filas = iteraciones/casos,
#    columnas = empresas) con cuotas en fracción (0 a 1) y devuelven un
#    vector con un valor por fila.
# ---------------------------------------------------------------------------
def cr_k(S, k):
    """Ratio de concentración: suma de las k mayores cuotas (en %)."""
    S = np.atleast_2d(S)
    return np.sort(S, axis=1)[:, -k:].sum(axis=1) * 100


def hhi(S):
    """Herfindahl-Hirschman: suma de cuotas^2, con cuotas en % (rango 0-10.000)."""
    S = np.atleast_2d(S)
    return (S ** 2).sum(axis=1) * 10000


def dominancia(S):
    """Índice de Dominancia: suma de (s_i^2 / IHH)^2. Rango [1/N, 1]."""
    S = np.atleast_2d(S)
    h = (S ** 2).sum(axis=1)
    return ((S ** 2 / h[:, None]) ** 2).sum(axis=1)


def entropia(S):
    """Entropía de Shannon (ln): -suma s_i ln(s_i). Rango [0, ln N]."""
    S = np.atleast_2d(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(S > 0, S * np.log(S), 0.0)
    return -t.sum(axis=1)


def calcular(S, indicador, k, N):
    if indicador == "CRk":
        return cr_k(S, k)
    if indicador == "IHH":
        return hhi(S)
    if indicador == "ID":
        return dominancia(S)
    return entropia(S)


# ---------------------------------------------------------------------------
# 2. MOTOR DE MONTE CARLO
#    Dirichlet(1,...,1): distribución uniforme sobre el simplex. Cada vector
#    cumple s_i >= 0 y suma(s_i) = 1 en cada iteración.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simular(N, iteraciones, semilla):
    rng = np.random.default_rng(semilla)
    S = rng.dirichlet(np.ones(N), size=iteraciones)
    assert np.allclose(S.sum(axis=1), 1.0), "Las cuotas no suman 1"
    return S


# ---------------------------------------------------------------------------
# 3. CLASIFICACIÓN POR UMBRALES TEÓRICOS
# ---------------------------------------------------------------------------
def clasificar(valor, indicador, k, N):
    """Devuelve (nivel, texto con los umbrales usados)."""
    if indicador == "IHH":
        nivel = "Baja" if valor < 1500 else ("Moderada" if valor <= 2500 else "Alta")
        return nivel, ("Umbrales (DOJ/FTC clásicos): IHH < 1.500 = baja; "
                       "1.500-2.500 = moderada; > 2.500 = alta.")
    if indicador == "CRk" and k == 4:
        nivel = "Baja" if valor < 40 else ("Moderada" if valor <= 60 else "Alta")
        return nivel, "Umbrales CR4: < 40 % = baja; 40-60 % = moderada; > 60 % = alta."
    if indicador == "CRk":
        base = k / N * 100
        norm = (valor - base) / (100 - base)
        nivel = "Baja" if norm < 0.25 else ("Moderada" if norm <= 0.5 else "Alta")
        return nivel, (f"CR{k} se normaliza respecto al mínimo posible ({base:.1f} %): "
                       f"(CR - mín)/(100 - mín) = {norm:.2f}. < 0,25 = baja; "
                       "0,25-0,50 = moderada; > 0,50 = alta.")
    if indicador == "ID":
        norm = (valor - 1 / N) / (1 - 1 / N)
        nivel = "Baja" if norm < 0.25 else ("Moderada" if norm <= 0.5 else "Alta")
        return nivel, (f"ID normalizado = (ID - 1/N)/(1 - 1/N) = {norm:.2f}. "
                       "< 0,25 = baja; 0,25-0,50 = moderada; > 0,50 = alta.")
    ratio = valor / np.log(N)
    nivel = "Alta" if ratio < 0.5 else ("Moderada" if ratio < 0.8 else "Baja")
    return nivel, (f"Entropía relativa = IE/ln(N) = {ratio:.2f}. Mucha entropía = mucha "
                   "igualdad = poca concentración: < 0,50 = alta; 0,50-0,80 = moderada; "
                   "> 0,80 = baja.")


# ---------------------------------------------------------------------------
# 4. INTERFAZ
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Concentración de mercado", layout="wide")
st.title("Simulador de concentración de mercado (Monte Carlo)")

with st.sidebar:
    st.header("Parámetros")
    indicador = st.selectbox("Indicador", ["CRk", "IHH", "ID", "IE"],
                             format_func=lambda x: {
                                 "CRk": "Ratio de concentración (CRk)",
                                 "IHH": "Herfindahl-Hirschman (IHH)",
                                 "ID": "Índice de Dominancia (ID)",
                                 "IE": "Índice de Entropía (IE)"}[x])
    N = st.number_input("Número de empresas (N)", min_value=2, max_value=100,
                        value=5, step=1)
    N = int(N)
    k = 1
    if indicador == "CRk":
        k = st.slider("k (empresas más grandes)", 1, N - 1, min(4, N - 1))

    iteraciones = st.number_input(
        "Iteraciones de Monte Carlo", min_value=100, max_value=50000,
        value=1000, step=100,
        help="Mínimo 100: con menos la distribución es muy ruidosa. "
             "Máximo 50.000: más allá el tiempo y la memoria en un servidor "
             "web gratuito crecen sin mejorar visiblemente la precisión.")
    st.warning(
        "⚠️ Más iteraciones = más precisión, pero también más tiempo de "
        "respuesta, más memoria RAM (matriz de iteraciones × N) y más uso de "
        "CPU. En servidores gratuitos valores altos pueden volver lenta o "
        "inestable la app.")
    semilla = st.number_input("Semilla aleatoria", value=42, step=1)

# --- Simulación ---
t0 = time.time()
S_sim = simular(N, int(iteraciones), int(semilla))
valores_sim = calcular(S_sim, indicador, k, N)
t_sim = time.time() - t0

# --- Caso particular ---
st.header("1. Caso particular")
modo = st.radio("¿Cómo definir el caso?", ["Generación aleatoria", "Entrada manual"],
                horizontal=True)
caso = None

if modo == "Generación aleatoria":
    if st.button("🎲 Generar nuevo caso aleatorio") or \
            st.session_state.get("caso_N") != N or "caso_rand" not in st.session_state:
        st.session_state["caso_rand"] = np.random.default_rng().dirichlet(np.ones(N))
        st.session_state["caso_N"] = N
    caso = st.session_state["caso_rand"]
else:
    st.caption("Ingresa N cuotas en %, separadas por punto y coma, espacio o salto "
               "de línea (decimales con punto). Deben sumar 100.")
    texto = st.text_area("Cuotas (%)", value=";".join([f"{100 / N:.2f}"] * N))
    try:
        partes = [p for p in re.split(r"[;\s]+", texto.strip()) if p]
        v = np.array([float(p) for p in partes])
        if len(v) != N:
            st.error(f"Ingresaste {len(v)} cuotas y N = {N}.")
        elif np.any(v < 0) or np.any(v > 100):
            st.error("Cada cuota debe estar entre 0 % y 100 %.")
        elif not np.isclose(v.sum(), 100, atol=0.01):
            st.error(f"Las cuotas suman {v.sum():.2f} %, deben sumar 100 %.")
        else:
            caso = v / 100
    except ValueError:
        st.error("Formato inválido: usa solo números.")

if caso is None:
    st.stop()

st.write("Cuotas del caso (%):", np.round(caso * 100, 2).tolist())
valor_caso = float(calcular(caso, indicador, k, N)[0])
percentil = float((valores_sim <= valor_caso).mean() * 100)

# --- Gráfico ---
st.header("2. Distribución de Monte Carlo y posición del caso")
nombres = {"CRk": f"CR{k} (%)", "IHH": "IHH (puntos, 0-10.000)",
           "ID": "Índice de Dominancia (0-1)", "IE": "Entropía (nats)"}
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(valores_sim, bins=40, density=True, alpha=0.7, color="steelblue",
        edgecolor="white", label=f"{int(iteraciones)} simulaciones")
ax.axvline(valor_caso, color="red", linewidth=2.5,
           label=f"Caso particular = {valor_caso:.3f} (percentil {percentil:.1f})")
ax.set_xlabel(nombres[indicador])
ax.set_ylabel("Densidad de probabilidad (1/unidad del indicador)")
ax.set_title(f"Distribución empírica de {nombres[indicador]} con N = {N} empresas")
ax.legend()
st.pyplot(fig)
st.caption(f"Simulación ejecutada en {t_sim:.3f} s. Media simulada: "
           f"{valores_sim.mean():.3f}. El caso supera al {percentil:.1f} % de las simulaciones.")

# --- Evaluador ---
st.header("3. Evaluador")
nivel_correcto, umbrales = clasificar(valor_caso, indicador, k, N)
resp = st.radio(f"Según {nombres[indicador]}, ¿cómo clasificarías la concentración "
                "del caso particular?", ["Baja", "Moderada", "Alta"], index=None)
if st.button("Verificar respuesta"):
    if resp is None:
        st.info("Elige una opción primero.")
    else:
        msg = (f"**Valor del caso:** {valor_caso:.3f}.  \n{umbrales}  \n"
               f"**Clasificación correcta:** {nivel_correcto}.  \n"
               f"**Posición relativa:** percentil {percentil:.1f} de la distribución "
               f"simulada (equivale a que solo {100 - percentil:.1f} % de los mercados "
               "aleatorios están más concentrados según este indicador"
               + (", salvo en la entropía, donde un valor bajo implica más concentración)."
                  if indicador == "IE" else ")."))
        if resp == nivel_correcto:
            st.success("✅ ¡Correcto!")
        else:
            st.error(f"❌ Incorrecto. Respondiste «{resp}».")
        st.markdown(msg)
