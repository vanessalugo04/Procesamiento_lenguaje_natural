'''
import streamlit as st
import json
import os
from pypdf import PdfReader
# from reglas_proyecto import verbs_rules, exceptions, reglas_morfologicas

#cerpeta de corpus
CARPETA_CORPUS = "corpus"
ARCHIVO_JSON   = "corpus_procesado.json"

st.set_page_config(
    page_title="Detector de Plagio",
    page_icon="🔍",
    layout="wide"
)

#just estilo 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
    }
    .metric-card {
        background: #0f0f0f;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        color: white;
    }
    .metric-card .valor {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        color: #00e5a0;
    }
    .metric-card .etiqueta {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .doc-chip {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        color: #ccc;
    }
    .doc-chip.admin { border-color: #00e5a0; color: #00e5a0; }
    .doc-chip.usuario { border-color: #7b61ff; color: #7b61ff; }
    .resultado-card {
        border-radius: 14px;
        padding: 1.4rem;
        margin: 0.7rem 0;
        border: 1px solid #2a2a2a;
        background: #111;
    }
    .badge-plagio {
        background: #ff4d4d22;
        color: #ff4d4d;
        border: 1px solid #ff4d4d55;
        border-radius: 8px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-sospechoso {
        background: #ffaa0022;
        color: #ffaa00;
        border: 1px solid #ffaa0055;
        border-radius: 8px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ok {
        background: #00e5a022;
        color: #00e5a0;
        border: 1px solid #00e5a055;
        border-radius: 8px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .stButton > button {
        background: #00e5a0;
        color: #000;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: #00c88a;
        color: #000;
    }
    section[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #333 !important;
        border-radius: 12px !important;
        background: #0a0a0a !important;
    }
</style>
""", unsafe_allow_html=True)

#funciones que ya tenemos de codigos pasados 
def leer_pdf(file):
    reader = PdfReader(file)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() or ""
    return texto

def to_minusculas(texto):
    letras = ""
    for letra in texto:
        if ord(letra) >= 65 and ord(letra) <= 90:
            letras += chr(ord(letra) + 32)
        else:
            letras += letra
    return letras

def contador_palabras(texto):
    contador = 0
    en_palabra = False
    for caracter in texto:
        if caracter not in (" ", "\n", "\t"):
            if not en_palabra:
                contador += 1
                en_palabra = True
        else:
            en_palabra = False
    return contador

STOPWORDS = ([
    'a','all','also','although','among','an','and','are','around','as','at',
    'be','been','being','both','but','by','can','continue','despite','do','due',
    'each','for','from','general','has','have','however','in','include',
    'including','into','is','it','its','like','many','more','most','must',
    'nevertheless','of','on','one','or','other','overall','plays','provide',
    'provides','remains','such','that','the','their','these','this','to',
    'under','various','well','with'
])

def es_valido(c):
    code = ord(c)
    if (code >= 97 and code <= 122) or (code >= 65 and code <= 90):
        return True
    if code in (225, 233, 237, 243, 250, 241, 209):
        return True
    if code in (160, 130, 161, 162, 163, 181, 144, 214, 224):
        return True
    return False

def tokenizador(texto):
    token = ""
    tokens = []
    for ch in texto:
        if es_valido(ch):
            token += ch
        else:
            if token and token not in STOPWORDS:
                tokens.append(token)
            token = ""
    if token and token not in STOPWORDS:
        tokens.append(token)
    return tokens

def longitud(palabra):
    return sum(1 for _ in palabra)

def termina_en(palabra, sufijo):
    lp, ls = longitud(palabra), longitud(sufijo)
    return lp >= ls and palabra[lp - ls:] == sufijo

def lematizador(tokens):
    resultado = []
    for palabra in tokens:
        lema = None
        if palabra in exceptions:
            lema = exceptions[palabra]
        if lema is None:
            for sufijo, terminacion in verbs_rules:
                if (longitud(palabra) - longitud(sufijo) >= 2) and termina_en(palabra, sufijo):
                    lema = palabra[:-longitud(sufijo)] + terminacion
                    break
        if lema is None:
            for sufijo, reemplazo in reglas_morfologicas:
                min_raiz = 3 if sufijo in ('es', 's') else 2
                if longitud(palabra) - longitud(sufijo) >= min_raiz and termina_en(palabra, sufijo):
                    lema = palabra[:-longitud(sufijo)] + reemplazo
                    break
        resultado.append(lema if lema else palabra)
    return resultado

def preprocesar(texto_crudo):
    texto = to_minusculas(texto_crudo)
    tokens = tokenizador(texto)
    lemas  = lematizador(tokens)
    return lemas

#AQUI TENEMOS QUE IMPLEMENTAR COMO VA A IR COMPARANDO SI ES PLAGIO O NO

#corpus
def cargar_corpus():
    if os.path.exists(ARCHIVO_JSON):
        with open(ARCHIVO_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def guardar_corpus(corpus):
    with open(ARCHIVO_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

def inicializar_corpus_admin():
    if not os.path.exists(CARPETA_CORPUS):
        os.makedirs(CARPETA_CORPUS)
        return 0
    corpus = cargar_corpus()
    nuevos = 0
    for archivo in os.listdir(CARPETA_CORPUS):
        if not archivo.endswith(".pdf"):
            continue
        nombre = archivo.replace(".pdf", "")
        if nombre in corpus:
            continue
        try:
            tokens = preprocesar(leer_pdf(os.path.join(CARPETA_CORPUS, archivo)))
            corpus[nombre] = {"tokens": tokens, "fuente": "admin", "num_tokens": len(tokens)}
            nuevos += 1
        except Exception as e:
            st.warning(f"No se pudo leer {archivo}: {e}")
    if nuevos:
        guardar_corpus(corpus)
    return nuevos

def agregar_documento_usuario(nombre, file_object):
    corpus = cargar_corpus()
    nombre_final = nombre.replace(".pdf", "")
    base, contador = nombre_final, 1
    while nombre_final in corpus:
        nombre_final = f"{base}_{contador}"
        contador += 1
    tokens = preprocesar(leer_pdf(file_object))
    corpus[nombre_final] = {"tokens": tokens, "fuente": "usuario", "num_tokens": len(tokens)}
    guardar_corpus(corpus)
    return nombre_final, len(tokens)

# ─── Inicializar corpus admin ─────────────────────────────────────────────────
nuevos = inicializar_corpus_admin()

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("# Detector de Plagio")
st.markdown("##### Análisis automático contra el corpus de referencia")
st.markdown("---")

# ── Panel izquierdo / derecho ─────────────────────────────────────────────────
col_izq, col_der = st.columns([1, 1.6], gap="large")

# ── Columna izquierda: Corpus ─────────────────────────────────────────────────
with col_izq:
    st.markdown("### Corpus")

    corpus = cargar_corpus()
    total      = len(corpus)
    de_admin   = sum(1 for d in corpus.values() if d["fuente"] == "admin")
    de_usuario = sum(1 for d in corpus.values() if d["fuente"] == "usuario")

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="valor">{total}</div><div class="etiqueta">Total docs</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="valor">{de_admin}</div><div class="etiqueta">Admin</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="valor">{de_usuario}</div><div class="etiqueta">Usuarios</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if nuevos > 0:
        st.success(f" {nuevos} documento(s) admin cargados al iniciar")

    with st.expander("Ver documentos en el corpus", expanded=False):
        if corpus:
            chips = ""
            for nombre, datos in corpus.items():
                clase = "admin" if datos["fuente"] == "admin" else "usuario"
                chips += f'<span class="doc-chip {clase}">{"⚙" if clase=="admin" else "👤"} {nombre}</span>'
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("El corpus está vacío.")

    st.markdown("---")
    st.markdown("**➕ Agregar documentos al corpus**")
    archivos_corpus = st.file_uploader(
        "Sube PDFs para enriquecer el corpus",
        type="pdf",
        accept_multiple_files=True,
        key="corpus_uploader",
        label_visibility="collapsed"
    )
    if st.button("Agregar al corpus"):
        if not archivos_corpus:
            st.warning("Selecciona al menos un archivo.")
        else:
            for arch in archivos_corpus:
                nombre_final, n_tokens = agregar_documento_usuario(arch.name, arch)
                st.success(f" '{nombre_final}' — {n_tokens} tokens")
            st.rerun()

# ── Columna derecha: Análisis ─────────────────────────────────────────────────
with col_der:
    st.markdown("### Analizar documento")

    archivo_analizar = st.file_uploader(
        "Sube el PDF que quieres analizar",
        type="pdf",
        key="analizador"
    )

    if st.button("Analizar contra el corpus"):
        corpus = cargar_corpus()

        if not archivo_analizar:
            st.warning("Sube un archivo PDF para analizar.")
        elif not corpus:
            st.error("El corpus está vacío. Agrega documentos primero.")
        else:
            with st.spinner("Procesando..."):
                tokens_doc = preprocesar(leer_pdf(archivo_analizar))
                nombre_doc = archivo_analizar.name.replace(".pdf", "")

            pass 


st.caption("Proyecto PLN · Detector de Plagio")
'''