import streamlit as st
from pypdf import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
from reglas_proyecto import verbs_rules, exceptions, reglas_morfologicas

#funcion para leer el pdf 
def leer_pdf(file):
    reader = PdfReader(file)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() or ""
    return texto

#funciones de preprocesamiento
def contador_palabras(texto):

    contador = 0
    en_palabra = False

    for caracter in texto:

        if caracter != " " and caracter != "\n" and caracter != "\t":

            if not en_palabra:
                contador += 1
                en_palabra = True

        else:
            en_palabra = False

    return contador

def to_minusculas(texto):

    letras = ""

    for letra in texto:

        if ord(letra) >= 65 and ord(letra) <= 90:
            nueva_letra = chr(ord(letra) + 32)
            letras += nueva_letra

        else:
            letras += letra

    return letras

stopwords = ([
    'a','all','also','although','among','an','and','are','around','as','at', 'be','been','being','both','but','by',
    'can','continue','despite','do','due', 'each', 'for','from','general', 'has','have','however',
    'in','include','including','into','is','it','its','like', 'many','more','most','must', 'nevertheless',
    'of','on','one','or','other','overall', 'plays','provide','provides', 'remains', 'such',
    'that','the','their','these','this','to','under','various','well','with'
])

def es_valido(caracter):
  c = ord(caracter)
    # minusculas y mayusculas
  if (c >= 97 and c <= 122) or (c >= 65 and c <= 90):
    return True
    #
  if ((c == 225) or (c == 233) or (c == 327) or (c == 243) or (c == 250) or (c == 241)):#unicode
    return True
    # minusculas con acento
  if c == 160 or c == 130 or c == 161 or c == 162 or c == 163:
    return True
    # mayusculas con acento
  if c == 181 or c == 144 or c == 214 or c == 224 or c == 233:
    return True
  else:
    return False

  
def tokenizador(texto, stopwords, valido):
    token = ""
    tokens = [None] * len(texto)
    j = 0

    for i in range(len(texto)):
        if valido(texto[i]):
            token += texto[i]
        else:
            if token != "":
                if token not in stopwords:
                    tokens[j] = token
                    j += 1
                token = ''

    return tokens[0:j]

def longitud(palabra):
    cantidad_caracteres = 0
    for caracter in palabra:
        cantidad_caracteres = cantidad_caracteres + 1
    return cantidad_caracteres


def termina_en(palabra, sufijo):
    long_p = longitud(palabra)
    long_s = longitud(sufijo)
    if long_s > long_p:
        return False
    return palabra[long_p - long_s:] == sufijo

def lematizador(tokens):
    texto_lematizado = [None] * len(tokens)
    j = 0 

    for palabra in tokens:
        palabra_lematizada = None
                
        if palabra in exceptions: 
            palabra_lematizada = exceptions[palabra]
                    
        if palabra_lematizada is None:
            for sufijo, terminacion in verbs_rules:
                if (longitud(palabra) - longitud(sufijo) >= 2) and termina_en(palabra, sufijo):
                    palabra_lematizada = palabra[:-longitud(sufijo)] + terminacion 
                    break
                        
        if palabra_lematizada is None:
            for sufijo, reemplazo in reglas_morfologicas:
                min_raiz = 3 if sufijo in ('es', 's') else 2 
                if longitud(palabra) - longitud(sufijo) >= min_raiz and termina_en(palabra, sufijo):
                    palabra_lematizada = palabra[:-longitud(sufijo)] + reemplazo
                    break

        if palabra_lematizada is None: 
            palabra_lematizada = palabra
                
        texto_lematizado[j] = palabra_lematizada
        j = j + 1
    
    return texto_lematizado[:j]

#generar pdfs
def generar_pdf_documentos(nombre_pdf, documentos):
    estilos = getSampleStyleSheet()
    elementos = []

    for nombre, texto in documentos.items():
        titulo = Paragraph(f"<b>{nombre}</b>", estilos['Normal'])
        contenido = Paragraph(str(texto).replace("\n","<br/>"), estilos['Normal'])
        bloque = [titulo, Spacer(1,10), contenido, Spacer(1,30)]
        elementos += bloque

    pdf = SimpleDocTemplate(nombre_pdf)
    pdf.build(elementos)

#interfaz
st.set_page_config(page_title="Detector de Plagio")

st.title("Detector de Plagio de Documentos PDF")
st.write("Sube tres archivos PDF para procesarlos y tokenizarlos")

archivo_original = st.file_uploader("Sube archivo ORIGINAL", type="pdf")
archivo_plagio = st.file_uploader("Sube archivo PLAGIO", type="pdf")
archivo_no_plagio = st.file_uploader("Sube archivo NO PLAGIO", type="pdf")

#botones
if st.button("Procesar documentos"):
    if not archivo_original or not archivo_plagio or not archivo_no_plagio:
        st.warning("Debes subir los tres archivos")
    else:
        st.info("Procesar")

        texto_original = to_minusculas(leer_pdf(archivo_original))
        texto_plagio = to_minusculas(leer_pdf(archivo_plagio))
        texto_no_plagio = to_minusculas(leer_pdf(archivo_no_plagio))

        tokens_original = tokenizador(texto_original, stopwords, es_valido)
        tokens_plagio = tokenizador(texto_plagio, stopwords, es_valido)
        tokens_no_plagio = tokenizador(texto_no_plagio, stopwords, es_valido)
        
        lematizador_original = lematizador(tokens_original)
        lematizador_plagio = lematizador(tokens_plagio)
        lematizador_no_plagio = lematizador(tokens_no_plagio)

        st.success("Procesamiento completo")

        st.write(f"Cantidad de palabras PDF original: {contador_palabras(texto_original)}")
        st.write(f"Cantidad de palabras PDF plagiado: {contador_palabras(texto_plagio)}")
        st.write(f"Cantidad de palabras PDF no plagiado: {contador_palabras(texto_no_plagio)}")

        #generar pdfs
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
            generar_pdf_documentos(tmp1.name, {"Original Preprocesado": (lematizador_original)})
            st.download_button("Descargar PDF Original Preprocesado", open(tmp1.name, "rb"), file_name="original_preprocesado.pdf")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
            generar_pdf_documentos(tmp2.name, {"Plagio Preprocesado":(lematizador_plagio)})
            st.download_button("Descargar PDF Plagio Preprocesado", open(tmp2.name, "rb"), file_name="plagio_preprocesado.pdf")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp3:

            generar_pdf_documentos(tmp3.name, {"No Plagio Tokenizado": (tokens_no_plagio)})
            st.download_button("Descargar PDF No Plagio Tokenizado", open(tmp3.name, "rb"), file_name="no_plagio_tokenizado.pdf")
            generar_pdf_documentos(tmp3.name, {"No Plagio Preprocesado": (lematizador_no_plagio)})
            st.download_button("Descargar PDF No Plagio Preprocesado", open(tmp3.name, "rb"), file_name="no_plagio_preprocesado.pdf")

st.markdown("---")
st.caption("Proyecto PLN")