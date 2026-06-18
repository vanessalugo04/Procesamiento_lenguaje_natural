from pypdf import PdfReader

import Stopwords_Copy as Stopwords
import Reglas_Lematizador_Copy as Reglas_Lematizador

# CARGA DEL DOCUMENTO PDF

def extraer_texto_pdf(ruta_archivo):
    texto_completo = ""

    try:
        reader = PdfReader(ruta_archivo)

        for pagina in reader.pages:
            texto_extraido = pagina.extract_text()

            if texto_extraido:
                texto_completo += texto_extraido + "\n"

        return texto_completo

    except FileNotFoundError:
        return f"Error: No se encontró el archivo '{ruta_archivo}'"

    except Exception as e:
        return f"Ocurrió un error al leer el PDF: {e}"

# CONTADORES

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


def contar_lista(lista):
    contador = 0

    for elemento in lista:
        contador += 1

    return contador


def longitud(palabra):
    contador = 0

    for caracter in palabra:
        contador += 1

    return contador

# CONVERTIR A MINÚSCULAS

def to_minusculas(texto):
    letras = ""

    for letra in texto:
        codigo = ord(letra)

        if codigo >= 65 and codigo <= 90:
            nueva_letra = chr(codigo + 32)
            letras += nueva_letra
        else:
            letras += letra

    return letras

# VALIDACIÓN DE CARACTERES

def es_valido(c):
    codigo = ord(c)

    if (codigo >= 65 and codigo <= 90) or (codigo >= 97 and codigo <= 122):
        return True

    # Letras con acento y ñ
    caracteres_especiales = [
        225, 233, 237, 243, 250, 
        193, 201, 205, 211, 218,
        241, 209,                
        252, 220                
    ]

    if codigo in caracteres_especiales:
        return True

    return False


def es_palabra_valida(palabra):
    if palabra == "":
        return False

    if longitud(palabra) <= 2:
        return False

    for caracter in palabra:
        if not es_valido(caracter):
            return False

    return True

# STOPWORDS

def obtener_stopwords():
    stopwords_limpias = []

    for palabra in Stopwords.STOPWORDS:
        palabra_limpia = to_minusculas(str(palabra).strip())

        if palabra_limpia != "":
            stopwords_limpias.append(palabra_limpia)

    return set(stopwords_limpias)

# TOKENIZADOR

def tokenizador(texto):
    stopwords = obtener_stopwords()

    texto = to_minusculas(texto)

    token = ""
    tokens = []

    for caracter in texto:
        if es_valido(caracter):
            token += caracter
        else:
            if token != "":
                token = to_minusculas(token.strip())

                if token not in stopwords and es_palabra_valida(token):
                    tokens.append(token)

            token = ""

    # Guardar el último token si el texto no termina con signo o espacio
    if token != "":
        token = to_minusculas(token.strip())

        if token not in stopwords and es_palabra_valida(token):
            tokens.append(token)

    return tokens
    
# FUNCIONES PARA LEMATIZACIÓN

def termina_en(palabra, sufijo):
    lp = longitud(palabra)
    ls = longitud(sufijo)

    if lp < ls:
        return False

    return palabra[lp - ls:] == sufijo


def lematizar_palabra(palabra):
    palabra = to_minusculas(str(palabra).strip())

    lema = None

    # Revisar excepciones
    if palabra in Reglas_Lematizador.exceptions:
        lema = Reglas_Lematizador.exceptions[palabra]

    # Reglas verbales
    if lema is None:
        for sufijo, terminacion in Reglas_Lematizador.verbs_rules:
            if longitud(palabra) - longitud(sufijo) >= 2:
                if termina_en(palabra, sufijo):
                    lema = palabra[:-longitud(sufijo)] + terminacion
                    break

    # Reglas morfológicas
    if lema is None:
        for sufijo, reemplazo in Reglas_Lematizador.reglas_morfologicas:

            if sufijo == "es" or sufijo == "s":
                min_raiz = 3
            else:
                min_raiz = 2

            if longitud(palabra) - longitud(sufijo) >= min_raiz:
                if termina_en(palabra, sufijo):
                    lema = palabra[:-longitud(sufijo)] + reemplazo
                    break

    if lema is None:
        lema = palabra

    lema = to_minusculas(str(lema).strip())

    return lema

# LEMATIZADOR PARA LISTA DE TOKENS

def lematizador(tokens):
    stopwords = obtener_stopwords()

    resultado = []

    for palabra in tokens:
        lema = lematizar_palabra(palabra)

        # Segunda limpieza después de lematizar
        if lema not in stopwords and es_palabra_valida(lema):
            resultado.append(lema)

    return resultado

# FILTRO FINAL PARA LDA

def filtrar_tokens_lda(tokens):
    stopwords = obtener_stopwords()

    tokens_limpios = []

    for token in tokens:
        token = to_minusculas(str(token).strip())

        if token == "":
            continue

        if token in stopwords:
            continue

        if not es_palabra_valida(token):
            continue

        tokens_limpios.append(token)

    return tokens_limpios

# PIPELINE PARA TEXTO NORMAL

def pipeline_texto(texto):
    texto_min = to_minusculas(texto)

    tokens = tokenizador(texto_min)

    tokens_lematizados = lematizador(tokens)

    tokens_filtrados = filtrar_tokens_lda(tokens_lematizados)

    return {
        "texto_minusculas": texto_min,
        "tokens": tokens,
        "tokens_lematizados": tokens_lematizados,
        "tokens_lda": tokens_filtrados
    }

# PIPELINE PARA DOCUMENTO PDF

def pipeline_procesamiento(archivo):
    texto = extraer_texto_pdf(archivo)

    if texto.startswith("Error:") or texto.startswith("Ocurrió un error"):
        return {
            "error": texto,
            "exito": False
        }

    resultado = pipeline_texto(texto)

    return {
        "exito": True,
        "texto_minusculas": resultado["texto_minusculas"],
        "tokens": resultado["tokens"],
        "tokens_lematizados": resultado["tokens_lematizados"],
        "resultado_lematizado": resultado["tokens_lda"]
    }