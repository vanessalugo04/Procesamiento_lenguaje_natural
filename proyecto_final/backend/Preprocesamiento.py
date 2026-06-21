from pypdf import PdfReader
import numpy as np

from backend import Stopwords
from backend import Reglas_Lematizador



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

# CONTADOR DE PALABRAS
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

# CONTADOR DE ELEMENTOS EN UNA LISTA
def contar_lista(lista):
    contador = 0
    for elemento in lista:
        contador += 1
    return contador

# CONTADOR DE CARACTERES EN UNA PALABRA
def longitud(palabra):
    return sum(1 for _ in palabra)

#CONVERTIR A MINUSCULA
def to_minusculas(texto):

    letras = ""

    for letra in texto:

        if ord(letra) >= 65 and ord(letra) <= 90:
            nueva_letra = chr(ord(letra) + 32)
            letras += nueva_letra

        else:
            letras += letra

    return letras

# VALIDACION DE CARACTERES
def es_valido(c):
    code = ord(c)
    if (code >= 97 and code <= 122) or (code >= 65 and code <= 90):
        return True
    if code in (225, 233, 237, 243, 250, 241, 209):
        return True
    if code in (160, 130, 161, 162, 163, 181, 144, 214, 224):
        return True
    return False

# TOKENIZADOR
def tokenizador(texto):
    token = ""
    tokens = []
    for ch in texto:
        if es_valido(ch):
            token += ch
        else:
            if token and token not in Stopwords.STOPWORDS:
                tokens.append(token)
            token = ""
    if token and token not in Stopwords.STOPWORDS:
        tokens.append(token)
    return tokens

# FUNCION SUFIJO
def termina_en(palabra, sufijo):
    lp, ls = longitud(palabra), longitud(sufijo)
    return lp >= ls and palabra[lp - ls:] == sufijo

# LEMATIZADOR PARA TEXTO TOKENIZADO
def lematizador(tokens):
    resultado = [None] * len(tokens)
    j = 0
    
    for palabra in tokens:
        lema = None
        
        if palabra in Reglas_Lematizador.exceptions:
            lema = Reglas_Lematizador.exceptions[palabra]
            
        if lema is None:
            for sufijo, terminacion in Reglas_Lematizador.verbs_rules:
                if (longitud(palabra) - longitud(sufijo) >= 2) and termina_en(palabra, sufijo):
                    lema = palabra[:-longitud(sufijo)] + terminacion
                    break
                    
        if lema is None:
            for sufijo, reemplazo in Reglas_Lematizador.reglas_morfologicas:
                min_raiz = 3 if sufijo in ('es', 's') else 2
                if longitud(palabra) - longitud(sufijo) >= min_raiz and termina_en(palabra, sufijo):
                    lema = palabra[:-longitud(sufijo)] + reemplazo
                    break
                    
        if lema is None:
            lema = palabra
            
        resultado[j] = lema
        j = j + 1
        
    return resultado

# PIPELINE PARA PRIMER LIMPIEZA DEL ARCHIVO DE USUARIO XD 
import re  


def pipeline_procesamiento(archivo):
    texto = extraer_texto_pdf(archivo)

    if texto.startswith("Error:") or texto.startswith("Ocurrió un error"):
        return {"error": texto, "exito": False}
        
    texto_min = to_minusculas(texto)
    tokens = tokenizador(texto_min)
    tokens_lematizados = lematizador(tokens)
    
    # -------------------------------------------------------------
    # SEGUNDA CAPA DE LIMPIEZA CON EXPRESIONES REGULARES (Añadir aquí)
    # -------------------------------------------------------------
    tokens_filtrados = []
    # Expresión regular: solo permite palabras con letras de 3 o más caracteres
    patron_letras = re.compile(r'^[a-zA-ZáéíóúüñÑ]{3,}$')
    
    for token in tokens_lematizados:
        # Si el token cumple con la expresión regular, se conserva
        if patron_letras.match(token):
            tokens_filtrados.append(token)
    
    return {
        "exito": True,
        "texto_crudo": texto,  # Texto original extraído del PDF (para el detector de IA)
        "resultado_lematizado": tokens_filtrados  # return de la lista refinada
    }