import json
import Preprocesamiento

# CARGA DE CORPUS
def cargar_corpus(ruta_corpus):

    with open(ruta_corpus, "r", encoding="utf-8") as archivo:

        corpus = json.load(archivo)

    return corpus

# PREPROCESAR TEXTO
def preprocess_texto(texto):

    texto_min = Preprocesamiento.to_minusculas(texto)

    tokens = Preprocesamiento.tokenizador(texto_min)

    tokens_lematizados = Preprocesamiento.lematizador(tokens)

    texto_limpio = " ".join(tokens_lematizados)

    return {

        "texto_limpio": texto_limpio,

        "tokens_lematizados": tokens_lematizados
    }

# PROCESAR DOCUMENTO DEL CORPUS
def procesar_documento(documento):

    texto_original = documento["texto"]

    resultado = preprocess_texto(texto_original)

    nuevo_documento = {

        "fuente": documento["fuente"],

        "titulo": documento["titulo"],

        "texto_original": texto_original,

        "texto_limpio": resultado["texto_limpio"],

        "tokens_lematizados": resultado["tokens_lematizados"]
    }

    return nuevo_documento

# PREPROCESAR CORPUS
def procesar_corpus(corpus):

    corpus_procesado = []

    for documento in corpus:

        try:

            nuevo_doc = procesar_documento(documento)

            corpus_procesado.append(nuevo_doc)

        except Exception:
            continue

    return corpus_procesado

# GUARDAR CORPUS
def guardar_corpus(corpus_procesado, ruta_salida):

    with open(ruta_salida, "w", encoding="utf-8") as archivo:

        json.dump(
            corpus_procesado,
            archivo,
            ensure_ascii=False,
            indent=4
        )

# PIPELINE PARA EL PREPROCESAMIENTO DEL CORPUS
def pipeline_preproceso_corpus():

    ruta_corpus = "../corpus/corpus_total.json"

    ruta_salida = "../corpus/corpus_procesado.json"

    corpus = cargar_corpus(ruta_corpus)

    corpus_procesado = procesar_corpus(corpus)

    guardar_corpus(corpus_procesado, ruta_salida)
    
pipeline_preproceso_corpus()