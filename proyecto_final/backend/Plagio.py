import numpy as np
import json
from gensim.models import FastText
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# N-GRAMAS
def ngramas_caracteres(texto, n=3):
    texto = texto.lower().replace(" ", "")
    lista = []
    
    for i in range(len(texto) - n + 1):
        lista.append(texto[i:i+n])
    return lista


def perfil_ngramas(texto, n=3):
    perfil = {}
    ngramas = ngramas_caracteres(texto, n)

    for ng in ngramas:
        if ng in perfil:
            perfil[ng] += 1
        else:
            perfil[ng] = 1

    return perfil


def distancia(p1, p2):
    dist = 0
    todos = set(p1.keys()) | set(p2.keys())

    for k in todos:
        v1 = p1[k] if k in p1 else 0
        v2 = p2[k] if k in p2 else 0
        dist += abs(v1 - v2)

    return dist


def similitud_ngramas(perfil1, perfil2):
    dist = distancia(perfil1, perfil2)

    total = (sum(perfil1.values()) + sum(perfil2.values()))

    if total == 0:
        return 0.0

    similitud = 1 - (dist / total)

    if similitud < 0:
        similitud = 0

    return similitud


# FASTTEXT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO = os.path.join(BASE_DIR,"..","corpus","fasttext.model")

def entrenar_y_guardar_fasttext(corpus_tokenizado):
    modelo = FastText(
        sentences=corpus_tokenizado,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    modelo.save(RUTA_MODELO)


def cargar_modelo_fasttext():
    if not os.path.exists(RUTA_MODELO):
        raise Exception(
            "No existe fasttext.model. "
        )

    modelo = FastText.load(RUTA_MODELO)
    return modelo


def obtener_vector_fasttext(modelo, tokens):
    vectores = []
    
    for token in tokens:
        if token in modelo.wv:
            vectores.append(modelo.wv[token])

    if len(vectores) == 0:
        return np.zeros(100)

    return np.mean(vectores, axis=0)


def similitud_coseno_vectores(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# CARGAR CORPUS
def cargar_datos(ruta_corpus="corpus/corpus_procesado.json"):
    with open(ruta_corpus, "r", encoding="utf-8") as f:
        documentos = json.load(f)
    corpus_tokenizado = []

    for doc in documentos:
        corpus_tokenizado.append(
            doc["tokens_lematizados"]
        )

    return documentos, corpus_tokenizado


# TF-IDF
def construir_tfidf(corpus_tokenizado, tokens_usuario):

    corpus_texto = []

    for doc in corpus_tokenizado:
        corpus_texto.append(" ".join(doc))

    texto_usuario = " ".join(tokens_usuario)
    corpus_total = corpus_texto + [texto_usuario]
    vectorizador = TfidfVectorizer()
    matriz_tfidf = vectorizador.fit_transform(corpus_total)

    return matriz_tfidf

"""
# PIPELINE
def pipeline_plagio_hibrido(texto_usuario, ruta_corpus="../corpus/corpus_procesado.json"):

    #corpus
    documentos, corpus_tokenizado = cargar_datos(ruta_corpus)

    #preprocesamiento
    tokens_usuario = (Preprocesamiento.pipeline_procesamiento(texto_usuario))

    #TF-IDF
    matriz_tfidf = construir_tfidf(corpus_tokenizado,tokens_usuario)

    indice_usuario = matriz_tfidf.shape[0] - 1
    vector_usuario_tfidf = matriz_tfidf[indice_usuario]

    #trigramas
    perfil_usuario = perfil_ngramas(texto_usuario, n=3)

    #fasttext
    modelo_ft = cargar_modelo_fasttext()

    vector_usuario_ft = obtener_vector_fasttext(modelo_ft, tokens_usuario)

    resultados = []

    #comparacion
    for i, doc in enumerate(documentos):

        texto_doc = doc["texto_original"]
        tokens_doc = corpus_tokenizado[i]

        perfil_doc = perfil_ngramas(texto_doc, n=3)

        score_ngram = similitud_ngramas(perfil_usuario, perfil_doc)

        vector_doc_tfidf = matriz_tfidf[i]

        score_tfidf = cosine_similarity(vector_usuario_tfidf,vector_doc_tfidf)[0][0]

        vector_doc_ft = obtener_vector_fasttext(modelo_ft, tokens_doc)

        score_ft = similitud_coseno_vectores(vector_usuario_ft, vector_doc_ft)

        #score final
        score_final = ((score_ngram * 0.30) + (score_tfidf * 0.30) + (score_ft * 0.40))

        resultados.append({
            "titulo":
                doc["titulo"],

            "score_ngramas":
                round(score_ngram * 100, 2),

            "score_tfidf":
                round(score_tfidf * 100, 2),

            "score_fasttext":
                round(score_ft * 100, 2),

            "probabilidad_plagio":
                round(score_final * 100, 2)

        })


    resultados_ordenados = sorted(
        resultados,
        key=lambda x: x["probabilidad_plagio"],
        reverse=True
    )

    return resultados_ordenados[:5]"""

# Modificamos la función para que reciba directamente los tokens que ya limpió app.py
def pipeline_plagio_hibrido(tokens_usuario, ruta_corpus="corpus/corpus_procesado.json"):

    # corpus
    documentos, corpus_tokenizado = cargar_datos(ruta_corpus)

    # Convertimos los tokens a un texto unido para que la función de N-gramas funcione
    texto_usuario = " ".join(tokens_usuario)

    # TF-IDF
    matriz_tfidf = construir_tfidf(corpus_tokenizado, tokens_usuario)
    indice_usuario = matriz_tfidf.shape[0] - 1
    vector_usuario_tfidf = matriz_tfidf[indice_usuario]

    # trigramas
    perfil_usuario = perfil_ngramas(texto_usuario, n=3)

    # fasttext
    modelo_ft = cargar_modelo_fasttext()
    vector_usuario_ft = obtener_vector_fasttext(modelo_ft, tokens_usuario)

    resultados = []

    # comparacion
    for i, doc in enumerate(documentos):
        texto_doc = doc["texto_original"]
        tokens_doc = corpus_tokenizado[i]

        perfil_doc = perfil_ngramas(texto_doc, n=3)
        score_ngram = similitud_ngramas(perfil_usuario, perfil_doc)

        vector_doc_tfidf = matriz_tfidf[i]
        score_tfidf = cosine_similarity(vector_usuario_tfidf, vector_doc_tfidf)[0][0]

        vector_doc_ft = obtener_vector_fasttext(modelo_ft, tokens_doc)
        score_ft = similitud_coseno_vectores(vector_usuario_ft, vector_doc_ft)

        # score final
        score_final = ((score_ngram * 0.30) + (score_tfidf * 0.30) + (score_ft * 0.40))

        resultados.append({
            "titulo": doc["titulo"],
            "score_ngramas": round(score_ngram * 100, 2),
            "score_tfidf": round(score_tfidf * 100, 2),
            "score_fasttext": round(score_ft * 100, 2),
            "probabilidad_plagio": round(score_final * 100, 2)
        })

    resultados_ordenados = sorted(
        resultados,
        key=lambda x: x["probabilidad_plagio"],     
        reverse=True
    )

    return resultados_ordenados[:5]

# PREPARAR MODELO
def preparar_modelo():
    documentos, corpus_tokenizado = cargar_datos()
    entrenar_y_guardar_fasttext(corpus_tokenizado)


if __name__ == "__main__":
    preparar_modelo() #esto solo se ejecuta la primera vez
    texto = """
    el algoritmo de inteligencia artificial
    clasifica datos rapidamente
    """

    resultados = pipeline_plagio_hibrido(texto)
    for r in resultados:
        print("Titulo:", r["titulo"])
        print("Ngramas:", r["score_ngramas"])

        print("TFIDF:", r["score_tfidf"])

        print("FastText:", r["score_fasttext"])

        print("Plagio:", r["probabilidad_plagio"])