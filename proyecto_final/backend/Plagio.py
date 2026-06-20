import json
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
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

# JACCARD

def ngramas_palabras(tokens, n=3):
    lista = []

    if len(tokens) < n:
        return lista

    for i in range(len(tokens) - n + 1):
        ng = " ".join(tokens[i:i+n])
        lista.append(ng)

    return lista


def similitud_jaccard(tokens1, tokens2, n=3):
    ngramas1 = set(ngramas_palabras(tokens1, n))
    ngramas2 = set(ngramas_palabras(tokens2, n))

    if len(ngramas1) == 0 and len(ngramas2) == 0:
        return 0.0

    interseccion = ngramas1 & ngramas2
    union = ngramas1 | ngramas2

    if len(union) == 0:
        return 0.0

    return len(interseccion) / len(union)

# SIMILITUD COSENO

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
        corpus_tokenizado.append(doc["tokens_lematizados"])

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

# TF-IDF PARA FRAGMENTOS

def construir_tfidf_fragmentos(textos_fragmentos_corpus, textos_fragmentos_usuario):
    corpus_total = textos_fragmentos_corpus + textos_fragmentos_usuario

    vectorizador = TfidfVectorizer()
    matriz_tfidf = vectorizador.fit_transform(corpus_total)

    total_corpus = len(textos_fragmentos_corpus)

    matriz_corpus = matriz_tfidf[:total_corpus]
    matriz_usuario = matriz_tfidf[total_corpus:]

    return matriz_corpus, matriz_usuario

# BERT EN CPU

RUTA_MODELO_BERT = "bert-base-uncased"


def cargar_modelo_bert(nombre_modelo=RUTA_MODELO_BERT):
    tokenizer = BertTokenizer.from_pretrained(nombre_modelo)
    modelo = BertModel.from_pretrained(nombre_modelo)
    modelo.to("cpu")
    modelo.eval()
    return tokenizer, modelo


def obtener_vectores_bert(textos, tokenizer, modelo, batch_size=8, max_length=128):
    vectores = []

    with torch.no_grad():
        for inicio in range(0, len(textos), batch_size):
            lote = textos[inicio:inicio + batch_size]

            entrada = tokenizer(
                lote,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            entrada = {
                clave: valor.to("cpu")
                for clave, valor in entrada.items()
            }

            salida = modelo(**entrada)

            # Promedio de los vectores de tokens usando attention_mask.
            token_embeddings = salida.last_hidden_state
            mascara = entrada["attention_mask"].unsqueeze(-1).float()

            suma_vectores = torch.sum(token_embeddings * mascara, dim=1)
            suma_mascara = torch.clamp(mascara.sum(dim=1), min=1e-9)
            promedio = suma_vectores / suma_mascara

            vectores.append(promedio.cpu().numpy())

    if len(vectores) == 0:
        return np.array([])

    return np.vstack(vectores)


# CLASIFICADOR TEMÁTICO OPCIONAL

try:
    from backend import Clasificador
except Exception:
    try:
        import Clasificador
    except Exception:
        Clasificador = None


def clasificar_tema(tokens):
    if Clasificador is None:
        return "SIN_TEMA"

    X = Clasificador.matriz_bag_words([tokens], Clasificador.vector_unico)
    tema = Clasificador.modelo_tema.predict(X)[0]
    return tema

# DICCIONARIOS DE CITAS Y EXCEPCIONES

PALABRAS_CITA = [
    "according", "cited", "citation", "reference", "references",
    "author", "authors", "researcher", "researchers",
    "study", "paper", "article", "journal", "source",
    "reported", "stated", "mentioned", "argued", "suggested",
    "published", "concluded", "indicated"
]

FRASES_CITA = [
    "according author",
    "according study",
    "previous research",
    "research suggest",
    "study show",
    "study indicate",
    "authors argue",
    "authors state",
    "paper present",
    "article mention",
    "source report",
    "results study"
]

FRASES_COMUNES_MENOR_PESO = [
    "introduction",
    "in conclusion",
    "this paper",
    "this study",
    "this work",
    "results show",
    "results indicate",
    "main objective",
    "research objective",
    "data analysis",
    "future work"
]


def calcular_ajuste_citas(tokens):
    texto = " ".join(tokens).lower()
    motivos = []
    factor = 1.0

    for palabra in PALABRAS_CITA:
        if palabra in tokens:
            motivos.append("palabra_cita:" + palabra)
            factor = factor * 0.75
            break

    for frase in FRASES_CITA:
        if frase in texto:
            motivos.append("frase_cita:" + frase)
            factor = factor * 0.70
            break

    for frase in FRASES_COMUNES_MENOR_PESO:
        if frase in texto:
            motivos.append("frase_comun:" + frase)
            factor = factor * 0.85
            break

    return factor, motivos


# FRAGMENTACIÓN

def aplanar_tokens(tokens):
    if len(tokens) == 0:
        return []

    if isinstance(tokens[0], list):
        todos = []
        for fragmento in tokens:
            for token in fragmento:
                todos.append(token)
        return todos

    return tokens


def crear_fragmento(tokens, numero, inicio):
    return {
        "numero_fragmento": numero,
        "inicio_token": inicio,
        "fin_token": inicio + len(tokens) - 1,
        "tokens": tokens,
        "texto": " ".join(tokens)
    }


def fragmentar_tokens(tokens, tamano_fragmento=120, min_tokens=20):
    fragmentos = []

    if len(tokens) == 0:
        return fragmentos

    # Si viene como lista de listas, se respeta como párrafos ya tokenizados.
    if isinstance(tokens[0], list):
        numero = 1
        inicio = 0

        for frag in tokens:
            if len(frag) >= min_tokens:
                fragmentos.append(crear_fragmento(frag, numero, inicio))
                numero += 1
            inicio += len(frag)

        if len(fragmentos) == 0:
            todos = aplanar_tokens(tokens)
            fragmentos.append(crear_fragmento(todos, 1, 0))

        return fragmentos

    numero = 1

    for inicio in range(0, len(tokens), tamano_fragmento):
        frag = tokens[inicio:inicio + tamano_fragmento]

        if len(frag) >= min_tokens:
            fragmentos.append(crear_fragmento(frag, numero, inicio))
            numero += 1

    if len(fragmentos) == 0:
        fragmentos.append(crear_fragmento(tokens, 1, 0))

    return fragmentos


# SELECCIÓN DE CANDIDATOS

def seleccionar_candidatos(
    documentos,
    corpus_tokenizado,
    tokens_usuario,
    top_n_candidatos=100,
    top_n_preliminar=300,
    usar_tema=True
):
    matriz_tfidf = construir_tfidf(corpus_tokenizado, tokens_usuario)

    indice_usuario = matriz_tfidf.shape[0] - 1
    vector_usuario = matriz_tfidf[indice_usuario]
    matriz_corpus = matriz_tfidf[:indice_usuario]

    similitudes = cosine_similarity(vector_usuario, matriz_corpus)[0]

    total_docs = len(documentos)
    top_n_preliminar = min(max(top_n_preliminar, top_n_candidatos), total_docs)
    top_n_candidatos = min(top_n_candidatos, total_docs)

    indices_preliminares = np.argsort(similitudes)[::-1][:top_n_preliminar]

    tema_usuario = "SIN_TEMA"
    if usar_tema:
        tema_usuario = clasificar_tema(tokens_usuario)

    candidatos = []

    for idx in indices_preliminares:
        tema_doc = "SIN_TEMA"
        bono_tema = 0.0

        if usar_tema:
            tema_doc = clasificar_tema(corpus_tokenizado[idx])
            if tema_doc == tema_usuario and tema_doc != "SIN_TEMA":
                bono_tema = 0.10

        score_candidato = (float(similitudes[idx]) * 0.90) + bono_tema

        candidatos.append({
            "indice": int(idx),
            "titulo": documentos[idx].get("titulo", "SIN_TITULO"),
            "fuente": documentos[idx].get("fuente", "SIN_FUENTE"),
            "score_tfidf_documento": float(similitudes[idx]),
            "tema_documento": tema_doc,
            "score_candidato": score_candidato
        })

    candidatos_ordenados = sorted(
        candidatos,
        key=lambda x: x["score_candidato"],
        reverse=True
    )

    return candidatos_ordenados[:top_n_candidatos], tema_usuario

# SCORE FINAL

def calcular_score_final(score_ngramas, score_jaccard, score_tfidf, score_bert, factor_citas):
    score_literal = (score_ngramas * 0.50) + (score_jaccard * 0.50)

    score_base = (
        (score_literal * 0.35) +
        (score_tfidf * 0.30) +
        (score_bert * 0.35)
    )

    score_final = score_base * factor_citas

    return score_literal, score_base, score_final

# PIPELINE PRINCIPAL

def pipeline_plagio_hibrido(
    tokens_usuario,
    ruta_corpus="corpus/corpus_procesado.json",
    top_n_candidatos=100,
    top_n_preliminar=300,
    tamano_fragmento=120,
    top_fragmentos=3,
    usar_tema=True
):
    documentos, corpus_tokenizado = cargar_datos(ruta_corpus)

    tokens_usuario_doc = aplanar_tokens(tokens_usuario)

    candidatos, tema_usuario = seleccionar_candidatos(
        documentos=documentos,
        corpus_tokenizado=corpus_tokenizado,
        tokens_usuario=tokens_usuario_doc,
        top_n_candidatos=top_n_candidatos,
        top_n_preliminar=top_n_preliminar,
        usar_tema=usar_tema
    )

    fragmentos_usuario = fragmentar_tokens(
        tokens_usuario,
        tamano_fragmento=tamano_fragmento
    )

    fragmentos_corpus = []

    for candidato in candidatos:
        idx = candidato["indice"]
        tokens_doc = corpus_tokenizado[idx]

        fragmentos_doc = fragmentar_tokens(
            tokens_doc,
            tamano_fragmento=tamano_fragmento
        )

        for frag in fragmentos_doc:
            frag["indice_documento"] = idx
            frag["titulo"] = documentos[idx].get("titulo", "SIN_TITULO")
            frag["fuente"] = documentos[idx].get("fuente", "SIN_FUENTE")
            frag["score_tfidf_documento"] = candidato["score_tfidf_documento"]
            frag["tema_documento"] = candidato["tema_documento"]
            fragmentos_corpus.append(frag)

    textos_usuario = []
    for frag in fragmentos_usuario:
        textos_usuario.append(frag["texto"])

    textos_corpus = []
    for frag in fragmentos_corpus:
        textos_corpus.append(frag["texto"])

    matriz_corpus_frag, matriz_usuario_frag = construir_tfidf_fragmentos(
        textos_corpus,
        textos_usuario
    )

    matriz_sim_tfidf = cosine_similarity(
        matriz_usuario_frag,
        matriz_corpus_frag
    )

    tokenizer, modelo_bert = cargar_modelo_bert()

    vectores_usuario = obtener_vectores_bert(
        textos_usuario,
        tokenizer,
        modelo_bert
    )

    vectores_corpus = obtener_vectores_bert(
        textos_corpus,
        tokenizer,
        modelo_bert
    )

    matriz_sim_bert = cosine_similarity(
        vectores_usuario,
        vectores_corpus
    )

    perfiles_usuario = []
    for frag in fragmentos_usuario:
        perfiles_usuario.append(perfil_ngramas(frag["texto"], n=3))

    perfiles_corpus = []
    for frag in fragmentos_corpus:
        perfiles_corpus.append(perfil_ngramas(frag["texto"], n=3))

    resultados_fragmentos = []

    for i in range(len(fragmentos_usuario)):
        frag_usuario = fragmentos_usuario[i]

        factor_citas, motivos_citas = calcular_ajuste_citas(
            frag_usuario["tokens"]
        )

        for j in range(len(fragmentos_corpus)):
            frag_corpus = fragmentos_corpus[j]

            score_ngramas = similitud_ngramas(
                perfiles_usuario[i],
                perfiles_corpus[j]
            )

            score_jaccard = similitud_jaccard(
                frag_usuario["tokens"],
                frag_corpus["tokens"],
                n=3
            )

            score_tfidf = float(matriz_sim_tfidf[i, j])
            score_bert = float(matriz_sim_bert[i, j])

            score_literal, score_base, score_final = calcular_score_final(
                score_ngramas=score_ngramas,
                score_jaccard=score_jaccard,
                score_tfidf=score_tfidf,
                score_bert=score_bert,
                factor_citas=factor_citas
            )

            resultados_fragmentos.append({
                "numero_fragmento_usuario": frag_usuario["numero_fragmento"],
                "inicio_token_usuario": frag_usuario["inicio_token"],
                "fin_token_usuario": frag_usuario["fin_token"],
                "fragmento_usuario": frag_usuario["texto"],

                "titulo_fuente": frag_corpus["titulo"],
                "fuente": frag_corpus["fuente"],
                "numero_fragmento_fuente": frag_corpus["numero_fragmento"],
                "inicio_token_fuente": frag_corpus["inicio_token"],
                "fin_token_fuente": frag_corpus["fin_token"],
                "fragmento_fuente": frag_corpus["texto"],

                "score_literal": {
                    "ngramas": round(score_ngramas * 100, 2),
                    "jaccard": round(score_jaccard * 100, 2),
                    "promedio_literal": round(score_literal * 100, 2)
                },

                "score_lexico": {
                    "tfidf": round(score_tfidf * 100, 2)
                },

                "score_semantico": {
                    "bert": round(score_bert * 100, 2)
                },

                "ajuste_citas": {
                    "factor": round(factor_citas, 3),
                    "motivos": motivos_citas
                },

                "score_base_sin_ajuste": round(score_base * 100, 2),
                "score_final": round(score_final * 100, 2),
                "tema_usuario": tema_usuario,
                "tema_documento": frag_corpus["tema_documento"]
            })

    resultados_ordenados = sorted(
        resultados_fragmentos,
        key=lambda x: x["score_final"],
        reverse=True
    )

    mejores_por_fragmento_usuario = []

    for frag_usuario in fragmentos_usuario:
        numero = frag_usuario["numero_fragmento"]
        mejores = []

        for resultado in resultados_fragmentos:
            if resultado["numero_fragmento_usuario"] == numero:
                mejores.append(resultado)

        if len(mejores) > 0:
            mejor = sorted(
                mejores,
                key=lambda x: x["score_final"],
                reverse=True
            )[0]
            mejores_por_fragmento_usuario.append(mejor)

    if len(mejores_por_fragmento_usuario) > 0:
        suma_final = 0
        suma_literal = 0
        suma_tfidf = 0
        suma_bert = 0

        for r in mejores_por_fragmento_usuario:
            suma_final += r["score_final"]
            suma_literal += r["score_literal"]["promedio_literal"]
            suma_tfidf += r["score_lexico"]["tfidf"]
            suma_bert += r["score_semantico"]["bert"]

        probabilidad_plagio = suma_final / len(mejores_por_fragmento_usuario)
        promedio_literal = suma_literal / len(mejores_por_fragmento_usuario)
        promedio_tfidf = suma_tfidf / len(mejores_por_fragmento_usuario)
        promedio_bert = suma_bert / len(mejores_por_fragmento_usuario)
    else:
        probabilidad_plagio = 0.0
        promedio_literal = 0.0
        promedio_tfidf = 0.0
        promedio_bert = 0.0

    salida = {
        "probabilidad_plagio": round(probabilidad_plagio, 2),

        "score_general": {
            "score_literal_promedio": round(promedio_literal, 2),
            "score_lexico_tfidf_promedio": round(promedio_tfidf, 2),
            "score_semantico_bert_promedio": round(promedio_bert, 2),
            "score_final_ajustado": round(probabilidad_plagio, 2)
        },

        "tema_usuario": tema_usuario,
        "documentos_candidatos_usados": len(candidatos),
        "fragmentos_usuario_analizados": len(fragmentos_usuario),
        "fragmentos_corpus_comparados": len(fragmentos_corpus),

        "top_3_fragmentos": resultados_ordenados[:top_fragmentos]
    }

    return salida


if __name__ == "__main__":
    tokens = [
        "artificial", "intelligence", "algorithm", "classify", "data",
        "model", "learn", "pattern", "information", "system",
        "machine", "learning", "process", "large", "dataset"
    ]

    resultado = pipeline_plagio_hibrido(
        tokens_usuario=tokens,
        ruta_corpus="corpus/corpus_procesado.json",
        top_n_candidatos=100,
        top_n_preliminar=300,
        tamano_fragmento=120,
        top_fragmentos=3,
        usar_tema=True
    )

    print(json.dumps(resultado, indent=4, ensure_ascii=False))
