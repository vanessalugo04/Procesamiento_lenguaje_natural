import numpy as np
import json
import os
import sys

# ============================================================
# SCRIPT DE ENTRENAMIENTO DEL DETECTOR DE IA
# ============================================================
# Usa el dataset HC3 (Human ChatGPT Comparison Corpus) para
# entrenar la Regresión Logística con features estilométricas
# + embeddings BERT.
#
# Ejecución:
#   python -m backend.Entrenar_detector_ia
#
# Solo se necesita ejecutar UNA VEZ. Genera el archivo:
#   corpus/modelo_ia.npz
# ============================================================

from backend.Estilometria import (
    extraer_features_estilometricas,
    TOTAL_FEATURES_ESTILOMETRICAS,
    NOMBRES_FEATURES
)
from backend.DetectorIA import (
    RegresionLogisticaDesdeCero,
    extraer_embedding_bert,
    comprimir_embedding_bert,
    DIM_BERT_COMPRIMIDO,
    RUTA_MODELO_IA
)
from backend.Preprocesamiento import tokenizador, to_minusculas, lematizador

import re

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# Cuántas muestras tomar de HC3 por categoría
MUESTRAS_POR_CATEGORIA = 200  # Total: 200 × 5 categorías × 2 clases = ~2000

# Longitud mínima de texto para que sea útil
MIN_LONGITUD_TEXTO = 100  # caracteres

# Split de entrenamiento/prueba
PORCENTAJE_TEST = 0.2

# Parámetros de la Regresión Logística
LEARNING_RATE = 0.05
EPOCHS = 1000


# ============================================================
# 2. DESCARGAR Y PROCESAR HC3
# ============================================================

def descargar_hc3():
    """
    Descarga el dataset HC3 desde HuggingFace y extrae pares
    de textos humano/ChatGPT.
    Usa la versión parquet convertida automáticamente por HuggingFace.
    """
    from datasets import load_dataset

    print("=" * 60)
    print("DESCARGANDO DATASET HC3")
    print("=" * 60)

    # Cargar desde los archivos parquet auto-convertidos por HuggingFace
    # Cada categoría tiene su propio subdirectorio de parquet
    categorias = ["finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]

    textos = []  # Lista de (texto, etiqueta, categoria)

    for categoria in categorias:
        print(f"\n[HC3] Descargando categoría: {categoria}")

        try:
            # Cargar directamente desde archivos parquet del repositorio
            ruta_parquet = f"hf://datasets/Hello-SimpleAI/HC3@refs/convert/parquet/{categoria}/train/*.parquet"
            dataset = load_dataset("parquet", data_files={"train": ruta_parquet}, split="train")
        except Exception as e:
            print(f"  [ERROR] No se pudo descargar {categoria}: {e}")
            continue

        conteo_humano = 0
        conteo_ia = 0

        for item in dataset:
            # Extraer respuestas humanas
            if conteo_humano < MUESTRAS_POR_CATEGORIA:
                for respuesta in item.get("human_answers", []):
                    if len(respuesta.strip()) >= MIN_LONGITUD_TEXTO and conteo_humano < MUESTRAS_POR_CATEGORIA:
                        textos.append((respuesta.strip(), 0, categoria))  # 0 = humano
                        conteo_humano += 1

            # Extraer respuestas de ChatGPT
            if conteo_ia < MUESTRAS_POR_CATEGORIA:
                for respuesta in item.get("chatgpt_answers", []):
                    if len(respuesta.strip()) >= MIN_LONGITUD_TEXTO and conteo_ia < MUESTRAS_POR_CATEGORIA:
                        textos.append((respuesta.strip(), 1, categoria))  # 1 = IA
                        conteo_ia += 1

            # Si ya tenemos suficientes de ambas clases, siguiente categoría
            if conteo_humano >= MUESTRAS_POR_CATEGORIA and conteo_ia >= MUESTRAS_POR_CATEGORIA:
                break

        print(f"  -> Humanos: {conteo_humano} | ChatGPT: {conteo_ia}")

    print(f"\n[HC3] Total de textos recolectados: {len(textos)}")
    return textos


# ============================================================
# 3. EXTRAER FEATURES
# ============================================================

def preprocesar_texto_para_tokens(texto):
    """
    Aplica el mismo preprocesamiento que el pipeline existente
    para obtener tokens lematizados.
    """
    texto_min = to_minusculas(texto)
    tokens = tokenizador(texto_min)
    tokens_lem = lematizador(tokens)

    # Filtro regex (igual que en Preprocesamiento.py)
    patron = re.compile(r'^[a-zA-ZáéíóúüñÑ]{3,}$')
    tokens_filtrados = [t for t in tokens_lem if patron.match(t)]

    return tokens_filtrados


def extraer_features_dataset(textos):
    """
    Extrae features estilométricas + BERT para cada texto del dataset.
    """
    print("\n" + "=" * 60)
    print("EXTRAYENDO FEATURES")
    print("=" * 60)

    total = len(textos)
    X = []
    y = []
    errores = 0

    for i, (texto, etiqueta, categoria) in enumerate(textos):
        try:
            # Tokenizar
            tokens = preprocesar_texto_para_tokens(texto)

            if len(tokens) < 10:
                errores += 1
                continue

            # Features estilometricas (15)
            feat_estilo = extraer_features_estilometricas(texto, tokens)

            # Embedding BERT (768) -> comprimido a 7 estadisticos
            embedding_crudo = extraer_embedding_bert(texto)
            feat_bert = comprimir_embedding_bert(embedding_crudo)

            # Concatenar
            vector = np.concatenate([feat_estilo, feat_bert])

            X.append(vector)
            y.append(etiqueta)

            if (i + 1) % 50 == 0:
                print(f"  Procesados: {i + 1}/{total} ({((i+1)/total)*100:.1f}%)")

        except Exception as e:
            errores += 1
            if errores <= 5:
                print(f"  [ERROR] Texto #{i}: {str(e)[:80]}")

    print(f"\n  Textos procesados correctamente: {len(X)}")
    print(f"  Textos con error/descartados: {errores}")

    return np.array(X), np.array(y)


# ============================================================
# 4. DIVIDIR TRAIN/TEST
# ============================================================

def dividir_train_test(X, y, porcentaje_test=0.2, semilla=42):
    """
    Divide los datos en entrenamiento y prueba de forma aleatoria
    pero reproducible.
    """
    np.random.seed(semilla)
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_test = int(n * porcentaje_test)
    indices_test = indices[:n_test]
    indices_train = indices[n_test:]

    X_train = X[indices_train]
    y_train = y[indices_train]
    X_test = X[indices_test]
    y_test = y[indices_test]

    return X_train, y_train, X_test, y_test


# ============================================================
# 5. MÉTRICAS DE EVALUACIÓN
# ============================================================

def calcular_metricas(y_real, y_pred):
    """Calcula Accuracy, Precision, Recall, F1-Score."""
    tp = 0  # True Positives (IA correctamente identificada)
    tn = 0  # True Negatives (Humano correctamente identificado)
    fp = 0  # False Positives (Humano clasificado como IA)
    fn = 0  # False Negatives (IA clasificado como Humano)

    for i in range(len(y_real)):
        if y_real[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_real[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_real[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_real[i] == 1 and y_pred[i] == 0:
            fn += 1

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


# ============================================================
# 6. ENTRENAMIENTO PRINCIPAL
# ============================================================

def entrenar():
    """Pipeline completo de entrenamiento."""

    print("=" * 60)
    print("ENTRENAMIENTO DEL DETECTOR DE IA")
    print("=" * 60)
    print(f"Features estilométricas: {TOTAL_FEATURES_ESTILOMETRICAS}")
    print(f"Features BERT comprimido:{DIM_BERT_COMPRIMIDO}")
    print(f"Total features:          {TOTAL_FEATURES_ESTILOMETRICAS + DIM_BERT_COMPRIMIDO}")
    print(f"Learning rate:           {LEARNING_RATE}")
    print(f"Epochs:                  {EPOCHS}")

    # Paso 1: Descargar HC3
    textos = descargar_hc3()

    if len(textos) < 100:
        print("[ERROR] Dataset muy pequeño. Abortando.")
        return

    # Paso 2: Extraer features
    X, y = extraer_features_dataset(textos)

    # Verificar balance de clases
    n_humano = int(np.sum(y == 0))
    n_ia = int(np.sum(y == 1))
    print(f"\n  Balance de clases:")
    print(f"    Humano (0): {n_humano}")
    print(f"    IA     (1): {n_ia}")

    # Paso 3: Dividir train/test
    X_train, y_train, X_test, y_test = dividir_train_test(X, y, PORCENTAJE_TEST)

    print(f"\n  Train: {len(y_train)} muestras")
    print(f"  Test:  {len(y_test)} muestras")

    # Paso 4: Entrenar
    print("\n" + "=" * 60)
    print("ENTRENANDO REGRESION LOGISTICA")
    print("=" * 60)

    modelo = RegresionLogisticaDesdeCero(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
    modelo.fit(X_train, y_train)

    # Paso 5: Evaluar en test
    print("\n" + "=" * 60)
    print("EVALUACIÓN EN DATOS DE TEST")
    print("=" * 60)

    y_pred = modelo.predict(X_test)
    metricas = calcular_metricas(y_test, y_pred)

    print(f"\n  Accuracy:  {metricas['accuracy']}%")
    print(f"  Precision: {metricas['precision']}%")
    print(f"  Recall:    {metricas['recall']}%")
    print(f"  F1-Score:  {metricas['f1_score']}%")
    print(f"\n  Matriz de confusion:")
    print(f"    TP (IA->IA):       {metricas['tp']}")
    print(f"    TN (Humano->Hum):  {metricas['tn']}")
    print(f"    FP (Humano->IA):   {metricas['fp']}")
    print(f"    FN (IA->Humano):   {metricas['fn']}")

    # Paso 6: Guardar modelo
    print("\n" + "=" * 60)
    print("GUARDANDO MODELO")
    print("=" * 60)

    modelo.guardar(RUTA_MODELO_IA)

    # Paso 7: Importancia de features estilométricas
    print("\n" + "=" * 60)
    print("IMPORTANCIA DE FEATURES ESTILOMETRICAS (top pesos)")
    print("=" * 60)

    pesos_estilo = modelo.pesos[:TOTAL_FEATURES_ESTILOMETRICAS]
    indices_ordenados = np.argsort(np.abs(pesos_estilo))[::-1]

    for rank, idx in enumerate(indices_ordenados):
        peso = pesos_estilo[idx]
        nombre = NOMBRES_FEATURES[idx]
        direccion = "-> indica IA" if peso > 0 else "-> indica Humano"
        print(f"  {rank + 1}. {nombre}: {peso:.4f} ({direccion})")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO!")
    print("=" * 60)
    print(f"Modelo guardado en: {RUTA_MODELO_IA}")
    print("El detector de IA está listo para usar en la aplicación.")

    return metricas


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":
    entrenar()
