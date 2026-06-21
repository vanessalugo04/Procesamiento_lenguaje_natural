import numpy as np
import os
import json

from backend.Estilometria import (
    extraer_features_estilometricas,
    TOTAL_FEATURES_ESTILOMETRICAS,
    NOMBRES_FEATURES
)

# ============================================================
# MÓDULO DETECTOR DE IA
# ============================================================
# Combina features estilométricas (15) + embeddings BERT (768)
# para clasificar textos como humano o IA usando una
# Regresión Logística implementada desde cero.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO_IA = os.path.join(BASE_DIR, "..", "corpus", "modelo_ia.npz")

# Dimension del embedding comprimido de BERT
# En vez de usar los 768 valores crudos (que capturan semantica de dominio
# y causan overfitting al tema), comprimimos a 7 estadisticos agregados
# que capturan las propiedades distribucionales del embedding.
DIM_BERT_COMPRIMIDO = 7

# Total de features: estilometricas + BERT comprimido
TOTAL_FEATURES = TOTAL_FEATURES_ESTILOMETRICAS + DIM_BERT_COMPRIMIDO


# ============================================================
# 1. REGRESIÓN LOGÍSTICA DESDE CERO
# ============================================================

class RegresionLogisticaDesdeCero:
    """
    Clasificador binario con gradient descent.
    Implementado completamente desde cero con NumPy.
    """

    def __init__(self, learning_rate=0.01, epochs=1000, umbral=0.5, lambda_reg=0.1):
        self.lr = learning_rate
        self.epochs = epochs
        self.umbral = umbral
        self.lambda_reg = lambda_reg  # Regularizacion L2
        self.pesos = None
        self.bias = None
        # Para normalizacion de features
        self.media = None
        self.std = None

    def sigmoid(self, z):
        """Función sigmoide: 1 / (1 + e^(-z))"""
        # Clip para evitar overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def normalizar_features(self, X, entrenamiento=False):
        """
        Normalizacion Z-score para que todas las features tengan
        media 0 y desviacion estandar 1.
        """
        if entrenamiento:
            self.media = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # Evitar division por cero
            self.std[self.std == 0] = 1.0

        return (X - self.media) / self.std

    def fit(self, X, y):
        """
        Entrena el modelo con gradient descent.

        Parámetros:
            X: numpy array (n_muestras, n_features)
            y: numpy array (n_muestras,) con valores 0 o 1
        """
        # Normalizar features
        X_norm = self.normalizar_features(X, entrenamiento=True)

        n_muestras, n_features = X_norm.shape

        # Inicializar pesos en cero
        self.pesos = np.zeros(n_features)
        self.bias = 0.0

        # Gradient descent
        for epoch in range(self.epochs):
            # Forward pass: z = X·w + b
            z = np.dot(X_norm, self.pesos) + self.bias
            predicciones = self.sigmoid(z)

            # Calcular gradientes (derivadas parciales del Binary Cross-Entropy)
            # + regularizacion L2 para evitar overfitting a features BERT
            error = predicciones - y
            grad_pesos = (1.0 / n_muestras) * np.dot(X_norm.T, error) + (self.lambda_reg / n_muestras) * self.pesos
            grad_bias = (1.0 / n_muestras) * np.sum(error)

            # Actualizar parametros
            self.pesos -= self.lr * grad_pesos
            self.bias -= self.lr * grad_bias

            # Imprimir progreso cada 100 epochs
            if (epoch + 1) % 100 == 0:
                loss = self._calcular_loss(y, predicciones)
                acc = self._calcular_accuracy(y, predicciones)
                print(f"  Epoch {epoch + 1}/{self.epochs} | Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

    def _calcular_loss(self, y_real, y_pred):
        """Binary Cross-Entropy Loss."""
        epsilon = 1e-15  # Evitar log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))
        return loss

    def _calcular_accuracy(self, y_real, y_pred_proba):
        """Accuracy del modelo."""
        y_pred = (y_pred_proba >= self.umbral).astype(int)
        correctos = np.sum(y_pred == y_real)
        return (correctos / len(y_real)) * 100

    def predict_proba(self, X):
        """
        Predice la probabilidad de ser IA (clase 1).

        Retorna:
            float o numpy array con probabilidades entre 0.0 y 1.0
        """
        X_norm = self.normalizar_features(X, entrenamiento=False)
        z = np.dot(X_norm, self.pesos) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """
        Predice la clase: 0 (humano) o 1 (IA).
        """
        proba = self.predict_proba(X)
        return (proba >= self.umbral).astype(int)

    def guardar(self, ruta):
        """Guarda los pesos del modelo entrenado."""
        np.savez(ruta,
                 pesos=self.pesos,
                 bias=np.array([self.bias]),
                 media=self.media,
                 std=self.std,
                 umbral=np.array([self.umbral]))
        print(f"[OK] Modelo guardado en: {ruta}")

    def cargar(self, ruta):
        """Carga los pesos del modelo entrenado."""
        datos = np.load(ruta)
        self.pesos = datos['pesos']
        self.bias = float(datos['bias'][0])
        self.media = datos['media']
        self.std = datos['std']
        self.umbral = float(datos['umbral'][0])


# ============================================================
# 2. EXTRACCIÓN DE EMBEDDINGS BERT
# ============================================================

# Variables globales para cachear el modelo BERT (solo se carga una vez)
_bert_tokenizer = None
_bert_model = None


def _cargar_bert():
    """Carga el modelo BERT una sola vez y lo cachea en memoria."""
    global _bert_tokenizer, _bert_model

    if _bert_tokenizer is None or _bert_model is None:
        print("[BERT] Cargando modelo bert-base-uncased...")

        from transformers import BertTokenizer, BertModel
        import torch

        _bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        _bert_model = BertModel.from_pretrained('bert-base-uncased')
        _bert_model.eval()  # Modo evaluación (no entrenamiento)

        print("[BERT] Modelo cargado correctamente.")

    return _bert_tokenizer, _bert_model


def extraer_embedding_bert(texto):
    """
    Extrae el embedding [CLS] de BERT para un texto dado.

    BERT se usa SOLO como extractor de features (embeddings).
    No se hace fine-tuning. Los pesos de BERT no se modifican.

    Parámetros:
        texto: str - Texto crudo

    Retorna:
        numpy array de 768 dimensiones
    """
    import torch

    tokenizer, model = _cargar_bert()

    # Tokenizar el texto (máximo 512 tokens para BERT)
    inputs = tokenizer(
        texto,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )

    # Forward pass sin calcular gradientes (solo extracción)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extraer el vector [CLS] (primer token) de la última capa
    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()

    return cls_embedding


# ============================================================
# 3. PIPELINE PRINCIPAL DE DETECCIÓN
# ============================================================

def comprimir_embedding_bert(embedding_768):
    """
    Comprime el embedding BERT de 768 dimensiones a 7 estadisticos.

    Usar los 768 valores crudos causa overfitting al dominio/tema del texto.
    Los estadisticos agregados capturan las propiedades distribucionales
    del embedding sin depender del tema especifico.

    Retorna 7 features:
        0: media, 1: std, 2: norma L2, 3: max, 4: min, 5: skewness, 6: kurtosis
    """
    media = np.mean(embedding_768)
    std = np.std(embedding_768)
    norma = np.linalg.norm(embedding_768)
    maximo = np.max(embedding_768)
    minimo = np.min(embedding_768)

    # Skewness (asimetria): mide si la distribucion es sesgada
    if std > 0:
        skewness = np.mean(((embedding_768 - media) / std) ** 3)
    else:
        skewness = 0.0

    # Kurtosis (curtosis): mide la "cola" de la distribucion
    if std > 0:
        kurtosis = np.mean(((embedding_768 - media) / std) ** 4) - 3.0
    else:
        kurtosis = 0.0

    return np.array([media, std, norma, maximo, minimo, skewness, kurtosis])


def construir_vector_features(texto, tokens):
    """
    Construye el vector completo de features combinando:
    - 15 features estilometricas
    - 7 features de BERT comprimidas (estadisticos del embedding)

    Total: 22 features

    Parametros:
        texto: str - Texto crudo completo
        tokens: list - Tokens lematizados

    Retorna:
        numpy array de 22 dimensiones
    """
    # Features estilometricas (15)
    feat_estilo = extraer_features_estilometricas(texto, tokens)

    # Embedding BERT (768) -> comprimido a 7 estadisticos
    embedding_crudo = extraer_embedding_bert(texto)
    feat_bert = comprimir_embedding_bert(embedding_crudo)

    # Concatenar ambos vectores
    vector_completo = np.concatenate([feat_estilo, feat_bert])

    return vector_completo


def pipeline_deteccion_ia(texto, tokens):
    """
    Pipeline completo de detección de IA.

    Parámetros:
        texto: str - Texto crudo extraído del PDF
        tokens: list - Tokens lematizados del preprocesamiento

    Retorna:
        float - Porcentaje de probabilidad de ser IA (0.0 a 100.0)
    """
    # Verificar que el modelo entrenado existe
    if not os.path.exists(RUTA_MODELO_IA):
        print("[ADVERTENCIA] No se encontró el modelo entrenado de IA.")
        print(f"  Ruta esperada: {RUTA_MODELO_IA}")
        print("  Ejecuta primero: python -m backend.Entrenar_detector_ia")
        return 0.0

    # Cargar modelo entrenado
    modelo = RegresionLogisticaDesdeCero()
    modelo.cargar(RUTA_MODELO_IA)

    # Construir vector de features
    vector = construir_vector_features(texto, tokens)

    # Predecir probabilidad
    vector_2d = vector.reshape(1, -1)
    probabilidad = modelo.predict_proba(vector_2d)

    # Convertir a porcentaje
    if isinstance(probabilidad, np.ndarray):
        porcentaje = float(probabilidad[0]) * 100.0
    else:
        porcentaje = float(probabilidad) * 100.0

    return round(porcentaje, 2)
