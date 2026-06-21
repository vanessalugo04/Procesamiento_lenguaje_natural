import numpy as np
import math

# ============================================================
# MÓDULO DE ESTILOMETRÍA - Features estilométricas desde cero
# ============================================================
# Extrae 15 características numéricas del texto que permiten
# distinguir entre escritura humana y escritura generada por IA.
# ============================================================


# ============================================================
# 1. UTILIDADES DE SEGMENTACIÓN
# ============================================================

def dividir_oraciones(texto):
    """Divide el texto en oraciones usando puntuación terminal."""
    oraciones = []
    oracion_actual = ""

    for char in texto:
        oracion_actual += char
        if char in '.!?':
            limpia = oracion_actual.strip()
            if len(limpia) > 5:  # Evitar fragmentos vacíos
                oraciones.append(limpia)
            oracion_actual = ""

    # Última oración sin punto final
    if len(oracion_actual.strip()) > 5:
        oraciones.append(oracion_actual.strip())

    return oraciones


def dividir_palabras(texto):
    """Divide texto en palabras (split simple por espacios)."""
    palabras = []
    palabra_actual = ""

    for char in texto:
        if char == ' ' or char == '\n' or char == '\t' or char == '\r':
            if len(palabra_actual) > 0:
                palabras.append(palabra_actual)
            palabra_actual = ""
        else:
            palabra_actual += char

    if len(palabra_actual) > 0:
        palabras.append(palabra_actual)

    return palabras


# ============================================================
# 2. FEATURES DE DIVERSIDAD LÉXICA
# ============================================================

def calcular_ttr(tokens):
    """
    Type-Token Ratio: proporción de palabras únicas.
    IA tiende a tener TTR más bajo (vocabulario más repetitivo).
    """
    if len(tokens) == 0:
        return 0.0

    tipos = set()
    for t in tokens:
        tipos.add(t.lower())

    return len(tipos) / len(tokens)


def calcular_hapax_ratio(tokens):
    """
    Hapax Legomena Ratio: proporción de palabras que aparecen solo una vez.
    Humanos tienden a usar más palabras únicas (hapax más alto).
    """
    if len(tokens) == 0:
        return 0.0

    conteo = {}
    for t in tokens:
        t_lower = t.lower()
        if t_lower in conteo:
            conteo[t_lower] += 1
        else:
            conteo[t_lower] = 1

    hapax = 0
    for palabra in conteo:
        if conteo[palabra] == 1:
            hapax += 1

    return hapax / len(tokens)


# ============================================================
# 3. FEATURES DE ESTRUCTURA ORACIONAL
# ============================================================

def calcular_promedio_longitud_oraciones(texto):
    """
    Longitud promedio de oraciones en palabras.
    IA tiende a producir oraciones de longitud más uniforme.
    """
    oraciones = dividir_oraciones(texto)
    if len(oraciones) == 0:
        return 0.0

    total = 0
    for oracion in oraciones:
        palabras = dividir_palabras(oracion)
        total += len(palabras)

    return total / len(oraciones)


def calcular_std_longitud_oraciones(texto):
    """
    Desviación estándar de longitud de oraciones (Burstiness).
    Humanos varían MÁS la longitud → std alta.
    IA es más uniforme → std baja.
    """
    oraciones = dividir_oraciones(texto)
    if len(oraciones) < 2:
        return 0.0

    longitudes = []
    for oracion in oraciones:
        palabras = dividir_palabras(oracion)
        longitudes.append(len(palabras))

    # Calcular media
    media = sum(longitudes) / len(longitudes)

    # Calcular varianza
    varianza = 0.0
    for lng in longitudes:
        varianza += (lng - media) ** 2
    varianza = varianza / len(longitudes)

    return math.sqrt(varianza)


def calcular_max_longitud_oracion(texto):
    """Longitud de la oración más larga (en palabras)."""
    oraciones = dividir_oraciones(texto)
    if len(oraciones) == 0:
        return 0

    maximo = 0
    for oracion in oraciones:
        palabras = dividir_palabras(oracion)
        if len(palabras) > maximo:
            maximo = len(palabras)

    return maximo


# ============================================================
# 4. FEATURES DE COMPLEJIDAD LÉXICA
# ============================================================

def calcular_promedio_longitud_palabras(tokens):
    """Longitud promedio de palabras en caracteres."""
    if len(tokens) == 0:
        return 0.0

    total_chars = 0
    for t in tokens:
        total_chars += len(t)

    return total_chars / len(tokens)


def calcular_ratio_palabras_largas(tokens, umbral=8):
    """
    Proporción de palabras con más de 'umbral' caracteres.
    IA usa vocabulario más sofisticado con palabras más largas.
    """
    if len(tokens) == 0:
        return 0.0

    largas = 0
    for t in tokens:
        if len(t) > umbral:
            largas += 1

    return largas / len(tokens)


# ============================================================
# 5. FEATURES DE MARCADORES TÍPICOS DE IA
# ============================================================

# Conectores y frases que la IA usa con frecuencia excesiva
CONECTORES_IA = [
    "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "nonetheless", "therefore", "thus",
    "hence", "accordingly", "subsequently", "notably",
    "significantly", "essentially", "fundamentally",
    "in conclusion", "in summary", "it is important to note",
    "it is worth noting", "it should be noted",
    "on the other hand", "in other words",
    "as a result", "for instance", "for example",
    "in particular", "specifically", "overall",
    "in this context", "with regard to",
    "delve", "crucial", "comprehensive", "multifaceted",
    "leverage", "facilitate", "utilize", "encompasses",
    "landscape", "paradigm", "robust", "streamline",
    "tapestry", "intricate", "pivotal", "realm"
]


def calcular_frecuencia_conectores_ia(texto):
    """
    Frecuencia de conectores y "buzzwords" típicos de IA.
    Normalizado por total de palabras.
    """
    texto_lower = texto.lower()
    palabras = dividir_palabras(texto_lower)

    if len(palabras) == 0:
        return 0.0

    conteo = 0
    for conector in CONECTORES_IA:
        # Buscar ocurrencias del conector en el texto
        pos = 0
        while pos < len(texto_lower):
            idx = texto_lower.find(conector, pos)
            if idx == -1:
                break
            conteo += 1
            pos = idx + len(conector)

    return conteo / len(palabras)


def calcular_densidad_transiciones(texto):
    """
    Densidad de frases de transición por oración.
    IA usa más transiciones entre oraciones.
    """
    TRANSICIONES = [
        "however", "meanwhile", "furthermore", "moreover",
        "additionally", "consequently", "therefore", "thus",
        "nevertheless", "nonetheless", "in addition",
        "as a result", "on the contrary", "in contrast",
        "similarly", "likewise", "conversely"
    ]

    oraciones = dividir_oraciones(texto)
    if len(oraciones) == 0:
        return 0.0

    conteo = 0
    texto_lower = texto.lower()
    for trans in TRANSICIONES:
        pos = 0
        while pos < len(texto_lower):
            idx = texto_lower.find(trans, pos)
            if idx == -1:
                break
            conteo += 1
            pos = idx + len(trans)

    return conteo / len(oraciones)


# ============================================================
# 6. FEATURES DE ENTROPÍA
# ============================================================

def calcular_entropia_ngramas(tokens, n=2):
    """
    Entropía de Shannon de n-gramas de palabras.
    IA tiene menor entropía (más predecible).
    Humanos tienen mayor entropía (más variado).
    """
    if len(tokens) < n:
        return 0.0

    # Construir n-gramas
    ngramas = {}
    total = 0

    for i in range(len(tokens) - n + 1):
        ngrama = ""
        for j in range(n):
            if j > 0:
                ngrama += " "
            ngrama += tokens[i + j].lower()

        if ngrama in ngramas:
            ngramas[ngrama] += 1
        else:
            ngramas[ngrama] = 1
        total += 1

    if total == 0:
        return 0.0

    # Calcular entropía de Shannon: H = -Σ p(x) * log2(p(x))
    entropia = 0.0
    for ngrama in ngramas:
        p = ngramas[ngrama] / total
        if p > 0:
            entropia -= p * math.log2(p)

    return entropia


# ============================================================
# 7. FEATURES DE PUNTUACIÓN
# ============================================================

def calcular_ratio_puntuacion(texto, caracter):
    """Ratio de un carácter de puntuación por cada 100 palabras."""
    palabras = dividir_palabras(texto)
    if len(palabras) == 0:
        return 0.0

    conteo = 0
    for ch in texto:
        if ch == caracter:
            conteo += 1

    return (conteo / len(palabras)) * 100


# ============================================================
# 8. FEATURE DE REPETICIÓN
# ============================================================

def calcular_indice_repeticion(tokens, window=50):
    """
    Índice de repetición: proporción promedio de tokens repetidos
    en ventanas deslizantes de tamaño 'window'.
    IA tiende a repetir más patrones en ventanas locales.
    """
    if len(tokens) < window:
        if len(tokens) == 0:
            return 0.0
        # Usar todo el texto como una sola ventana
        unicos = set()
        for t in tokens:
            unicos.add(t.lower())
        return 1.0 - (len(unicos) / len(tokens))

    total_repeticion = 0.0
    num_ventanas = 0

    for i in range(len(tokens) - window + 1):
        ventana = tokens[i:i + window]
        unicos = set()
        for t in ventana:
            unicos.add(t.lower())
        repeticion = 1.0 - (len(unicos) / window)
        total_repeticion += repeticion
        num_ventanas += 1

    return total_repeticion / num_ventanas


# ============================================================
# 9. PIPELINE PRINCIPAL - EXTRACTOR DE FEATURES
# ============================================================

def extraer_features_estilometricas(texto, tokens):
    """
    Extrae un vector de 15 features estilométricas del texto.

    Parámetros:
        texto: str - Texto crudo completo (con puntuación y mayúsculas)
        tokens: list - Lista de tokens lematizados (del preprocesamiento existente)

    Retorna:
        numpy array de 15 features numéricas
    """
    features = np.array([
        # Diversidad léxica (2)
        calcular_ttr(tokens),                                    # 0: TTR
        calcular_hapax_ratio(tokens),                            # 1: Hapax ratio

        # Estructura oracional (3)
        calcular_promedio_longitud_oraciones(texto),             # 2: Promedio largo oraciones
        calcular_std_longitud_oraciones(texto),                  # 3: Burstiness
        calcular_max_longitud_oracion(texto),                    # 4: Max largo oración

        # Complejidad léxica (2)
        calcular_promedio_longitud_palabras(tokens),             # 5: Promedio largo palabras
        calcular_ratio_palabras_largas(tokens, umbral=8),        # 6: Ratio palabras largas

        # Marcadores IA (2)
        calcular_frecuencia_conectores_ia(texto),                # 7: Frecuencia conectores IA
        calcular_densidad_transiciones(texto),                   # 8: Densidad transiciones

        # Entropía (2)
        calcular_entropia_ngramas(tokens, n=2),                  # 9: Entropía bigramas
        calcular_entropia_ngramas(tokens, n=3),                  # 10: Entropía trigramas

        # Puntuación (3)
        calcular_ratio_puntuacion(texto, ','),                   # 11: Ratio comas
        calcular_ratio_puntuacion(texto, ';'),                   # 12: Ratio punto y coma
        calcular_ratio_puntuacion(texto, '?'),                   # 13: Ratio interrogaciones

        # Repetición (1)
        calcular_indice_repeticion(tokens, window=50),           # 14: Índice repetición
    ], dtype=np.float64)

    return features


# Nombres de las features para referencia y debugging
NOMBRES_FEATURES = [
    "TTR (Type-Token Ratio)",
    "Hapax Ratio",
    "Promedio Longitud Oraciones",
    "Burstiness (Std Oraciones)",
    "Max Longitud Oracion",
    "Promedio Longitud Palabras",
    "Ratio Palabras Largas",
    "Frecuencia Conectores IA",
    "Densidad Transiciones",
    "Entropia Bigramas",
    "Entropia Trigramas",
    "Ratio Comas",
    "Ratio Punto y Coma",
    "Ratio Interrogaciones",
    "Indice Repeticion",
]

TOTAL_FEATURES_ESTILOMETRICAS = 15
