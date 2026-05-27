import numpy as np

# 1. FUNCIÓN DE MATRIZ BAG OF WORDS
def matriz_bag_words(corpus, vector_unico):
    num_doc_corpus = len(corpus)
    total_vector_unico = len(vector_unico)
    matriz_BOW = np.zeros((num_doc_corpus, total_vector_unico)) 

    for i in range(num_doc_corpus):
        j = 0
        for pal_unico in vector_unico:
            for pal in corpus[i]:
                if pal == pal_unico: 
                    matriz_BOW[i, j] = matriz_BOW[i, j] + 1
            j += 1
    return matriz_BOW


# 2. CLASE DE NAIVE BAYES DESDE CERO
class NaiveBayesDesdeCero:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Suavizado de Laplace
        self.clases = None
        self.prob_previas = {}
        self.prob_condicionales = {}
        
    def fit(self, X, y):
        y = np.array(y)
        self.clases = np.unique(y)
        total_documentos = len(y)
        num_caracteristicas = X.shape[1]
        
        for clase in self.clases:
            # Calcular probabilidad previa: P(Clase)
            documentos_clase = X[y == clase]
            self.prob_previas[clase] = len(documentos_clase) / total_documentos
            
            # Conteo de palabras + Laplace
            conteo_palabras_por_caracteristica = np.sum(documentos_clase, axis=0)
            total_palabras_clase = np.sum(conteo_palabras_por_caracteristica)
            
            # Calcular P(Palabra | Clase)
            numerador = conteo_palabras_por_caracteristica + self.alpha
            denominador = total_palabras_clase + (self.alpha * num_caracteristicas)
            self.prob_condicionales[clase] = numerador / denominador
            
    def predict(self, X):
        predicciones = []
        for x in X:
            posteriors = {}
            for clase in self.clases:
                log_prior = np.log(self.prob_previas[clase])
                log_likelihood = np.sum(x * np.log(self.prob_condicionales[clase]))
                posteriors[clase] = log_prior + log_likelihood
                
            clase_ganadora = max(posteriors, key=posteriors.get)
            predicciones.append(clase_ganadora)
            
        return predicciones


# ----------------------------------------------------------------------------
# 3. CONFIGURACIÓN Y ENTRENAMIENTOOOOOOO

# Vocabulario global con palabras clave
vector_unico = [
    # Astronomy
    "planet", "galaxy", "telescope",
    # Gastronomy
    "recipe", "ingredient", "cooking",
    # Finances
    "bank", "inflation", "investment",
    # Medicine
    "symptom", "patient", "doctor",
    # Law
    "law", "lawyer", "court",
    # Music
    "chord", "melody", "instrument",
    # Sports
    "ball", "train", "competition",
    # Botany
    "plant", "chlorophyll", "root",
    # Archaeology
    "fossil", "ruin", "ancient",
    # Computation
    "code", "software", "server", "intelligence", "artificial"
]

# Corpus de entrenamiento simulado
corpus_entrenamiento = [
    ["planet", "galaxy", "telescope"], ["planet", "telescope"], # Astronomy
    ["recipe", "ingredient", "cooking"], ["recipe", "cooking"],     # Gastronomy
    ["bank", "inflation", "investment"], ["bank", "investment"],     # Finances
    ["symptom", "patient", "doctor"], ["symptom", "doctor"],        # Medicine
    ["law", "lawyer", "court"], ["law", "court"],             # Law
    ["chord", "melody", "instrument"], ["chord", "instrument"], # Music
    ["ball", "train", "competition"], ["ball", "train"],     # Sports
    ["plant", "chlorophyll", "root"], ["plant", "root"],             # Botany
    ["fossil", "ruin", "ancient"], ["ruina", "ancient"],             # Archaeology
    ["code", "software", "server", "intelligence", "artificial"], ["code", "software", "intelligence"] # Computation
]

# Las 10 etiquetas correspondientes
y_entrenamiento = [
    "Astronomy", "Astronomy",
    "Gastronomy", "Gastronomy",
    "Finances", "Finances",
    "Medicine", "Medicine",
    "Law", "Law",
    "Music", "Music",
    "Sports", "Sports",
    "Botany", "Botany",
    "Archaeology", "Archaeology",
    "Computation", "Computation"
]

# Ejecutar el entrenamiento matemático de forma automática al importar este archivo
X_train = matriz_bag_words(corpus_entrenamiento, vector_unico)
modelo_tema = NaiveBayesDesdeCero()
modelo_tema.fit(X_train, y_entrenamiento)