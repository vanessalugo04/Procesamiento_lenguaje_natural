import numpy as np

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

class NaiveBayesDesdeCero:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Suavizado de Laplace para evitar probabilidades de 0
        self.clases = None
        self.prob_previas = {}
        self.prob_condicionales = {}
        
    def fit(self, X, y):
        y = np.array(y)
        self.clases = np.unique(y)
        total_documentos = len(y)
        num_caracteristicas = X.shape[1]
        
        for clase in self.clases:
            # 1. Calcular probabilidad previa de la clase: P(Clase)
            documentos_clase = X[y == clase]
            self.prob_previas[clase] = len(documentos_clase) / total_documentos
            
            # 2. Conteo total de palabras en esta clase + Suavizado de Laplace
            conteo_palabras_por_caracteristica = np.sum(documentos_clase, axis=0)
            total_palabras_clase = np.sum(conteo_palabras_por_caracteristica)
            
            # 3. Calcular P(Palabra | Clase) aplicando Laplace
            numerador = conteo_palabras_por_caracteristica + self.alpha
            denominador = total_palabras_clase + (self.alpha * num_caracteristicas)
            self.prob_condicionales[clase] = numerador / denominador
            
    def predict(self, X):
        predicciones = []
        for x in X:
            posteriors = {}
            for clase in self.clases:
                # Usamos logaritmos para evitar subdesbordamiento (underflow) numérico
                log_prior = np.log(self.prob_previas[clase])
                
                # Multiplicar las frecuencias del vector por los logaritmos de las probabilidades condicionales
                log_likelihood = np.sum(x * np.log(self.prob_condicionales[clase]))
                
                posteriors[clase] = log_prior + log_likelihood
                
            # La clase con el valor logarítmico más alto es la ganadora
            clase_ganadora = max(posteriors, key=posteriors.get)
            predicciones.append(clase_ganadora)
            
        return predicciones