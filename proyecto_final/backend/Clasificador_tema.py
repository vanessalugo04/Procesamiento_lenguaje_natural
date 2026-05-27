from backend.Clasificador import matriz_bag_words, NaiveBayesDesdeCero

# =========================================================================
# ENTRENAMIENTO DEL CLASIFICADOR 

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
    "Gastronomy", "Gastronomia",
    "Finances", "Finances",
    "Medicine", "Medicine",
    "Law", "Law",
    "Music", "Music",
    "Sports", "Sports",
    "Botany", "Botany",
    "Archaeology", "Archaeology",
    "Computation", "Computation"
]

# Ejecutar el entrenamiento matemático nativo al cargar el módulo
X_train = matriz_bag_words(corpus_entrenamiento, vector_unico)
modelo_tema = NaiveBayesDesdeCero()
modelo_tema.fit(X_train, y_entrenamiento)