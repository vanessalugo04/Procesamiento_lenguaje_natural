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

# vocabulario extendido de las palabras clave
# 1. ASTRONOMY
astronomy_words = [
    "planet", "galaxy", "telescope", "star", "nebula", "orbit", "gravity", "comet", "asteroid", "universe", "cosmos", "supernova", "eclipse", "constellation", "satellite", "rocket", "astronaut", "spacecraft", "crater", "meteor",
    "exoplanet", "radiation", "telescopic", "observatory", "lunar", "solar", "interstellar", "wavelength", "spectrum", "hubble", "nasa", "kepler", "milkyway", "andromeda", "jupiter", "saturn", "mars", "venus", "mercury", "neptune",
    "uranus", "pluto", "astrophysics", "cosmology", "cluster", "singularity", "horizon", "spacetime", "quasar", "pulsar", "parallax", "magnitude", "luminosity", "rotation", "revolution", "axis", "zenith", "nadir", "solstice", "equinox",
    "flare", "sunspot", "corona", "aurora", "magnetosphere", "trajectory", "propulsion", "telemetry", "payload", "orbiting", "terrestrial", "gaseous", "vacuum", "darkmatter", "darkenergy", "holographic", "expansion", "redshift", "blueshift", "lightyear",
    "parsec", "declination", "altitude", "azimuth", "celestial", "equator", "hemisphere", "photosphere", "chromosphere", "neutrino", "photon", "spectroscopy", "interplanetary", "reentry", "capsule", "rover", "probe", "flyby", "crust", "mantle"
]

# 2. GASTRONOMY
gastronomy_words = [
    "recipe", "ingredient", "cooking", "chef", "baking", "restaurant", "menu", "flavor", "taste", "cuisine", "delicious", "gourmet", "kitchen", "utensil", "seasoning", "spice", "herb", "sauce", "marinate", "grill", "roast", "boil", "fry", "saute", "simmer", "steam", "chop", "slice", "dice", "blend",
    "whisk", "bake", "pastry", "dessert", "appetizer", "entree", "beverage", "wine", "sommelier", "tastebuds", "culinary", "gastronomic", "refrigeration", "pantry", "ovens", "stove", "skillet", "platter", "garnish", "plating",
    "palate", "savory", "sweet", "bitter", "sour", "salty", "umami", "aroma", "texture", "crunchy", "tender", "juicy", "dough", "yeast", "flour", "sugar", "butter", "oil", "vinegar", "reduction",
    "broth", "soup", "stew", "salad", "dressing", "condiment", "fermentation", "pickle", "brine", "infusion",
    "confectionery", "bakery", "michelin", "degustation", "catering", "buffet", "smoked", "charred", "braised", "poached",
    "simmered", "knead", "sift", "grate", "peel", "carve", "sear", "glaze", "zest", "bouillon"
]

# 3. FINANCES
finances_words = [
    "bank", "inflation", "investment", "economy", "market", "stock", "bond", "shares", "dividend", "interest",
    "capital", "asset", "liability", "revenue", "profit", "loss", "budget", "expense", "tax", "fiscal",
    "monetary", "currency", "dollar", "euro", "crypto", "bitcoin", "blockchain", "wallet", "ledger", "transaction",
    "portfolio", "broker", "trader", "exchange", "nasdaq", "index", "commodity", "future", "option", "derivative",
    "liquidity", "solvency", "bankruptcy", "audit", "accounting", "loan", "mortgage", "credit", "debt", "equity",
    "valuation", "deprecation", "appreciation", "recession", "depression", "growth", "gdp", "commerce", "trade", "tariff",
    "subsidy", "deficit", "surplus", "funding", "capitalism", "venture", "startup", "investor", "shareholder", "yield",
    "return", "risk", "hedge", "arbitrage", "speculation", "leverage", "margin", "collateral", "forex", "treasury",
    "fund", "endowment", "pension", "annuity", "insurance", "premium", "claim", "actuary", "underwriting", "microeconomics",
    "macroeconomics", "econometrics", "wealth", "income", "salary", "wage", "payroll", "invoice", "receipt", "ledger"
]

# 4. MEDICINE
medicine_words = [
    "symptom", "patient", "doctor", "hospital", "clinic", "nurse", "surgery", "operation", "treatment", "therapy",
    "diagnosis", "prognosis", "disease", "illness", "infection", "virus", "bacteria", "pathogen", "immune", "vaccine",
    "antibiotic", "prescription", "medication", "drug", "pharmacy", "dosage", "pill", "capsule", "injection", "needle",
    "blood", "plasma", "cell", "tissue", "organ", "transplant", "anatomy", "physiology", "pathology", "oncology",
    "cardiology", "neurology", "pediatrics", "geriatrics", "psychiatry", "radiology", "ultrasound", "mri", "ctscan", "xray",
    "trauma", "emergency", "ambulance", "stretcher", "ward", "icu", "biopsy", "tumor", "cancer", "benign",
    "malignant", "metastasis", "remission", "chronic", "acute", "syndrome", "disorder", "genetic", "mutation", "chromosome",
    "dna", "rna", "protein", "enzyme", "hormone", "endocrine", "metabolism", "diabetes", "hypertension", "stroke",
    "heartattack", "aneurysm", "respiratory", "asthma", "pneumonia", "allergy", "inflammation", "swelling", "fever", "chills",
    "nausea", "vomiting", "dizziness", "fatigue", "pain", "anesthesia", "surgeon", "incision", "suture", "sterile"
]

# 5. LAW
law_words = [
    "law", "lawyer", "court", "judge", "jury", "trial", "verdict", "sentence", "appeal", "litigation",
    "lawsuit", "plaintiff", "defendant", "prosecutor", "attorney", "counsel", "witness", "testimony", "evidence", "exhibit",
    "crime", "felony", "misdemeanor", "offense", "illegal", "legal", "statute", "regulation", "code", "constitution",
    "amendment", "right", "justice", "equity", "jurisdiction", "jurisprudence", "decree", "injunction", "subpoena", "warrant",
    "arrest", "bail", "custody", "prison", "jail", "parole", "probation", "guilty", "innocent", "acquittal",
    "conviction", "plea", "bargain", "testify", "perjury", "fraud", "negligence", "liability", "tort", "contract",
    "agreement", "breach", "clause", "provision", "treaty", "charter", "arbitration", "mediation", "settlement", "damages",
    "indemnity", "property", "tenant", "landlord", "lease", "deed", "estate", "will", "trust", "probate",
    "copyright", "patent", "trademark", "infringement", "piracy", "libel", "slander", "defamation", "bribery", "corruption",
    "extradition", "asylum", "amnesty", "clemency", "malpractice", "notary", "brief", "deposition", "hearing", "docket"
]

# 6. MUSIC
music_words = [
    "chord", "melody", "instrument", "song", "album", "artist", "band", "concert", "gig", "tour",
    "track", "rhythm", "beat", "tempo", "pitch", "tone", "harmony", "scale", "interval", "key",
    "clef", "staff", "notation", "score", "lyrics", "verse", "chorus", "bridge", "outro", "intro",
    "acoustic", "electric", "amplifier", "microphone", "speaker", "headphones", "studio", "recording", "mixing", "mastering",
    "producer", "composer", "conductor", "orchestra", "symphony", "ensemble", "choir", "vocalist", "singer", "guitar",
    "piano", "drums", "bass", "violin", "cello", "flute", "trumpet", "saxophone", "synthesizer", "keyboard",
    "genre", "rock", "pop", "jazz", "blues", "classical", "hiphop", "rap", "reggae", "electronic",
    "techno", "ambient", "folk", "metal", "punk", "indie", "alternative", "festival", "stage", "venue",
    "audience", "fan", "applause", "encore", "rehearsal", "practice", "jam", "solos", "improvisation", "syncopation",
    "arpeggio", "staccato", "legato", "vibrato", "tremolo", "distortion", "reverb", "delay", "equalizer", "compressor"
]

# 7. SPORTS
sports_words = [
    "ball", "train", "competition", "game", "match", "team", "player", "athlete", "coach", "manager",
    "referee", "umpire", "whistle", "stadium", "arena", "field", "court", "track", "gym", "fitness",
    "workout", "exercise", "training", "practice", "session", "tournament", "championship", "league", "cup", "trophy",
    "medal", "gold", "silver", "bronze", "podium", "victory", "defeat", "score", "point", "goal",
    "touchdown", "home-run", "basket", "net", "ring", "glove", "boot", "jersey", "helmet", "pads",
    "racket", "bat", "club", "stick", "board", "skis", "skates", "bicycle", "runner", "swimmer",
    "marathon", "sprint", "relay", "athletics", "gymnastics", "football", "soccer", "basketball", "baseball", "tennis",
    "golf", "cricket", "rugby", "volleyball", "handball", "hockey", "boxing", "wrestling", "karate", "judo",
    "swimming", "diving", "rowing", "sailing", "cycling", "skiing", "snowboarding", "skateting", "surfing", "climbing",
    "hiking", "jogging", "stretching", "warmup", "cardio", "endurance", "strength", "speed", "agility", "strategy"
]

# 8. BOTANY
sports_words = ["train" if w == "train" else w for w in sports_words] # Mantenemos coherencia
botany_words = [
    "plant", "chlorophyll", "root", "stem", "leaf", "flower", "petal", "sepal", "stamen", "pistil",
    "pollen", "pollination", "seed", "fruit", "berry", "spore", "fern", "moss", "algae", "fungi",
    "mushroom", "tree", "shrub", "herb", "grass", "vine", "bark", "branch", "twig", "bud",
    "blossom", "foliage", "canopy", "flora", "vegetation", "ecosystem", "habitat", "biome", "photosynthesis", "respiration",
    "transpiration", "absorption", "osmosis", "vascular", "xylem", "phloem", "stomata", "chloroplast", "cellulose", "sap",
    "resin", "latex", "nectar", "soil", "dirt", "compost", "fertilizer", "nutrients", "nitrogen", "phosphorus",
    "potassium", "water", "moisture", "humidity", "climate", "temperature", "sunlight", "shade", "growth", "germination",
    "seedling", "development", "maturity", "reproduction", "hybrid", "variety", "species", "genus", "family", "order",
    "class", "phylum", "kingdom", "taxonomy", "classification", "identification", "specimen", "herbarium", "greenhouse", "garden",
    "agriculture", "horticulture", "forestry", "conservation", "ecology", "botanist", "research", "study", "analysis", "experiment"
]

# 9. ARCHAEOLOGY
archaeology_words = [
    "fossil", "ruin", "ancient", "dig", "excavation", "site", "artifact", "relic", "history", "prehistory",
    "past", "culture", "civilization", "society", "period", "era", "age", "stone", "bronze", "iron",
    "pottery", "ceramic", "sherd", "vessel", "tool", "weapon", "axe", "knife", "arrowhead", "flint",
    "bone", "skeleton", "skull", "burial", "grave", "tomb", "cemetery", "monument", "temple", "palace",
    "pyramid", "castle", "fortress", "wall", "settlement", "village", "city", "architecture", "structure", "foundation",
    "stratigraphy", "layer", "sediment", "soil", "dating", "radiocarbon", "carbon-14", "chronology", "analysis", "conservation",
    "restoration", "museum", "collection", "curator", "archive", "document", "record", "survey", "mapping", "gis",
    "photography", "drawing", "illustration", "measurement", "description", "catalog", "inventory", "report", "publication", "theory",
    "interpretation", "anthropology", "ethnography", "heritage", "preservation", "protection", "looting", "vandalism", "treasure", "gold",
    "silver", "copper", "bronze", "jewelry", "bead", "coin", "inscription", "hieroglyph", "petroglyph", "pictograph"
]

# 10. COMPUTATION
computation_words = [
    "code", "software", "server", "intelligence", "artificial", "algorithm", "program", "developer", "programmer", "coding",
    "hardware", "computer", "processor", "cpu", "gpu", "ram", "memory", "storage", "disk", "ssd",
    "network", "internet", "web", "website", "application", "app", "database", "sql", "data", "analytics",
    "cloud", "aws", "azure", "google", "microsoft", "apple", "linux", "windows", "ios", "android",
    "security", "cybersecurity", "hacking", "hacker", "encryption", "decryption", "password", "authentication", "authorization", "firewall",
    "antivirus", "malware", "virus", "bug", "error", "debugging", "testing", "quality", "assurance", "deployment",
    "github", "git", "repository", "branch", "commit", "push", "pull", "merge", "conflict", "version",
    "control", "framework", "library", "api", "json", "xml", "html", "css", "javascript", "python",
    "java", "c++", "c#", "php", "ruby", "go", "rust", "typescript", "compiler", "interpreter",
    "ide", "editor", "terminal", "console", "command", "line", "script", "automation", "robotics", "machine"
]


# Unir todas las listas en un conjunto para eliminar duplicados automáticamente
vocabulario_set = set(
    astronomy_words + gastronomy_words + finances_words + medicine_words + law_words +
    music_words + sports_words + botany_words + archaeology_words + computation_words
)

# Convertir a lista y ordenar alfabéticamente (el ordenamiento ayuda a la consistencia visual)
vector_unico = sorted(list(vocabulario_set))

# Corpus de entrenamiento expandido automáticamente usando tus variables temáticas
corpus_entrenamiento = [
    astronomy_words[:20], astronomy_words[20:40],  # 2 documentos de Astronomía
    gastronomy_words[:20], gastronomy_words[20:40],  # 2 de Gastronomía
    finances_words[:20], finances_words[20:40],      # ... etc
    medicine_words[:20], medicine_words[20:40],
    law_words[:20], law_words[20:40],
    music_words[:20], music_words[20:40],
    sports_words[:20], sports_words[20:40],
    botany_words[:20], botany_words[20:40],
    archaeology_words[:20], archaeology_words[20:40],
    computation_words[:20], computation_words[20:40]
]

# Mantienes tus 20 etiquetas correspondientes
y_entrenamiento = [
    "Astronomy", "Astronomy", "Gastronomy", "Gastronomy", "Finances", "Finances",
    "Medicine", "Medicine", "Law", "Law", "Music", "Music", "Sports", "Sports",
    "Botany", "Botany", "Archaeology", "Archaeology", "Computation", "Computation"
]

# Ejecutar el entrenamiento matemático de forma automática al importar este archivo
X_train = matriz_bag_words(corpus_entrenamiento, vector_unico)
modelo_tema = NaiveBayesDesdeCero()
modelo_tema.fit(X_train, y_entrenamiento)

