import numpy as np
import tensorflow as tf
from voz import hablar, escuchar_voz

#  Cuento El Principito  
TEXTO = """
El Principito es una novela corta y la obra más famosa del escritor y aviador
francés Antoine de Saint-Exupéry, publicada en 1943. La historia comienza cuando
el narrador, un aviador adulto, se queda varado en el desierto del Sahara tras
una avería en su avión. Allí conoce a un extraño niño de cabello dorado que le
pide que le dibuje un cordero. Este niño es el Principito, un ser llegado de un
asteroide muy pequeño llamado B-612.

El Principito le cuenta al aviador sus aventuras y los planetas que visitó antes
de llegar a la Tierra. Su propio planeta era tan diminuto que podía ver la puesta
de sol varias veces al día con solo mover su silla. En ese asteroide habitaba
una rosa muy orgullosa y vanidosa que el Principito amaba profundamente, pero
cuyos caprichos y mentiras le hacían sufrir. La rosa afirmaba ser única en el
mundo y necesitaba protección constante bajo un fanal de cristal. Incapaz de
comprender los sentimientos de la flor, el Principito decidió partir para explorar
otros mundos y comprender mejor a los seres vivos.

Durante su viaje por el universo, el Principito visitó seis planetas habitados
por adultos que le parecieron muy extraños. En el primero vivía un Rey que
quería gobernar sobre todo, pero que en realidad no tenía súbditos. El Rey
insistía en que sus órdenes siempre eran obedecidas, pero solo daba órdenes
que ya eran razonables, como ordenarle al sol que se pusiera cuando ya era
de noche. En el segundo planeta vivía un Vanidoso que solo quería recibir
aplausos y admiración, sin importarle nada más en el mundo.

El tercer planeta estaba habitado por un Bebedor que bebía para olvidar
la vergüenza que le daba beber, un círculo vicioso que dejó al Principito
muy triste y confundido. En el cuarto planeta vivía un Hombre de Negocios
que contaba estrellas para creer que las poseía, aunque en realidad no
sabía hacer nada útil con ellas. El quinto planeta era tan pequeño que
apenas cabía un farol y un Farolero que encendía y apagaba el farol
cada minuto, siguiendo órdenes antiguas sin cuestionarlas jamás.
El Principito admiró al Farolero por ser el único que pensaba en algo
distinto a sí mismo.

En el sexto planeta vivía un Geógrafo anciano que escribía libros sobre
montañas y ríos, pero que nunca había salido a explorar su propio planeta.
El Geógrafo le recomendó al Principito visitar la Tierra porque era
un planeta reputado. Así fue como el pequeño viajero llegó a nuestro planeta
y aterrizó en el desierto.

En la Tierra, el Principito encontró primero una serpiente que le habló
con enigmas y le advirtió que podía devolverle a su planeta cuando quisiera.
Después encontró una flor de tres pétalos en el desierto y atravesó un
jardín lleno de rosas exactamente iguales a su rosa. Esto le causó una
gran tristeza porque su rosa le había dicho que era única en el universo.
El Principito se sentó en la hierba y lloró.

Fue entonces cuando apareció el Zorro, el personaje más sabio de toda la historia.
El Zorro le pidió al Principito que lo domesticara, es decir, que creara lazos
entre ellos. El Zorro le explicó que domesticar significa crear vínculos únicos
y que hacen que algo o alguien sea especial para otro. Le dijo que su rosa era
única en el mundo porque el Principito la había regado, le había puesto un fanal,
la había escuchado quejarse y presumir. El Zorro le enseñó el secreto más
importante: solo se ve bien con el corazón, pues lo esencial es invisible
a los ojos. También le dijo que uno es responsable para siempre de lo que
ha domesticado, es decir, de aquello a lo que ha creado lazos afectivos.

El Principito comprendió entonces que su rosa, con todos sus defectos, era
su rosa y eso la hacía única. Regresó donde estaba la serpiente y le pidió
que lo devolviera a su planeta junto a su flor. El aviador intentó disuadirlo
pero el Principito le explicó que su cuerpo era demasiado pesado para llevar
tan lejos, que solo podía volver como alma. La serpiente lo picó y el Principito
cayó suavemente como un árbol en el desierto dorado.

Al día siguiente el aviador no encontró el cuerpo del niño. Miró las estrellas
y recordó las palabras del Principito: que cuando mirara el cielo nocturno y
viera una estrella reír, sabría que él estaba allí. Desde entonces el aviador
mira las estrellas de manera diferente. La novela termina con una reflexión
sobre la importancia de mantener viva la capacidad de asombro y de ver el
mundo con los ojos del corazón, sin dejarse atrapar por la rutina y la
seriedad innecesaria del mundo adulto. El Principito nos recuerda que lo más
importante en la vida no son las posesiones ni el poder, sino los lazos que
creamos con quienes amamos y la capacidad de ver la belleza donde los adultos
ya no son capaces de verla. Es un cuento para niños y adultos que nos invita
a recordar quiénes éramos antes de olvidar que lo esencial es invisible a los ojos.
"""

#  Preprocesamiento
STOPWORDS = [
    'a', 'al', 'antes', 'como', 'con', 'de', 'del', 'desde', 'durante',
    'el', 'en', 'era', 'eran', 'es', 'estas', 'fue', 'fueron', 'ha',
    'han', 'hasta', 'la', 'las', 'le', 'lo', 'los', 'o', 'otra', 'otro',
    'para', 'por', 'que', 'quienes', 'se', 'si', 'sin', 'su', 'sus',
    'también', 'un', 'una', 'y', 'no', 'más', 'pero', 'ser', 'son',
    'esto', 'este', 'esta', 'hay', 'ni', 'cual', 'cuyo', 'cuya',
    'le', 'les', 'me', 'nos', 'te', 'muy', 'tan', 'ya', 'cuando',
    'todo', 'todos', 'toda', 'todas', 'así', 'bien', 'fue', 'ser',
]


def es_valido(caracter):
    c = ord(caracter)
    if (65 <= c <= 90) or (97 <= c <= 122):
        return True
    vocales_tildes_n = [
        225, 233, 237, 243, 250, 241,
        193, 201, 205, 211, 218, 209,
        252, 220
    ]
    return c in vocales_tildes_n


def to_minusculas(texto):
    letras = ""
    for letra in texto:
        if 65 <= ord(letra) <= 90:
            letras += chr(ord(letra) + 32)
        else:
            letras += letra
    return letras


def tokenizador(texto, stopwords):
    token = ""
    tokens = [None] * len(texto)
    j = 0
    for i in range(len(texto)):
        if es_valido(texto[i]):
            token += texto[i]
        else:
            if token != "":
                if token not in stopwords:
                    tokens[j] = token
                    j += 1
                token = ''
    if token and token not in stopwords:
        tokens[j] = token
        j += 1
    return tokens[0:j]


def dividir_oraciones(texto):
    oraciones = []
    actual = ""
    for c in texto:
        actual += c
        if c in ['.', '!', '?'] and len(actual.strip()) > 15:
            oraciones.append(actual.strip())
            actual = ""
    return [o for o in oraciones if len(o) > 20]


# WORD2VEC  
# Arquitectura Skip-gram
def one_hot(indice, tamaño):
    vector = np.zeros(tamaño)
    vector[indice] = 1
    return vector

def similitud_coseno(a, b):
    numerador   = np.dot(a, b)
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    if denominador == 0:
        return 0.0
    return numerador / denominador

def vector_oracion(tokens, word2idx, embeddings, embedding_dim):
    vecs = []
    for t in tokens:
        if t in word2idx:
            vecs.append(embeddings[word2idx[t]])
    if len(vecs) == 0:
        return np.zeros(embedding_dim)
    total = np.zeros(embedding_dim)
    for v in vecs:
        total = total + v
    return total / len(vecs)


def entrenar_word2vec(tokens_global, vocab_size, word2idx,
                      embedding_dim=30, window_size=2, epochs=500):
    # Entrena Word2Vec Skip-gram con TensorFlow y devuelve la matriz de embeddings

    # Pares Skip-gram
    X_idx, y_idx = [], []
    for i in range(len(tokens_global)):
        central_idx = word2idx[tokens_global[i]]
        inicio = max(0, i - window_size)
        fin    = min(len(tokens_global), i + window_size + 1)
        for j in range(inicio, fin):
            if i != j:
                X_idx.append(central_idx)
                y_idx.append(word2idx[tokens_global[j]])

    X_oh = np.array([one_hot(i, vocab_size) for i in X_idx], dtype=np.float32)
    y_oh = np.array([one_hot(i, vocab_size) for i in y_idx], dtype=np.float32)

    # Modelo
    W1 = tf.Variable(tf.random.normal([vocab_size, embedding_dim]))
    W2 = tf.Variable(tf.random.normal([embedding_dim, vocab_size]))

    def forward(x):
        hidden = tf.matmul(x, W1)
        return tf.matmul(hidden, W2)

    def loss_fn(logits, labels):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        )

    optimizer = tf.optimizers.Adam(0.01)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            logits = forward(X_oh)
            loss   = loss_fn(logits, y_oh)
        grads = tape.gradient(loss, [W1, W2])
        optimizer.apply_gradients(zip(grads, [W1, W2]))
        #if epoch % 100 == 0:
            #print(f"  Epoch {epoch:>4}  Loss: {loss.numpy():.4f}")

    return W1.numpy()


def generar_resumen(n_oraciones=8):
    # Genera el resumen de El Principito usando Word2Vec
    print("Entrenando Word2Vec para el resumen de El Principito...")

    # Preprocesamiento
    oraciones_originales = dividir_oraciones(TEXTO)
    tokens_global        = tokenizador(to_minusculas(TEXTO), STOPWORDS)

    # Vocabulario
    vocab_lista = []
    for t in tokens_global:
        ya_esta = False
        for v in vocab_lista:
            if v == t:
                ya_esta = True
                break
        if not ya_esta:
            vocab_lista.append(t)

    vocab_size = len(vocab_lista)
    word2idx   = {vocab_lista[i]: i for i in range(vocab_size)}

    # Entrenar Word2Vec
    embedding_dim = 30
    embeddings    = entrenar_word2vec(
        tokens_global, vocab_size, word2idx,
        embedding_dim=embedding_dim, window_size=2, epochs=500
    )

    # Vector del documento completo
    vec_doc = vector_oracion(tokens_global, word2idx, embeddings, embedding_dim)

    # Puntuar oraciones por similitud coseno con el documento
    puntuaciones = []
    for i, oracion in enumerate(oraciones_originales):
        toks  = tokenizador(to_minusculas(oracion), STOPWORDS)
        vec   = vector_oracion(toks, word2idx, embeddings, embedding_dim)
        score = similitud_coseno(vec, vec_doc)
        puntuaciones.append((i, score))

    # Ordenar descendente 
    for i in range(len(puntuaciones) - 1):
        for j in range(i + 1, len(puntuaciones)):
            if puntuaciones[j][1] > puntuaciones[i][1]:
                puntuaciones[i], puntuaciones[j] = puntuaciones[j], puntuaciones[i]

    # Reordenadar por posición original
    top_indices = [p[0] for p in puntuaciones[:n_oraciones]]
    top_indices_ordenados = sorted(top_indices)

    resumen = " ".join([oraciones_originales[i] for i in top_indices_ordenados])
    return resumen


#  Función principal
def cuento():
    hablar("¿Quieres escuchar el resumen de El Principito?")
    respuesta = escuchar_voz(5)

    if any(p in respuesta for p in ["si", "sí", "claro", "dale", "quiero", "sí quiero", "por supuesto", "si, por favor"]):
        hablar("Un momento, estoy preparando el resumen.")
        resumen = generar_resumen(n_oraciones=8)
        print("\nResumen del Principito")
        hablar(resumen)
        print("\nListo :)")
    else:
        hablar("De acuerdo, hasta luego.")