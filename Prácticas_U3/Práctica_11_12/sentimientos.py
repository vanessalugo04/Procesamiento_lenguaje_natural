from voz import hablar

positivas = [
    "feliz",
    "contento",
    "excelente",
    "genial",
    "alegre"
]

negativas = [
    "triste",
    "cansado",
    "mal",
    "estresado",
    "enojado"
]

def analizar_sentimiento(texto):

    texto = texto.lower()

    for palabra in positivas:

        if palabra in texto:
            return "positivo"

    for palabra in negativas:

        if palabra in texto:
            return "negativo"

    return "neutral"

def responder_sentimiento(sentimiento):

    if sentimiento == "positivo":

        hablar(
            "Me alegra escuchar que tuviste un buen día."
        )

    elif sentimiento == "negativo":

        hablar(
            "Lamento que tu día haya sido complicado."
        )

    else:

        hablar(
            "Gracias por compartir cómo estuvo tu día."
        )
