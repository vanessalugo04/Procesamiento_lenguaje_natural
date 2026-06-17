from voz import hablar, escuchar_voz

# Datos curiosos por categoría
datos_pasatiempo = {
    "leer":        "leer tan solo 6 minutos reduce el estrés hasta en un 68 por ciento",
    "música":      "escuchar música activa más regiones del cerebro que cualquier otra actividad humana",
    "videojuegos": "los videojuegos de acción mejoran la capacidad de tomar decisiones rápidas",
    "cocinar":     "cocinar activa zonas del cerebro relacionadas con la creatividad",
    "dibujar":     "dibujar mejora la memoria visual y ayuda a procesar emociones",
    "bailar":      "bailar es uno de los ejercicios que más activa el cerebro completo",
    "deporte":     "hacer deporte libera endorfinas que mejoran el estado de ánimo de forma natural",
    "futbol":      "el fútbol es el deporte más practicado con más de 250 millones de jugadores",
    "natación":    "nadar ejercita el 90 por ciento de los músculos del cuerpo al mismo tiempo",
    "correr":      "correr 30 minutos al día puede aumentar la esperanza de vida hasta 3 años",
    "fotografía":  "la fotografía entrena el ojo humano para notar detalles que normalmente ignoramos",
    "cantar":      "cantar libera oxitocina, la hormona que genera sensación de bienestar y confianza",
    "escribir":    "escribir a mano activa más áreas del cerebro que escribir en computadora",
}

datos_gustos = {
    "chocolate":  "el chocolate oscuro contiene flavonoides que mejoran el flujo sanguíneo al cerebro",
    "gatos":      "los gatos pueden reconocer la voz de sus dueños pero eligen cuándo hacerles caso",
    "perros":     "los perros pueden oler enfermedades como el cáncer con una precisión del 97 por ciento",
    "pizza":      "la pizza fue nombrada patrimonio cultural inmaterial de la humanidad en 2017",
    "tacos":      "el taco es uno de los platillos más antiguos de México con más de 500 años de historia",
    "café":       "el café es el segundo producto más comercializado en el mundo después del petróleo",
    "anime":      "el anime representa más del 60 por ciento de la animación producida en el mundo",
    "paletas":    "las paletas de hielo fueron inventadas accidentalmente en 1905 por un niño de 11 años",
    "helado":     "el helado fue originalmente un postre exclusivo para reyes y la realeza europea",
    "viajar":     "viajar a nuevos lugares estimula la formación de nuevas conexiones neuronales",
    "dormir":     "dormir consolida los recuerdos del día y elimina toxinas acumuladas en el cerebro",
    "series":     "ver series activa el mismo sistema de recompensa del cerebro que los videojuegos",
}

datos_fechas = {
    "enero":      "en enero se celebra el año nuevo en más de 90 países al mismo tiempo",
    "febrero":    "febrero es el único mes que puede tener 28 o 29 días dependiendo del año",
    "marzo":      "en marzo ocurre el equinoccio de primavera, cuando el día y la noche duran igual",
    "abril":      "abril viene del latín aperire que significa abrir, por las flores que florecen",
    "mayo":       "mayo es el mes con más cumpleaños registrados en el mundo",
    "junio":      "junio tiene los días más largos del año en el hemisferio norte",
    "julio":      "julio fue nombrado así en honor a Julio César por el senado romano",
    "agosto":     "agosto fue nombrado en honor al emperador romano Augusto",
    "septiembre": "septiembre marca el inicio del otoño y la vuelta a clases en muchos países",
    "octubre":    "octubre es el único mes cuyo nombre tiene la misma cantidad de letras en español e inglés",
    "noviembre":  "noviembre es el mes con menos días festivos oficiales en México",
    "diciembre":  "diciembre tiene la noche más larga del año con el solsticio de invierno",
    "1":  "el número 1 es el único número que no es primo ni compuesto en matemáticas",
    "5":  "el 5 de febrero es la fecha en que se promulgó la Constitución Mexicana de 1917",
    "10": "el 10 es considerado el número de la perfección en la numerología antigua",
    "15": "el 15 de septiembre es la noche del grito de independencia en México",
    "16": "el 16 de septiembre es el día de la independencia de México",
}


# Generador de dato curioso combinado

def buscar_dato(texto, diccionario):
    texto = texto.lower()
    for clave, dato in diccionario.items():
        if clave in texto:
            return dato
    return None

def generar_dato_curioso(fecha, pasatiempo, gusto):
    dato_p = buscar_dato(pasatiempo, datos_pasatiempo)
    dato_g = buscar_dato(gusto,      datos_gustos)
    dato_f = buscar_dato(fecha,      datos_fechas)

    # Combinar los que se encontraron
    partes = []
    if dato_p:
        partes.append(dato_p)
    if dato_g:
        partes.append(dato_g)
    if dato_f:
        partes.append(dato_f)

    if len(partes) == 0:
        return (
            f"Qué combinación tan interesante: {fecha}, {pasatiempo} y {gusto}. "
            "Las personas con gustos únicos suelen tener una creatividad fuera de lo común."
        )

    if len(partes) == 1:
        return f"Aquí va tu dato curioso: {partes[0]}."

    if len(partes) == 2:
        return (
            f"Tengo dos datos curiosos para ti. "
            f"Primero: {partes[0]}. "
            f"Y segundo: {partes[1]}."
        )

    return (
        f"Tengo tres datos curiosos para ti. "
        f"Sobre tu pasatiempo: {partes[0]}. "
        f"Sobre lo que te gusta: {partes[1]}. "
        f"Y sobre tu fecha: {partes[2]}."
    )


# Funciones principales

def dar_dato_curioso(fecha, pasatiempo, gusto):
    dato = generar_dato_curioso(fecha, pasatiempo, gusto)
    hablar(dato)

def preguntar_datos():
    hablar("Dime una fecha importante para ti.")
    fecha = escuchar_voz()

    hablar("¿Cuál es tu pasatiempo favorito?")
    pasatiempo = escuchar_voz()

    hablar("Dime algo que te guste mucho.")
    gusto = escuchar_voz()

    return fecha, pasatiempo, gusto
