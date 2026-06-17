import requests
from datetime import datetime
from voz import hablar

# Códigos de clima de Open-Meteo traducidos al español
DESCRIPCIONES_CLIMA = {
    0:  "cielo despejado",
    1:  "principalmente despejado",
    2:  "parcialmente nublado",
    3:  "nublado",
    45: "neblina",
    48: "neblina con escarcha",
    51: "llovizna ligera",
    53: "llovizna moderada",
    55: "llovizna intensa",
    61: "lluvia ligera",
    63: "lluvia moderada",
    65: "lluvia intensa",
    71: "nieve ligera",
    73: "nieve moderada",
    75: "nieve intensa",
    80: "chubascos ligeros",
    81: "chubascos moderados",
    82: "chubascos intensos",
    95: "tormenta eléctrica",
    99: "tormenta eléctrica con granizo",
}

def obtener_clima():
    try:
        # Coordenadas de Gustavo A. Madero, CDMX
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=19.4978&longitude=-99.1269"
            "&current=temperature_2m,weathercode"
            "&timezone=America%2FMexico_City"
        )

        response = requests.get(url, timeout=10)
        data = response.json()

        temperatura = data["current"]["temperature_2m"]
        codigo = data["current"]["weathercode"]
        descripcion = DESCRIPCIONES_CLIMA.get(codigo, "clima variable")

        return temperatura, descripcion

    except Exception as e:
        print("Error al obtener clima:", e)
        return None, None

def decir_fecha_y_clima():
    # Fecha de hoy en español
    meses = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]
    hoy = datetime.now()
    fecha_str = f"{hoy.day} de {meses[hoy.month - 1]} de {hoy.year}"

    hablar(f"Hoy es {fecha_str}.")

    temperatura, descripcion = obtener_clima()

    if temperatura is not None:
        hablar(
            f"En la colonia Gustavo A. Madero, la temperatura actual "
            f"es de {temperatura} grados centígrados con {descripcion}."
        )
    else:
        hablar("No pude obtener el clima en este momento.")
