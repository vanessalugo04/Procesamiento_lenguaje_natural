import urllib.request
import json
from voz import hablar, escuchar_voz


# ──────────────────────────────────────────────
# Llamada a la API de Anthropic
# ──────────────────────────────────────────────
def generar_dato_curioso_api(fecha, pasatiempo, gusto):
    """Llama a claude-sonnet-4-6 y pide UN dato curioso relacionado
    con los tres datos del usuario. Devuelve el texto o None si falla."""

    prompt = (
        f"El usuario mencionó: fecha '{fecha}', "
        f"pasatiempo '{pasatiempo}' y algo que le gusta '{gusto}'. "
        "Dame UN solo dato curioso breve (máximo 2 oraciones) que relacione "
        "esos tres elementos de forma creativa. "
        "Responde SOLO con el dato curioso, sin introducción ni comillas."
    )

    cuerpo = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=cuerpo,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            datos = json.loads(resp.read().decode("utf-8"))
            return datos["content"][0]["text"].strip()
    except Exception as e:
        print("Error al llamar a la API:", e)
        return None


# ──────────────────────────────────────────────
# Función principal que habla el dato curioso
# ──────────────────────────────────────────────
def dar_dato_curioso(fecha, pasatiempo, gusto):
    dato = generar_dato_curioso_api(fecha, pasatiempo, gusto)

    if dato:
        hablar(f"Aquí va un dato curioso para ti: {dato}")
    else:
        # Respaldo si la API falla
        hablar(
            f"Qué interesante que te guste {gusto} y que tu pasatiempo sea {pasatiempo}. "
            "Las personas con gustos variados suelen ser muy creativas."
        )


# ──────────────────────────────────────────────
# Preguntas al usuario
# ──────────────────────────────────────────────
def preguntar_datos():
    hablar("Dime una fecha importante para ti.")
    fecha = escuchar_voz()

    hablar("¿Cuál es tu pasatiempo favorito?")
    pasatiempo = escuchar_voz()

    hablar("Dime algo que te guste mucho.")
    gusto = escuchar_voz()

    return fecha, pasatiempo, gusto
