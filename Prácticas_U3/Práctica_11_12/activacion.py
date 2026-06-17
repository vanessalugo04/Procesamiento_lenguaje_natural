from voz import hablar, escuchar_voz

# NOMBRE_EQUIPO = "equipo"   # ← cambia aquí el nombre de tu equipo

def detectar_activacion():
    while True:
        texto = escuchar_voz(3)
        print("Detectado:", texto)

        if "rafita" in texto:
            hablar(
                "Hola Equipo 3. Describe tu día de hoy."
            )

            return

            

