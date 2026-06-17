import pyttsx3
import speech_recognition as sr

def hablar(texto):
    print("Rafa Polinesio:", texto)
    engine = pyttsx3.init()
    engine.say(texto)
    engine.runAndWait()
    engine.stop()

def escuchar_voz(tiempo=5):

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Escuchando...")

        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(
            source,
            phrase_time_limit=tiempo
        )

    try:

        texto = recognizer.recognize_google(
            audio,
            language="es-MX"
        )

        print("Usuario:", texto)

        return texto.lower()

    except (sr.UnknownValueError, sr.RequestError):

        return ""
