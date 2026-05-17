from datasets import load_dataset
import json
import os

print("=" * 70)
print("SISTEMA DE DESCARGA DE CORPUS HÍBRIDO")
print("=" * 70)

# =========================================================
# CREAR CARPETA corpus
# =========================================================

CARPETA_CORPUS = "../corpus"

if not os.path.exists(CARPETA_CORPUS):
    os.makedirs(CARPETA_CORPUS)

# =========================================================
# LISTA GENERAL DEL CORPUS
# =========================================================

corpus_total = []

# =========================================================
# FUNCIÓN AUXILIAR
# =========================================================

def agregar_documentos(dataset, fuente, cantidad):

    contador = 0

    for item in dataset:

        try:

            # =================================================
            # EXTRAER TEXTO
            # =================================================

            texto = ""

            posibles_campos = [
                "text",
                "content",
                "article",
                "description",
                "summary"
            ]

            for campo in posibles_campos:
                if campo in item:
                    texto = str(item[campo])
                    break

            if texto == "":
                continue

            # =================================================
            # VALIDAR LONGITUD
            # =================================================

            if len(texto.strip()) < 150:
                continue

            # =================================================
            # EXTRAER TITULO
            # =================================================

            titulo = item.get("title", "SIN_TITULO")

            # =================================================
            # GUARDAR DOCUMENTO
            # =================================================

            documento = {
                "fuente": fuente,
                "titulo": titulo,
                "texto": texto
            }

            corpus_total.append(documento)

            contador += 1

            if contador % 500 == 0:
                print(f"[OK] {contador} documentos de {fuente}")

            if contador >= cantidad:
                break

        except Exception:
            continue

    print(f"[FINALIZADO] {fuente}: {contador} documentos")

# =========================================================
# 1. WIKIPEDIA ESTABLE
# =========================================================

try:

    print("\nDescargando Wikipedia EN...")

    wikipedia = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train[:4000]"
    )

    agregar_documentos(
        wikipedia,
        "wikipedia",
        4000
    )

except Exception as e:
    print("[ERROR] Wikipedia:", e)

# =========================================================
# 2. AG NEWS
# =========================================================

try:

    print("\nDescargando AG News...")

    ag_news = load_dataset(
        "ag_news",
        split="train[:3000]"
    )

    agregar_documentos(
        ag_news,
        "ag_news",
        3000
    )

except Exception as e:
    print("[ERROR] AG News:", e)

# =========================================================
# 3. DBPEDIA
# =========================================================

try:

    print("\nDescargando DBPedia...")

    dbpedia = load_dataset(
        "dbpedia_14",
        split="train[:3000]"
    )

    agregar_documentos(
        dbpedia,
        "dbpedia",
        3000
    )

except Exception as e:
    print("[ERROR] DBPedia:", e)

# =========================================================
# 4. YAHOO ANSWERS
# =========================================================

try:

    print("\nDescargando Yahoo Answers...")

    yahoo = load_dataset(
        "yahoo_answers_topics",
        split="train[:2000]"
    )

    agregar_documentos(
        yahoo,
        "yahoo_answers",
        2000
    )

except Exception as e:
    print("[ERROR] Yahoo Answers:", e)

# =========================================================
# 5. AMAZON REVIEWS
# =========================================================

try:

    print("\nDescargando Amazon Reviews...")

    amazon = load_dataset(
        "amazon_polarity",
        split="train[:2000]"
    )

    agregar_documentos(
        amazon,
        "amazon_reviews",
        2000
    )

except Exception as e:
    print("[ERROR] Amazon Reviews:", e)

# =========================================================
# 6. MULTI NEWS
# =========================================================

try:

    print("\nDescargando Multi News...")

    multinews = load_dataset(
        "multi_news",
        split="train[:1000]"
    )

    agregar_documentos(
        multinews,
        "multi_news",
        1000
    )

except Exception as e:
    print("[ERROR] Multi News:", e)

# =========================================================
# GUARDAR JSON
# =========================================================

ruta_json = os.path.join(
    CARPETA_CORPUS,
    "corpus_total.json"
)

print("\nGuardando corpus híbrido...")

with open(ruta_json, "w", encoding="utf-8") as archivo:

    json.dump(
        corpus_total,
        archivo,
        ensure_ascii=False,
        indent=4
    )

# =========================================================
# RESUMEN FINAL
# =========================================================

print("\n" + "=" * 70)
print("CORPUS HÍBRIDO GENERADO CORRECTAMENTE")
print("=" * 70)

print(f"\nTotal documentos: {len(corpus_total)}")

print(f"\nArchivo generado:")
print(ruta_json)

print("\nSistema listo para preprocessing.")