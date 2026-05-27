import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Importamos tus módulos personalizados
from backend.Preprocesamiento import pipeline_procesamiento
from backend.Clasificador import matriz_bag_words, NaiveBayesDesdeCero
<<<<<<< HEAD
=======
from backend.Plagio import pipeline_plagio_hibrido
>>>>>>> 1146bacec6af451e3f5d59d5cd4cbdc6e1220ef8

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================================================================
# ENTRENAMIENTO SIMULADO DEL CLASIFICADOR (Vocabulario base de desarrollo)
# =========================================================================
# 1. Vocabulario global con palabras clave muy específicas por cada área semántica
vector_unico = [
    # Astronomía
    "planeta", "galaxia", "telescopio",
    # Gastronomía
    "receta", "ingrediente", "cocinar",
    # Finanzas
    "banco", "inflacion", "inversion",
    # Medicina
    "sintoma", "paciente", "medico",
    # Derecho
    "ley", "abogado", "tribunal",
    # Música
    "acorde", "melodia", "instrumento",
    # Deportes
    "balon", "entrenar", "competencia",
    # Botánica
    "planta", "clorofila", "raiz",
    # Arqueología
    "fosil", "ruina", "antiguo",
    # Computación
    "codigo", "software", "servidor", "inteligencia", "artificial"
]

# 2. Corpus de entrenamiento con dos mini-documentos de ejemplo por cada uno de los 10 temas
corpus_entrenamiento = [
    ["planeta", "galaxia", "telescopio"], ["planeta", "telescopio"], # Astronomía
    ["receta", "ingrediente", "cocinar"], ["receta", "cocinar"],     # Gastronomía
    ["banco", "inflacion", "inversion"], ["banco", "inversion"],     # Finanzas
    ["sintoma", "paciente", "medico"], ["sintoma", "medico"],        # Medicina
    ["ley", "abogado", "tribunal"], ["ley", "tribunal"],             # Derecho
    ["acorde", "melodia", "instrumento"], ["acorde", "instrumento"], # Música
    ["balon", "entrenar", "competencia"], ["balon", "entrenar"],     # Deportes
    ["planta", "clorofila", "raiz"], ["planta", "raiz"],             # Botánica
    ["fosil", "ruina", "antiguo"], ["ruina", "antiguo"],             # Arqueología
    ["codigo", "software", "servidor"], ["codigo", "software"]       # Computación
]

# 3. Las 10 etiquetas duplicadas correspondientes al corpus (2 documentos por tema)
y_entrenamiento = [
    "Astronomia", "Astronomia",
    "Gastronomia", "Gastronomia",
    "Finanzas", "Finanzas",
    "Medicina", "Medicina",
    "Derecho", "Derecho",
    "Musica", "Musica",
    "Deportes", "Deportes",
    "Botanica", "Botanica",
    "Arqueologia", "Arqueologia",
    "Computacion", "Computacion"
]

# Instanciar y entrenar nuestro modelo matemático propio desde el inicio
X_train = matriz_bag_words(corpus_entrenamiento, vector_unico)
modelo_tema = NaiveBayesDesdeCero()
modelo_tema.fit(X_train, y_entrenamiento)
# =========================================================================

def file_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('Prueba.html')

@app.route('/analizar', methods=['POST'])
def analizar_documento():
    if 'archivo' not in request.files:
        return jsonify({"exito": False, "error": "No se seleccionó ningún archivo"}), 400
    
    file = request.files['archivo']
    
    if file.filename == '':
        return jsonify({"exito": False, "error": "Nombre de archivo vacío"}), 400
    
    if file and file_allowed(file.filename):
        filename = secure_filename(file.filename)
        ruta_guardado = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(ruta_guardado)
        
        try:
            # 1. Pipeline de preprocesamiento (Incluye la capa de Regex)
            resultado_prepro = pipeline_procesamiento(ruta_guardado)
            
            if not resultado_prepro["exito"]:
                return jsonify({"exito": False, "error": resultado_prepro["error"]}), 400
            
            tokens_usuario = resultado_prepro["resultado_lematizado"]
            primeros_10_tokens = tokens_usuario[:10]
            
            # 2. CLASIFICACIÓN DEL TEMA REAL DESDE CERO
            # Sacamos el vector BoW del texto subido por el usuario
            X_usuario = matriz_bag_words([tokens_usuario], vector_unico)
            tema_detectado = modelo_tema.predict(X_usuario)[0]
            
            # Porcentajes de prueba (Siguientes módulos)
<<<<<<< HEAD
            porcentaje_plagio = 24.5
            porcentaje_ia = 12.0
=======
            #porcentaje_plagio = 24.5
            resultados_plagio = pipeline_plagio_hibrido(tokens_usuario, ruta_corpus="corpus/corpus_procesado.json")
            
            # Tomamos el porcentaje del documento que haya salido con mayor similitud (el top 1)
            if len(resultados_plagio) > 0:
                porcentaje_plagio = resultados_plagio[0]["probabilidad_plagio"]
            else:
                porcentaje_plagio = 0.0
            
            # Detección de IA (Este lo dejamos estático por ahora para tu siguiente fase)
            porcentaje_ia = 12.0 #TRABAJO A FUTUROOOOO LOL 
>>>>>>> 1146bacec6af451e3f5d59d5cd4cbdc6e1220ef8
            
            return jsonify({
                "exito": True,
                "nombre_archivo": filename,
                "total_tokens": len(tokens_usuario),
                "primeros_tokens": primeros_10_tokens,
                "tema_predicho": tema_detectado,  # <-- Enviamos la predicción real
                "plagio": porcentaje_plagio,
                "ia": porcentaje_ia,
                "mensaje": "Análisis completado de forma exitosa."
            })
            
        except Exception as e:
            return jsonify({"exito": False, "error": f"Error interno: {str(e)}"}), 500
            
        finally:
            if os.path.exists(ruta_guardado):
                os.remove(ruta_guardado)
                
    return jsonify({"exito": False, "error": "Solo se aceptan PDFs."}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)