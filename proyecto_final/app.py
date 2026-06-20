import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Importamos los módulos del preprocesamiento y plagio
from backend.Preprocesamiento import pipeline_procesamiento
from backend.Plagio import pipeline_plagio_hibrido

# importamos el mod del clasificador de tema
from backend.Clasificador import modelo_tema, vector_unico, matriz_bag_words

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
            
            # 3. DETECCIÓN DE PLAGIO GLOBAL
            # El nuevo pipeline de BERT devuelve un diccionario, no una lista.
            resultado_plagio = pipeline_plagio_hibrido(
                tokens_usuario=tokens_usuario,
                ruta_corpus="corpus/corpus_procesado.json",
                top_n_candidatos=100,
                top_n_preliminar=300,
                tamano_fragmento=120,
                top_fragmentos=3,
                usar_tema=True
            )

            # Solo mostramos el plagio global en la app.
            porcentaje_plagio = resultado_plagio["probabilidad_plagio"]
            
            # Detección de IA (Este lo dejamos estático por ahora para tu siguiente fase)
            porcentaje_ia = 12.0 #TRABAJO A FUTUROOOOO LOL 
            
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