import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Importamos tu pipeline de la carpeta backend (en minúsculas)
from backend.Preprocesamiento import pipeline_procesamiento

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
            resultado_prepro = pipeline_procesamiento(ruta_guardado)
            
            if not resultado_prepro["exito"]:
                return jsonify({"exito": False, "error": resultado_prepro["error"]}), 400
            
            tokens_usuario = resultado_prepro["resultado_lematizado"]
            
            # Simulación de porcentajes por ahora
            porcentaje_plagio = 24.5
            porcentaje_ia = 12.0
            
            return jsonify({
                "exito": True,
                "nombre_archivo": filename,
                "total_tokens": len(tokens_usuario),
                "plagio": porcentaje_plagio,
                "ia": porcentaje_ia,
                "mensaje": "Análisis completado"
            })
            
        except Exception as e:
            return jsonify({"exito": False, "error": f"Error interno: {str(e)}"}), 500
        finally:
            if os.path.exists(ruta_guardado):
                os.remove(ruta_guardado)
                
    return jsonify({"exito": False, "error": "Solo se aceptan PDFs."}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)