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
    try:
            # 1. Ejecutar tu pipeline de preprocesamiento real
            resultado_prepro = pipeline_procesamiento(ruta_guardado)
            
            if not resultado_prepro["exito"]:
                return jsonify({"exito": False, "error": resultado_prepro["error"]}), 400
            
            # Obtener la lista de todos los tokens lematizados del PDF
            tokens_usuario = resultado_prepro["resultado_lematizado"]
            
            # Tomar únicamente los primeros 10 tokens reales
            primeros_10_tokens = tokens_usuario[:10]
            
            # Porcentajes de prueba (temporales)
            porcentaje_plagio = 24.5
            porcentaje_ia = 12.0
            
            # ENVIAMOS LOS DATOS REALES AL FRONTEND
            return jsonify({
                "exito": True,
                "nombre_archivo": filename,
                "total_tokens": len(tokens_usuario),
                "primeros_tokens": primeros_10_tokens,  # <-- Enviamos el arreglo real
                "plagio": porcentaje_plagio,
                "ia": porcentaje_ia,
                "mensaje": "Análisis completado"
            })


if __name__ == '__main__':
    app.run(debug=True, port=5000)