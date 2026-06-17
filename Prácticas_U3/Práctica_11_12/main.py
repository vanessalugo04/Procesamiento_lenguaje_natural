import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from voz import *
from activacion import *
from sentimientos import *
from datos_curiosos_bueno import *
# from datos_curiosos import *
from cuento import *
from clima import *

def main():

    detectar_activacion()

    descripcion = escuchar_voz(5)

    sentimiento = analizar_sentimiento(descripcion)

    responder_sentimiento(sentimiento)

    fecha, pasatiempo, gusto = preguntar_datos()

    dar_dato_curioso(fecha, pasatiempo, gusto)

    cuento()

    decir_fecha_y_clima()

if __name__ == "__main__":
    main()
