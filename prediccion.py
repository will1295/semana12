import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("modelo_mnist.h5")

def predecir_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).convert('L').resize((28,28))
    img_array = np.array(img,dtype=np.float32)
    img_array = img_array/255.0
    img_array = 1.0 - img_array
    img_array = np.expand_dims(img_array, axis=0)
    predicciones = model.predict(img_array)
    digito_predicho = np.argmax(predicciones[0])
    precision = np.max(predicciones[0]*100)

    print(f"\El numero es : {digito_predicho}")
    print(f"\Precision del {precision:.2f}%")

predecir_imagen("cero_numero.png")






