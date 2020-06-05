

# Load keras model from disk
model = load_model('./model/saved_model.pb')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert the model
tflite_model = converter.convert()

# Create the tflite model file
with open('model/cells.tflite', "wb") as f:
    f.write(tflite_model)