from keras.models import load_model

# Load the saved model from file
loaded_model = load_model('my_model.h5')

predictions = loaded_model.predict([40.06, 60.15, 20.64])

print(predictions)