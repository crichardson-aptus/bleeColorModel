from keras.models import load_model

# Load the saved model from file
loaded_model = load_model('my_model.h5')

new_inputs = [[32.68, -6.21, -19.36], [40.06, 60.15, 20.64], [7.0, 8.0, 9.0]]
predictions = loaded_model.predict(new_inputs)
x = predictions[0]
R = round(x[0])
Y = round(x[1])
B = round(x[2])
W = round(x[3])
M = round(x[4])
X = round(x[5])
U = round(x[6])
K = round(x[7])
I = round(x[8])
P = round(x[9])
# L = round(x[10])

print("R:{} Y:{} B:{} W:{} M:{} X:{} U:{} K:{} I:{} P:{}".format(R,Y,B,W,M,X,U,K,I,P))