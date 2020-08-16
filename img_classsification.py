from keras.models import load_model


model = load_model("/home/alpha/Desktop/PreNeu/Office Day/13 Aug/nlp-rnd/keras weights/mnist_cnn.h5")
model.summary()
load_input = model.input
input_shape= list(load_input.shape)

height = int(input_shape[1])
width = int(input_shape[2])
channel = int(input_shape[3])
print(height, width, channel)
