from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

# print(x_test.shape)
# print(x_train.shape)
print(x_test[0].shape)
