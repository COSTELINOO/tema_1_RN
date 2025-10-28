import pickle
import pandas as pd
import numpy as np


train_file = "kaggle/input/fii-nn-2025-homework-2/extended_mnist_train.pkl"
test_file = "kaggle/input/fii-nn-2025-homework-2/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)


train_data = []
train_labels = []

for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = [image.flatten() for image, _ in test]

input_matrix = np.array(train_data)
label_vector = np.array(train_labels)
test_matrix = np.array(test_data)


input_matrix = input_matrix / 255.0
test_matrix = test_matrix / 255.0


def weighted_sum(X, W, b):
    return np.dot(X, W) + b

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy_loss(Y_true, y_predicted):
    return -np.sum(Y_true * np.log(y_predicted + 1e-8))

def init_label_matrix(y, num_classes=10):
    Y = np.zeros((y.size, num_classes))
    for index,i in enumerate (Y):
        Y[index][y[index]]=1
    return Y

def update_weights(X, Y_true, y_predicted, W, b, learning_rate):

    m = X.shape[0]
    error = Y_true - y_predicted
    dW = np.dot(X.T, error) / m
    db = np.sum(error, axis=0) / m

    W += learning_rate * dW
    b += learning_rate * db
    return W, b


num_inputs = 784
num_outputs = 10
learning_rate = 0.8
iterari = 15000


np.random.seed(42)
W = np.random.randn(num_inputs, num_outputs) * 0.01
b = np.zeros((num_outputs,))

print(label_vector)
label_matrix = init_label_matrix(label_vector, num_outputs)

accuracy=0
for iterare in range(iterari):
    if accuracy>=0.9601:
        break

    Z = weighted_sum(input_matrix, W, b)
    y_predicted = softmax(Z)

    loss = cross_entropy_loss(label_matrix, y_predicted)

    W, b = update_weights(input_matrix, label_matrix, y_predicted, W, b, learning_rate)

    pred_labels = np.argmax(y_predicted, axis=1)
    acc = np.mean(pred_labels == label_vector)
    if(iterare%100==0):
        print(f"Iteration  {iterare+1}/{iterari} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")



Z_test = weighted_sum(test_matrix, W, b)
Y_test_pred = softmax(Z_test)
predictions = np.argmax(Y_test_pred, axis=1)


predictions_csv = {
    "ID": list(range(len(predictions))),
    "target": predictions.tolist()
}

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)