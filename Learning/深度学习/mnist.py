#Liangyz
#2024/6/6  1:20

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from 神经网络 import MultilayerPerceptron

# Load the data
data = pd.read_csv(r'E:\Project\data\mnist-demo.csv')
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))

# Display the first numbers_to_display numbers
plt.figure(figsize=(10, 10))
for plot_index in range(numbers_to_display):
    digit = data[plot_index:plot_index + 1].values
    digit_label = digit[0][0]
    digit_pixels = digit[0][1:]
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    frame = digit_pixels.reshape((image_size, image_size))
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.savefig('mnist-demo.png')
plt.show()


# Prepare the data
train_data = data.sample(frac=0.8, random_state=200)
test_data = data.drop(train_data.index).values
train_data = train_data.values

num_training_samples = train_data.shape[0]

X_train= train_data[:num_training_samples,1:]
y_train = train_data[:num_training_samples,[0]]

X_test = test_data[:,1:]
y_test = test_data[:,[0]]

layers = [784, 32, 10]

nomalize_data = True
max_iteration = 1000
alpha = 0.1

mlp = MultilayerPerceptron(X_train, y_train, layers, normalize_data=nomalize_data)
(theta, costs) = mlp.train(max_iteration=max_iteration, alpha=alpha)

plt.plot(range(len(costs)), costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')
plt.savefig('mnist-cost10000.png')
plt.show()

y_train_predicted = mlp.predict(X_train)
y_test_predicted = mlp.predict(X_test)

train_accuracy = np.mean(y_train_predicted == y_train) * 100
test_accuracy = np.mean(y_test_predicted == y_test) * 100

print(f'Training accuracy: {train_accuracy}%')
print(f'Test accuracy: {test_accuracy}%')

numbers_to_display = 64

num_cells = math.ceil(math.sqrt(numbers_to_display))

plt.figure(figsize=(15, 15))

for plot_index in range(numbers_to_display):
    digit_label = y_test[plot_index][0]
    digit_pixels = X_test[plot_index]

    predicted_label = y_test_predicted[plot_index][0]

    image_size = int(math.sqrt(digit_pixels.shape[0]))

    frame = digit_pixels.reshape((image_size, image_size))

    color_map = 'Greens' if predicted_label == digit_label else 'Reds'
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap=color_map)
    plt.title(f'Predicted: {predicted_label}')
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.savefig('mnist-predictions10000.png')
plt.show()

