import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline


training_file = "train.p"
validation_file="valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

#code that has been commented below has been left here for debug purposes
#plt.figure(figsize = (1,1))
#plt.imshow(image)

EPOCHS = 150
BATCH_SIZE = 256

# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_valid)

# Number of testing examples.
n_test = len(X_test)

# dtermine the shape of an traffic sign image?
image_shape = X_train[0].shape

# Number of unique classes/labels there are in the dataset.
n_classes = 43

# Print out all relavant information before processing.
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.


### Feel free to use as many code cells as needed.



### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.




### Feel free to use as many code cells as needed.

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.


### Define your architecture here.
### Training my model here
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation and dropout.
    conv1 = tf.nn.dropout((tf.nn.relu(conv1)), 0.7, noise_shape=None, seed=None, name=None)


    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
   
    

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    tf.nn.dropout((conv2), 0.7, noise_shape=None, seed=None, name=None)
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    tf.nn.dropout((flatten(conv2)), 0.5, noise_shape=None, seed=None, name=None)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.dropout((tf.matmul(fc0, fc1_W) + fc1_b), 0.7, noise_shape=None, seed=None, name=None)
    
    # SOLUTION: Activation & dropout
    #fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout((tf.nn.relu(fc1)), 0.7, noise_shape=None, seed=None, name=None)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation & Dropout
    fc2 = tf.nn.dropout((tf.nn.relu(fc2)), 0.7, noise_shape=None, seed=None, name=None)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

### Calculate and report the accuracy on the training and validation set.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def plot_signs(signs, nrows = 1, ncols=1, labels=None):   
    fig, axs = plt.subplots(ncols, nrows, figsize=(15, 8))
    axs = axs.ravel()
    for index, title in zip(range(len(signs)), signs):
        axs[index].imshow(signs[title])
        axs[index].set_title(labels[index], fontsize=10)
    return()


def normalize_image(image):

    a = 0
    b = 1
    pixel_min = 0
    pixel_max = 255
    normalized_image = ((image - pixel_min)*(b - a))/(pixel_max - pixel_min)
    return normalized_image



### Load the images and plot them here.

sign_text = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('myint','i8'), ('mystring','S55')], delimiter=',')

number_of_images_to_display = 10
signs = {}
labels = {}
for i in range(number_of_images_to_display):
    index = random.randint(0, n_train-1)
    labels[i] = sign_text[y_train[index]][1].decode('ascii')
    signs[i] = X_train[index]    
plot_signs(signs, 5, 2, labels)


# Finding unique elements in train, test and validation arrays
      
train_unique, counts_train = np.unique(y_train, return_counts=True)
plt.bar(train_unique, counts_train)
plt.grid()
plt.title("\nTrain Dataset Unique Sign Counts")
plt.show()

test_unique, counts_test = np.unique(y_test, return_counts=True)
plt.bar(test_unique, counts_test)
plt.grid()
plt.title("Test Dataset Unique Sign Counts")
plt.show()

valid_unique, counts_valid = np.unique(y_valid, return_counts=True)
plt.bar(valid_unique, counts_valid)
plt.grid()
plt.title("Valid Dataset Unique Sign Counts")
plt.show()

#Normalize images
X_train = normalize_image(X_train) 
X_valid = normalize_image(X_valid) 
X_test = normalize_image(X_test)

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
dropout = tf.placeholder(tf.float32)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    
    
    
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

