import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.chdir("C:/Users/Satyam/Desktop/Programs/Datasets/Digit_recognition")
training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


def interference(x,weight,bias):
    return tf.add(tf.matmul(x,weight),bias)


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

    
#Splitting the testing data into X and Y.
X_data = training_data.iloc[:20000,1:]
y_data = training_data.iloc[:20000,0]
y_check = training_data.iloc[:20000,0]
x_check = training_data.iloc[10000:,1:]

def modify_y_data(y_data): #Modifies Y to readable code. Like 3 gets converted to [0,0,0,1,0,0,0,0,0,0]
    answers = []
    for i in range(len(y_data)):
        temp = [0.0] * 10
        temp[y_data[i]] = 1.0
        answers.append(temp)   
    return answers

y_data = pd.DataFrame(modify_y_data(y_data))
y_check = pd.DataFrame(modify_y_data(y_check))

#Setting the max number of rows to be displayed at each time.
pd.options.display.max_rows = 5

#Some basic stuff
number_of_attributes = len(X_data.columns)
number_of_samples = len(X_data)

X = tf.placeholder(dtype = tf.float32, shape = [None,number_of_attributes])
y_true = tf.placeholder(dtype = tf.float32, shape = [None,10])
weights1 = tf.Variable(tf.random_normal([number_of_attributes,25],stddev=0.5),name = 'weights1')  #connecting layer1 to layer2
bias1 = tf.Variable(tf.zeros(1),name = 'bias1') #bias in the first layer
weights2 = tf.Variable(tf.random_normal([25,10], stddev = 0.5),name = 'weights2') #connecting layer2 to layer3
bias2 = tf.Variable(tf.zeros(1),name = 'bias2') #bias in second layer.
layer2 = interference(X,weights1,bias1)
layer2 = tf.sigmoid(layer2)
layer3 = interference(layer2,weights2,bias2)
y_predict = tf.sigmoid(layer3)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits = y_predict))
learning_rate = 3.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)
eval_op = evaluate(y_predict,y_true)
losses = []
accuracies = []
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    dict = {X:X_data,y_true:y_data}
    for i in range(200):
        output = sess.run(loss,feed_dict = dict)
        sess.run(train,feed_dict = dict)
        print("Loss = ", output)
        losses.append(output)
        acc = sess.run(eval_op,feed_dict = dict)
        print("Accuracy = ",acc) 
        accuracies.append(acc)
    

plt.subplot(1,2,1)
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Number of Epochs")
plt.subplot(1,2,2)
plt.plot(accuracies)
plt.ylabel("Accuracy")
plt.xlabel("Number of Epochs")