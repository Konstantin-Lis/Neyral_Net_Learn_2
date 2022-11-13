import numpy
import cv2
import time
import math

# neural network class definition
class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih1 = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.wh1h2 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.wh2o = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

        # learning rate
        self.lr = learningrate

        # activation function is Linear
        self.activation_function = lambda x: x
        self.differential = lambda x: 1

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into 1-st hidden layer
        hidden_1_inputs = numpy.dot(self.wih1, inputs)
        # calculate the signals emerging from 1-st hidden layer
        hidden_1_outputs = self.activation_function(hidden_1_inputs)

        # calculate signals into 2-nd hidden layer
        hidden_2_inputs = numpy.dot(self.wh1h2, hidden_1_outputs)
        # calculate the signals emerging from 2-nd hidden layer
        hidden_2_outputs = self.activation_function(hidden_2_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wh2o, hidden_2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_2_errors = numpy.dot(self.wh2o.T, output_errors)
        hidden_1_errors = numpy.dot(self.wh1h2.T, hidden_2_errors)

        # update the weights for the links between the hidden and output layers
        self.wh2o += self.lr * numpy.dot((output_errors * self.differential(final_outputs)), numpy.transpose(hidden_2_outputs))

        self.wh1h2 += self.lr * numpy.dot((hidden_2_errors * self.differential(hidden_2_outputs)), numpy.transpose(hidden_1_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih1 += self.lr * numpy.dot((hidden_1_errors * self.differential(hidden_1_outputs)), numpy.transpose(inputs))

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into 1-st hidden layer
        hidden_1_inputs = numpy.dot(self.wih1, inputs)
        # calculate the signals emerging from 1-st hidden layer
        hidden_1_outputs = self.activation_function(hidden_1_inputs)

        # calculate signals into 2-nd hidden layer
        hidden_2_inputs = numpy.dot(self.wh1h2, hidden_1_outputs)
        # calculate the signals emerging from 2-nd hidden layer
        hidden_2_outputs = self.activation_function(hidden_2_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wh2o, hidden_2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# number of input, hidden and output nodes
input_nodes = 192
hidden_nodes_1 = 100
hidden_nodes_2 = 100
output_nodes = 64

# learning rate is 0.1
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes, learning_rate)


# beginning of time count
time0 = time.time()

#training network
data_image = cv2.imread('img_1.jpg')

epochs = 1
for e in range(epochs):
    for i in range(100):
        for j in range(100):
            for k in range(3):
                targets = []
                for x in range(8):
                    for y in range(8):
                        targets.append(data_image[i+4+x, j+4+y][k]/255)
                inputs = []
                for x in range(16):
                    for y in range(16):
                        if x not in range(4, 12) or y not in range(4, 12):
                            inputs.append(data_image[i+x, j+y][k]/255)
                n.train(inputs, targets)

# end of learning
time1 = time.time()

# save into files
f = open('wih1.txt', 'w')
for i in range(100):
    s = ''
    for j in n.wih1[i]:
        s += str(j)
        s += ' '
    f.write(s)
    f.write('/n')
f.close()

f = open('wh1h2.txt', 'w')
for i in range(100):
    s = ''
    for j in n.wh1h2[i]:
        s += str(j)
        s += ' '
    f.write(s)
    f.write('/n')
f.close()

f = open('wh20.txt', 'w')
for i in range(64):
    s = ''
    for j in n.wh2o[i]:
        s += str(j)
        s += ' '
    f.write(s)
    f.write('/n')
f.close()

#end of saving
time2 = time.time()

print("Learning time: ", time1-time0, ' seconds')
print("Saving time: ", time2-time1, ' seconds')
print("Whole time: ", time2-time0, 'seconds')