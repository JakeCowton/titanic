# The MIT License (MIT)

# Copyright (c) 2014 Jake Cowton

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from perceptron import Perceptron

# Lowest MSE
LMSE = 0.001

# Because there are 3 classes
NORMALISATION_VALUE = 1/3.0

def normalise(data):
    """
    Turn data into values between 0 and 1
    @param data list of lists of input data and output e.g.
        [
            [[0,0,255], 1],
            ...
        ]
    @returns Normalised training data
    """
    temp_list = []
    for entry in data:
        entry_list = []
        for value in entry[0]:
            entry_list.append(float(value*NORMALISATION_VALUE))
        temp_list.append([entry_list, entry[1]])
    return temp_list

def create_slp(inputs, outputs):

    # Create the perceptron
    p = Perceptron(len(inputs))

    # Number of full iterations
    epochs = 0

    # Instantiate mse for the loop
    mse =999

    while (abs(mse-LMSE) > 0.002 and epochs < 500):

        # Epoch cumulative error
        error = 0

        # For each set in the training_data
        for i in range(len(inputs)):

            # Calculate the result
            output = p.result(inputs[i])

            # Calculate the error
            iter_error = outputs[i] - output

            # Add the error to the epoch error
            error += iter_error

            # Adjust the weights based on inputs and the error
            p.weight_adjustment(inputs[i], iter_error)

        # Calculate the MSE - epoch error / number of sets
        mse = float(error/len(inputs))

        # Increment the epoch number
        epochs += 1

    return p
