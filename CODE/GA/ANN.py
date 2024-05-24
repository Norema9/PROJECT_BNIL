import numpy

""" This code is un update of the code found here:
         ref: https://github.com/ahmedfgad/NeuralGenetic/blob/master/Tutorial%20Project/ANN.py
"""



def sigmoid(inpt):
    return 1.0/(1.0+numpy.exp(-1*inpt))

def relu(inpt):
    result = inpt
    result[inpt<0] = 0
    return result

def binary_crossentropy(y_true, y_pred):
    # Clip predictions to avoid log(0) errors
    y_pred = numpy.clip(y_pred, 1e-15, 1 - 1e-15)
    # Calculate binary cross-entropy
    return -numpy.mean(y_true * numpy.log(y_pred) + (1 - y_true) * numpy.log(1 - y_pred))

def predict_outputs(weights_mat:dict, data_inputs, data_outputs, activation = "relu"):
    threshold = 0.5
    predictions = numpy.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for layer_key, curr_weights in weights_mat.items():
            r1 = numpy.matmul(r1, curr_weights)
            if layer_key != "output_layer":
                if activation == "relu":
                    r1 = relu(r1)
                elif activation == "sigmoid":
                    r1 = sigmoid(r1)
            else:
                r1 = sigmoid(r1)
        predicted_label = 0 if r1 < threshold else 1
        predictions[sample_idx] = predicted_label
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    accuracy = (correct_predictions/data_outputs.size)*100
    loss = binary_crossentropy(data_outputs, predictions)
    return loss, accuracy, predictions
    
def fitness(weights_mat, data_inputs, data_outputs, activation="relu"):
    accuracy = numpy.empty(shape = len(weights_mat))
    losses = numpy.empty(shape = len(weights_mat))
    for sol_idx in range(len(weights_mat)):
        curr_sol_mat = weights_mat[sol_idx]
        losses[sol_idx], accuracy[sol_idx], _ = predict_outputs(curr_sol_mat, data_inputs, data_outputs, activation=activation)
    return losses, accuracy
