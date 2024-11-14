import numpy as np
import matplotlib.pyplot as plt
import sympy
from Perceptron import Perceptron


class HiddenLayer:
    def __init__(self, *, inputs, outputs, no_hidden_layer, num_hidden_nodes, output_layer = False, input_expression = None, multiclass = False ):
        self.inputs = inputs
        self.outputs = outputs
        self.no_hidden_layer = no_hidden_layer
        self.num_hidden_nodes = num_hidden_nodes
        self.output_layer = output_layer
        #Initialise all the variables

        if self.output_layer==False:
            self.activation_func = "relu"
        else:
            if not multiclass:
                self.activation_func = "sigmoid"            
            else:
                self.activation_func = "softmax"
        
        #The output layer has the sigmoid activation function but the rest have RELU 


        self.list_perceptrons = []
        # This is a list of all the Perceptron objects in the hidden layer
        # It will be length of Hidden layer 


        for i in range(self.num_hidden_nodes):
            self.list_perceptrons.append(Perceptron(inputs=self.inputs, outputs=self.outputs, activation_func=self.activation_func, hidden_layer=self.no_hidden_layer, hidden_node=i, input_expression=input_expression))
            self.list_perceptrons[i].calculate_activation()
            self.list_perceptrons[i].calculate_expression()
        

        self.list_activations = np.array([])
        #This is the list of outputs from the perceptrons in the hidden layer

        self.list_expressions = []
        #This is the list of sympy expressions that are outputted from the perceptrons in the hidden layer

        for i in range(len(self.list_perceptrons)):
            if not self.output_layer:
                self.list_activations = np.append(self.list_activations, self.list_perceptrons[i].activation)
                self.list_expressions.append(self.list_perceptrons[i].output_expression)

            else:
                self.pred_outputs = self.list_perceptrons[i].pred_outputs
                self.list_expressions = self.list_perceptrons[i].output_expression
        

        '''
        if self.output_layer == False:
            print(f"List_Activations: {self.list_activations}")
        print(f"List_Expressions: {self.list_expressions}")
        if self.output_layer:
            print(f"Predicted Outputs: {self.pred_outputs}")
        print()
        '''



def main():
    inputs = np.array([0, 1, 1, 2])
    outputs = np.array([1])

    list_hiddenlayer = []
    list_hiddenlayer.append(HiddenLayer(inputs = inputs, outputs = outputs, no_hidden_layer = 1, num_hidden_nodes = 2, input_expression = None))
    list_hiddenlayer.append(HiddenLayer(inputs = list_hiddenlayer[0].list_activations, outputs = outputs, no_hidden_layer = 2, num_hidden_nodes = 1, input_expression = list_hiddenlayer[0].list_expressions))

   


if __name__ == "__main__":
    main()