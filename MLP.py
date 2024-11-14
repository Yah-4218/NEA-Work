import numpy as np
import matplotlib.pyplot as plt
import sympy
from HiddenLayer import HiddenLayer


class MultiLayerPerceptron:
    def __init__(self, *, inputs, outputs, size, learning_rate):
        self.inputs = inputs
        self.outputs = outputs
        self.size = size
        self.learning_rate = learning_rate

        if len(self.size) < 1:
            raise ValueError("Cannot have a model of less than 1 hidden layers")
        
        self.list_hiddenlayers = []
        #This is a list of the Hidden Layer class which will have a length of size
        
        #print(f"Size of MLP: {size}")

        for i in range(len(size)+1):
            if i == 0:
                self.list_hiddenlayers.append(HiddenLayer(inputs = self.inputs, outputs=self.outputs, no_hidden_layer = i+1, num_hidden_nodes=size[i], output_layer=False, input_expression=None))
            elif 0 < i < len(size):
                self.list_hiddenlayers.append(HiddenLayer(inputs = self.list_hiddenlayers[i-1].list_activations, outputs=self.outputs, no_hidden_layer = i+1, num_hidden_nodes=size[i], output_layer=False, input_expression=self.list_hiddenlayers[i-1].list_expressions))
            
            elif i == len(size):    
                if len(outputs) == 1:
                    self.list_hiddenlayers.append(HiddenLayer(inputs = self.list_hiddenlayers[i-1].list_activations, outputs=self.outputs, no_hidden_layer = i+1, num_hidden_nodes=1, output_layer=True, input_expression=self.list_hiddenlayers[i-1].list_expressions, multiclass=False))
                elif len(outputs) > 1:
                    if len(outputs)!=size[-1]:
                        raise ValueError("The output layer, producing probabilities requires last layer to be the same size as the output vector")
                    self.list_hiddenlayers.append(HiddenLayer(inputs = self.list_hiddenlayers[i-1].list_activations, outputs=self.outputs, no_hidden_layer = i+1, num_hidden_nodes=1, output_layer=True, input_expression=self.list_hiddenlayers[i-1].list_expressions, multiclass = True))
            
        #This is adding all the hidden layers to the list_hiddenlayers with respects to their own layers(1st layer and output layer have slightly different parameters)


    def CrossEntropyLoss(self):
        self.list_symbol_theta = []
        self.list_thetas = []
        self.inputs = np.insert(self.inputs, 0, 1)
        for hl in range(len(self.list_hiddenlayers)):
            for p in range(len(self.list_hiddenlayers[hl].list_perceptrons)):
                for t in range(len(self.list_hiddenlayers[hl].list_perceptrons[p].symbol_theta)):
                    self.list_thetas.append(self.list_hiddenlayers[hl].list_perceptrons[p].theta[t])
                    self.list_symbol_theta.append(self.list_hiddenlayers[hl].list_perceptrons[p].symbol_theta[t])
        
        
        self.symbol_x = [sympy.Symbol(f"x{i}") for i in range(len(self.inputs))]
                
        logloss, val_logloss = None, None
        for m in range(self.size[-1]):
            if self.size[-1]==1:
                logloss = sympy.log(self.list_hiddenlayers[-1].list_expressions)*self.outputs[m]
            else:
                if not logloss:
                    logloss = sympy.log(self.list_hiddenlayers[-1].list_expressions[m])*self.outputs[m]
                else:
                    logloss += sympy.log(self.list_hiddenlayers[-1].list_expressions[m])*self.outputs[m]
        logloss = -1*logloss
        self.loss = logloss
        #print(f"Logloss: {self.loss}")
        
        val_logloss = logloss
        for l in range(len(self.list_symbol_theta)):
            val_logloss = val_logloss.subs(self.list_symbol_theta[l], self.list_thetas[l])
        for l in range(len(self.symbol_x)):
            val_logloss = val_logloss.subs(self.symbol_x[l], self.inputs[l])
        self.val_loss = val_logloss
        print(f"Val_Loss: {self.val_loss}")

       # print("CROSSENTROPY DONE")


        
    


    def BackPropogation(self):
      
        #print(f"List_thetas: {self.list_thetas}")
        #print(f"List_symbol_theta {self.list_symbol_theta}")

        for i in range(len(self.size)):
            for j in range(len(self.list_hiddenlayers[i].list_perceptrons)):
                for k in range(len(self.list_hiddenlayers[i].list_perceptrons[j].symbol_theta)):
                    der = sympy.diff(self.loss, self.list_hiddenlayers[i].list_perceptrons[j].symbol_theta[k])
                    #print(f"{self.list_hiddenlayers[i].list_perceptrons[j].symbol_theta[k]}, {self.list_hiddenlayers[i].list_perceptrons[j].theta[k]}, {der}")
                    #print(f"Old Der: {der}")
                    for l in range(len(self.list_thetas)):
                        der = der.subs(self.list_symbol_theta[l], self.list_thetas[l])
                    for l in range(len(self.inputs)):
                        der = der.subs(self.symbol_x[l], self.inputs[l])
                        #diff = diff.subs(self.list_thetas[i], self.list_thetas[])
                    #print(f"New Der:{der}")
                    #print(f"Old Theta: {self.list_hiddenlayers[i].list_perceptrons[j].theta[k]}")
                    self.list_hiddenlayers[i].list_perceptrons[j].theta[k] = self.list_hiddenlayers[i].list_perceptrons[j].theta[k] - der*self.learning_rate
                    #print(f"New Theta: {self.list_hiddenlayers[i].list_perceptrons[j].theta[k]}\n")

        #print("BACKPROPOGATION DONE")

                    
                    
        





def training(*, inputs, outputs, epochs):
    mlp = MultiLayerPerceptron(inputs=inputs, outputs=outputs, size = (2, 1), learning_rate=0.1)
    mlp.CrossEntropyLoss()


    mlp.BackPropogation()

    for a in range(10000):

        for i in range(len(mlp.list_hiddenlayers)):
            for j in range(len(mlp.list_hiddenlayers[i].list_perceptrons)):
                mlp.list_hiddenlayers[i].list_perceptrons[j].calculate_activation
                mlp.list_hiddenlayers[i].list_perceptrons[j].calculate_expression

        mlp.CrossEntropyLoss()
        mlp.BackPropogation()
        if mlp.val_loss <0.00001:
            break
        if a ==10000:
            print("Solution not achieved")
        
    print(f"Val_Loss: {mlp.val_loss}")
    print(f"List_thetas: {mlp.list_thetas}")
    print(f"Predicted Output: {mlp.list_hiddenlayers[-1].pred_outputs}")
    print(f"Actual output: {mlp.outputs}")


def main():
    inputs = np.array([[0, 1, 2],[1, 2, 3], [2, 3, 4], [1,2,3]])
    outputs = np.array([1, 2, 3])
    epochs = 50
    print(np.shape(inputs))
    print(inputs[0])
    #training(inputs = inputs, outputs = outputs, epochs = epochs)





if __name__=="__main__":
    main()