import numpy as np
import matplotlib.pyplot as plt
import sympy
class Perceptron: 
    def __init__(self, *, inputs, outputs, activation_func= "RELU", hidden_layer, hidden_node, input_expression = None):
        '''
            inputs: np.array of inputs in form [1, x1, x2, ..., xn]
            outputs: np.array of true outputs in form [y1, y2, ..., yj]
                One for each training example
            activation_function = the non-linear function that will be used to transform the dot product of theta and inputs
            theta = np.array of parameters that represent the theta of "inputs" in form [theta0, theta1, theta2, ..., thetan]
            activation = output of perceptron
            expression = sympy expression which will be used in backpropogation
            hidden_layer = Which hidden layer this neuron is from
            hidden_node = which node of the hidden layer is this neuron from
            input_expression = the sympy expression of the inputs which will be used for backpropogation

            output_expression = the sympy expression that will be outputted as the inputs of the next hidden layer nodes

        '''

        self.inputs = inputs
        self.outputs = outputs
        activation_func = activation_func.lower()
        if activation_func in ["relu", "sigmoid", "softmax"]:
            self.activation_func = activation_func
        else:
            raise ValueError("Activation functions allowed: RELU, Sigmoid, Softmax")
        #Validate activation function to make sure it is either RELU or sigmoid


        self.theta = np.array([np.random.standard_normal() for _ in range(len(self.inputs)+1)])
        self.theta = np.append(self.theta, 0.1)
        if self.activation_func!="softmax":
            self.inputs = np.insert(self.inputs, 0, 1)

        self.hidden_layer = hidden_layer
        self.hidden_node = hidden_node
       
        if hidden_layer == 1:
            self.output_expression = []
        

       #print(f"Input_expression: {input_expression}")
        self.input_expression = input_expression
       #print(f"self.input_expression : {self.input_expression}")

        self.output_expression = []



        

    @property
    def activation(self):
        return self._activation
    @activation.setter
    def activation(self, activation):
        self._activation = activation

    @property
    def output_expression(self):
        return self._output_expression
    @output_expression.setter
    def output_expression(self, output_expression):
        self._output_expression = output_expression
    
    @property
    def pred_outputs(self):
        return self._pred_outputs
    @pred_outputs.setter
    def pred_outputs(self, pred_outputs):
        self._pred_outputs = pred_outputs
    



    def calculate_activation(self):
       #print(f"Inputs: {self.inputs}")
       #print(f"theta: {self.theta}")
       #print(f"True Outputs: {self.outputs}")
        if self.activation_func !="softmax":
            self.d_product = (np.dot(self.inputs, self.theta[:-1]))

        if self.activation_func == "relu":
            self.activation = max(0, self.d_product)
        elif self.activation_func == "sigmoid":
            self.pred_outputs = 1/(1+np.exp(-self.d_product))
        elif self.activation_func == "softmax":
            self.pred_outputs = []
            sum_exponents = 0
            for i in range(len(self.outputs)):
                sum_exponents += np.exp(self.inputs[i])
            for i in range(len(self.outputs)):
                self.pred_outputs.append((np.exp(self.inputs[i]))/float(sum_exponents))



        
        if self.activation_func =="relu":
            self.activation += self.theta[-1]
        
        #if self.activation_func=="relu":
           #print(f"Activation: {self.activation}")

    
    def calculate_expression(self):

        if self.activation_func !="softmax":
            if self.hidden_layer == 1:
                symbol_x = [sympy.Symbol(f"x{i}") for i in range(len(self.inputs))]

            # When perceptron is from the first hidden layer, the symbol_x will be "x1", "x2"..."xn"  
            
            else:
                symbol_x = self.input_expression
                
                if symbol_x[0] != sympy.Symbol("x0"):
                    symbol_x.insert(0, sympy.Symbol("x0"))

                #Symbol_x should be equal to the input expression when the neuron is not from the first hidden layer
                #This allows for easier substitution during backpropogation


            self.symbol_theta = [sympy.Symbol(f"theta{self.hidden_layer}{i}{self.hidden_node}") for i in range(len(self.theta))]
            # Intitialising the symbol_theta as "theta{hidden layer}{input node}{output node}"

            self.output_expression = []
            #This is the temporary expression that will be added onto the final expression
           #print(f"Symbol_x: {symbol_x}")
           #print(f"Symbol_theta: {self.symbol_theta}")

            for i in range(len(self.inputs)):

                if self.output_expression == []:
                    self.output_expression.append(symbol_x[i]*self.symbol_theta[i])
                else:
                    self.output_expression[0] += symbol_x[i]*self.symbol_theta[i]

            
            if self.activation_func =="relu":
                if self.d_product<=0:
                    self.output_expression = []
                    # No effect on loss function if output of actication function is 0 (except end bias term)

            elif self.activation_func =="sigmoid":
                self.output_expression[0] = 1 * (1 + sympy.exp(-self.output_expression[0]))**-1 


            if self.activation_func == "relu":
                if self.output_expression:
                    self.output_expression[0] += self.symbol_theta[len(self.symbol_theta)-1]
                else:
                    self.output_expression.append(self.symbol_theta[len(self.symbol_theta)-1])

            #Adding the end bias term
        else:
            for i in range(len(self.input_expression)):
                if  i == 0:
                    sum_exponents_expression = sympy.exp(self.input_expression[i])
                else:
                    sum_exponents_expression +=sympy.exp(self.input_expression[i])
            
            self.output_expression = []
            for i in range(len(self.input_expression)):
                self.output_expression.append((sympy.exp(self.input_expression[i]))/sum_exponents_expression)
            

        
        if self.activation_func != "softmax":
            self.output_expression = self.output_expression[0]


       #print("Output Expression:",self.output_expression)
       #print()

    
    
    

def main():
    input = np.array([0, 1, 1, 2])
    outputs = np.array([1, 0])
    activation_func = "sigmoid"
    p1 = Perceptron(inputs = input, outputs=outputs, activation_func="relu", hidden_layer=1, hidden_node=0)
    p1.calculate_activation()
    p1.calculate_expression()
   #print("\n")
    p2 = Perceptron(inputs = input, outputs = outputs, activation_func="relu", hidden_layer=1, hidden_node=1)
    p2.calculate_activation()
    p2.calculate_expression()
   #print("\n")
    p3 = Perceptron(inputs = [p1.activation, p2.activation], outputs =  outputs, activation_func="softmax", hidden_layer=2, hidden_node=0, input_expression=[p1.output_expression, p2.output_expression])
    p3.calculate_activation()
    p3.calculate_expression()

if __name__=="__main__":
    main()