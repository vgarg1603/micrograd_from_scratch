
# Derivative function:

$$ \frac{dy}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$


[![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Tangent_to_a_curve.svg/330px-Tangent_to_a_curve.svg.png)](https://en.wikipedia.org/wiki/File:Tangent_to_a_curve.svg)
## Chain Rule:

For a single neuron:

- d (activation fxn OR o) / d (activation fxn OR o) = 1
- d (o) / d (x1w1 + x2w2 + b OR n) = derivative of activation function used
- d (o) / d (b) = d (o) / d (n) * d(n) / d (b)
- d (o) / d (x1w1 + x2w2 OR x1w1x2w2) = d (o) / d(n) * d(n) / d (x1w1x2w2)
- d (o) / d (x1w1) = d (o) / d (x1w1x2w2) * d(x1w1x2w2) / d(x1w1)
- And so on

Note:  **To create backpropogation we can now do it recursively in order to check the influence of weights and bias on final output
and also update weights in the direction of final output i.e gradient
or in reverse direction without having to recalculate intermediate results**

Essentially our formula should be :

# $\frac{d\,\text{output}}{d\,\text{current state}} = \frac{d\,\text{output}}{d\,\text{next state}} \cdot \frac{d\,\text{next state}}{d\,\text{current state}}$


To automatically perform back-propogation we need to use [[Topological Sort]] algorithm to call the _backward function described in the notebook in topological order.

## Implementing backpropogation using pytorch:

initialise input vectors: torch.Tensor([-2.0]). **Note: use .double() in the end to convert default 32bit floating point to 64bit for consistency with the micrograd**

torch.tanh(output) for implementing activation function on w1x1 + w2x2 + ... + b


## Neuron class:
![[Pasted image 20260519005841.png]]
### Initialising a neuron:

	1. to initialise a neuron we need number of inputs into the neuron (number of axons): 
	
	Neuron(nin)

	2. we initialise weights for each input neuron (axon) hence: 
	
	number of weights = nin
	
	2. initialise bias b

### Calling a neuron Neuron(x):
1. perform $\sum_{i=1}^{i=n}w_i * x_i + b$
2. Perform activation function on this result and return it

## Layer class:
[![CS231n Deep Learning for Computer Vision](https://cs231n.github.io/assets/nn1/neural_net.jpeg)![CS231n Deep Learning for Computer Vision](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSEi1tlkc_X3o6lXjVV0_FxfIOTdlQpnh_gtg&s)](https://www.google.com/url?sa=t&source=web&rct=j&url=https%3A%2F%2Fcs231n.github.io%2Fneural-networks-1%2F&ved=0CBYQjRxqFwoTCMi-qIvJw5QDFQAAAAAdAAAAABAb&opi=89978449)
### Initialising a Layer:
	1. We need number of input neurons into the layer: 
	
	nin
	
	2. We need number of neurons needed in the layer: 
	
	nout

	3. Initialise same number of neurons as required in the needed neurons:
	
	Number Of Neurons Required = nout
	
	3. Each initialised neuron should have number of input neurons passed to it:

	Neuron(nin)

### Calling A Layer Layer(x):
	1. Each neuron present in the layer should have its:

	output of each Neuron_i = Neuron(x)

## MLP (Multi-Layer Perceptron):
### Initialise Layers in MLP:
	1. Input neurons in first layer
	2. Number of neurons in each layer as a list

### Call MLP Neural Net (NN):
	1. Each layer returns layer(x) on x input data
	2. overwrite x = layer(x) and return x as final output given by NN

# Train A Neural Net:

## 1. Forward Pass:
	Calculate loss (in this case: (ypred - yactual)**2))

## 2. Backpropogation (Backward Pass):
	1. First initialise all parameter's gradient to zero in each iteration to avoid add previous gradient values
		zerograd

	2. Call loss.backward() to calculate gradients of each parameter

## 3. Update parameter values (in this case:) $p.data+=-0.01 * p.grad$

