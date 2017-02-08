import numpy as np
import copy

def generate_player_nn(to_mutate=None,k=99,N1=40,N2=10,start_spread=0.2,sigma=0.05):
	if to_mutate is None:
		#print('generating new player with random weights and biases')
		W1 = np.empty((0, k+1)) # 33 because we want the bias term too!

		for line in range(1,N1+1):
			W1 = np.append(W1, [start_spread*2*(np.random.random(k+1)-0.5)], axis=0)

		W2 = np.empty((0, N1+1)) # 41 because we want the bias term too!
		for line in range(1,N2+1):
			W2 = np.append(W2, [start_spread*2*(np.random.random(N1+1)-0.5)], axis=0)

		W3 = np.array([start_spread*2*(np.random.random(N2+1)-0.5)])

		mod = {'W1': W1,'W2': W2,'W3': W3}
	else:
		#print('Mutating existing player, with sigma ',sigma)
		mod = copy.deepcopy(to_mutate)
		k =  mod['W1'].shape[1]-1
		N1 = mod['W1'].shape[0]
		N2 = mod['W2'].shape[0]

		noise = np.random.normal(1,sigma,(N1,k+1))
		mod['W1'] = mod['W1']+noise

		noise = np.random.normal(1,sigma,(N2,N1+1))
		mod['W2'] = mod['W2']+noise

		noise = np.random.normal(1,sigma,(1,N2+1))
		mod['W3'] = mod['W3']+noise
	return mod

def regeneration(prev_gen=None,spawn_ratio=3,N_players=15,sigma=0.05):
	if prev_gen is None:
		new_gen = {i: generate_player_nn() for i in range(N_players)}
	else:
		new_gen = {}
		for i in prev_gen:
			for j in range(spawn_ratio):
				new_gen[len(new_gen)] = generate_player_nn(to_mutate=prev_gen[i],sigma=sigma)
	return new_gen

def board_to_input(board):
	
	x = board['microboard'][:]+board['win_macroboard'][:]
	# fliperoo
	if board['next_turn'] == 2:
		for i in x:
			if x[i] == 1:
				x[i] = 2
			if x[i] == 2:
				x[i] == 1
	x = x + board['macroboard'][:]
	return x


def predict_nn(model, board):
	x = board_to_input(board)
	x = np.append(x,1)
	W1, W2, W3 = model['W1'], model['W2'], model['W3']
	# Forward propagation
	z1 = np.dot(W1,x)
	a1 = np.tanh(z1)
	z2 = np.dot(W2,np.append(a1,1))
	a2 = np.tanh(z2)
	z3 = np.dot(W3,np.append(a2,1))
	a3 = np.tanh(z3)
	return a3

class Network:
	""" using the excellend tutorial at http://neuralnetworksanddeeplearning.com/chap1.html"""
	def __init__(self,sizes,act_func,act_grad):
		self.sizes=sizes
		self.num_layers = len(sizes)
		self.biases = [np.random.randn(y, 1) for y in sizes]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.act_func = act_func
		self.act_grad = act_grad

	def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))

	def feedforward(self, a):
		"""Return the output of the network if "a" is input."""
		for b, w in zip(self.biases, self.weights):
			a = self.act_func(np.dot(w, a)+b)
		return a
		
	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The "mini_batch" is a list of tuples "(x, y)", and "eta"
		is the learning rate."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw 
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb 
					   for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = self.act_func(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			self.act_grad(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = self.act_grad(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)
	
	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations.
		y is the actual win/loss. """
		return (output_activations-y)

if __name__ == '__main__':

	def grad(x):
		return 1 - (np.tanh(x))**2

	nn = Network([3,2,1],np.tanh,grad)
	print nn.weights
	print nn.biases
	print nn.feedforward((1,1,1))




