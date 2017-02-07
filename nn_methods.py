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