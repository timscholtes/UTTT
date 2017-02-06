from itertools import chain
state = {'board': [0 for i in range(81)],
	'macroboard': [-1 for i in range(9)],
	'win_macroboard': [-1 for i in range(9)],
	'internal_pid': 1}



def state_to_input(state):

	x = state['board'][:]+state['win_macroboard'][:]
	# fliperoo
	if state['internal_pid'] == 2:
		for i in x:
			if x[i] == 1:
				x[i] = 2
			if x[i] == 2:
				x[i] == 1
	x = x + state['macroboard'][:]
	return x

print state_to_input(state)