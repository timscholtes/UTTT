from random import randint
from utils import *
from position import Position
from nn_methods import *

class RandomBot:
	
	def get_move(self, pos,state,nn, tleft=100):
		
		lmoves = pos.legal_moves(state)
		rm = randint(0, len(lmoves)-1)
		return lmoves[rm]


class AlphabetaBot:

	def __init__(self,d):
		#self.nn = nn
		self.d = d
		pass


	def alphabeta_search(self,pos,state,nn):
		"""Search state to determine best action; use alpha-beta pruning.
		This version cuts off search and uses an evaluation function."""

		def depth_eval_fn(state,nn):
			return predict_nn(nn,state)

		def max_value( state,alpha, beta, depth,a):
			# manual cutoff tests should be faster than reevaluation
			# it's mostly going to be depth!
			if depth>self.d:
				return depth_eval_fn(state,nn)
			if pos.terminal_test(state,a):
				return pos.terminal_util(state)

			v = -infinity
			GS = pos.successors(state)
			for (a, s) in GS:
				v = max(v, min_value(s, alpha, beta, depth+1,a))
				if v >= beta:
					return v
				alpha = max(alpha, v)
			return v

		def min_value(state, alpha, beta, depth,a):
			# manual cutoff tests should be faster than reevaluation
			if depth>self.d:
				return depth_eval_fn(state,nn)
			if pos.terminal_test(state,a):
				return pos.terminal_util(state)
				
			v = infinity
			GS = pos.successors(state)
			for (a, s) in GS:
				v = min(v, max_value(s, alpha, beta, depth+1,a))
				if v <= alpha:
					return v
				beta = min(beta, v)
			return v

		# The default test cuts off at depth d or at a terminal state
		
		action_states = pos.successors(state)
		actions = [i[0] for i in action_states]

		# if there's only 1 available action, just take it
		if len(actions) == 1:
			action=actions[0]
		else:
			Z = argmax(action_states,lambda (a,s): min_value(s, -infinity, infinity, 0,a))
			action = actions[Z]
		return action


	def get_move(self,pos,state,nn,tleft=100):
		return self.alphabeta_search(pos,state,nn)



class MCTSBot:

	def __init__(self, board, **kwargs):
		self.state = state
		self.states = []

	def update(self,state):
		self.states.append(state)
		pass

	def get_move(self):
		pass

	def run_simulation(self):
		pass












