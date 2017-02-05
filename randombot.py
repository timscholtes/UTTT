from random import randint
from utils import *
from position import Position

class RandomBot:
	
	def get_move(self, pos,state, tleft):
		
		lmoves = pos.legal_moves(state)
		rm = randint(0, len(lmoves)-1)
		return lmoves[rm]


class AlphabetaBot:

	def __init__(self,nn,d):
		self.nn = nn
		self.d = d
		pass

	# def eval_fn(self,pos,state,nn,a):
	# 	if pos.terminal_test(state,a):
	# 		return pos.terminal_util(state)
	# 	else:
	# 		return 0.5#predict_nn(self.nn,pos.board,pid)

	def alphabeta_search(self,pos,state):
		"""Search state to determine best action; use alpha-beta pruning.
		This version cuts off search and uses an evaluation function."""
		def __init__(self):
			self.nn = 0

		def depth_eval_fn(pos,state,nn):
			return 0.5#predict_nn(self.nn,pos.board,pid)

		def max_value( state,alpha, beta, depth,a):
			#if cutoff_test(pos, depth,a):
			#	return self.eval_fn(pos,state,self.nn,a)
			# manual cutoff tests should be faster than reevaluation
			# it's mostly going to be depth!
			if depth>self.d:
				return depth_eval_fn(pos,state,self.nn)
			if pos.terminal_test(state,a):
				return pos.terminal_util(state)

			v = -infinity
			GS = pos.successors(state)
			for (a, s) in GS:
				# if len(GS[0]) == 0:
				# 	v = max(v, min_value(s, alpha, beta, depth))
				# else:
				v = max(v, min_value(s, alpha, beta, depth+1,a))
				if v >= beta:
					return v
				alpha = max(alpha, v)
			return v

		def min_value(state, alpha, beta, depth,a):
			# if cutoff_test(pos, depth,a):
			# 	return self.eval_fn(pos,state,self.nn,a)
			# manual cutoff tests should be faster than reevaluation
			if depth>self.d:
				return depth_eval_fn(pos,state,self.nn)
			if pos.terminal_test(state,a):
				return pos.terminal_util(state)
				
			v = infinity
			GS = pos.successors(state)
			for (a, s) in GS:
			#	if len(GS[0]) == 0:
			#		v = min(v, max_value(s, alpha, beta, depth))
			#	else:
				v = min(v, max_value(s, alpha, beta, depth+1,a))
				if v <= alpha:
					return v
				beta = min(beta, v)
			return v

		# The default test cuts off at depth d or at a terminal state
		#cutoff_test = (lambda pos,depth,action: depth>self.d or pos.terminal_test(state,action))

		action_states = pos.successors(state)
		states  = [i[1] for i in action_states]
		actions = [i[0] for i in action_states]

		# if there's only 1 available action, just take it
		if len(actions) == 1:
			action=actions[0]
		else:
			Z = argmax(action_states,lambda (a,s): min_value(s, -infinity, infinity, 0,a))
			action = actions[Z]

		return action


	def get_move(self,pos,state,tleft=100):
		return self.alphabeta_search(pos,state)

