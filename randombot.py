from random import randint
from utils import *
from position import Position

class RandomBot:
	
	def get_move(self, pos, tleft):
		
		lmoves = pos.legal_moves()
		rm = 1#randint(0, len(lmoves)-1)
		return lmoves[rm]


class AlphabetaBot:

	def __init__(self,nn,d):
		self.nn = nn
		self.d = d
		
		pass

	# experimental - have make move and successors in the player class!
	def make_move(self,pos, move, pid):
		# this is one way to do it. Could be very slow with all the copying!
		# Might be quicker to separate positions and methods for searching.
		branch_pos = copy.deepcopy(pos)

		x=move[0]
		y=move[1]
		#mbx, mby = x/3, y/3
		#self.macroboard[3*mby+mbx] = -1
		branch_pos.board[9*y+x] = pid

		#Update situation for next player
		branch_pos.internal_pid = 3-branch_pos.internal_pid
		branch_pos.determine_win_macroboard(move,pid)
		# determine allowable next move:
		branch_pos.determine_macroboard(move,pid)
		return branch_pos

	def successors(self,pos):
		return [(move, self.make_move(pos,move,pos.internal_pid)) for move in pos.legal_moves()]


	def eval_fn(self,pos,nn):
		if pos.terminal_test():
			return pos.terminal_util(pos.internal_pid)
		else:
			return 0.5#predict_nn(self.nn,pos.board,pid)

	def alphabeta_search(self,pos):
		"""Search pos to determine best action; use alpha-beta pruning.
		This version cuts off search and uses an evaluation function."""
		def __init__(self):
			self.nn = 0

		def max_value(pos, alpha, beta, depth):
			if cutoff_test(pos, depth):
				return self.eval_fn(pos,self.nn)
			v = -infinity
			GS = self.successors(pos)
			for (a, s) in GS:
				if len(GS[0]) == 0:
					v = max(v, min_value(s, alpha, beta, depth))
				else:
					v = max(v, min_value(s, alpha, beta, depth+1))
				if v >= beta:
					return v
				alpha = max(alpha, v)
			return v

		def min_value(pos, alpha, beta, depth):
			if cutoff_test(pos, depth):
				return self.eval_fn(pos,self.nn)
			v = infinity
			GS = self.successors(pos)
			for (a, s) in GS:
				if len(GS[0]) == 0:
					v = min(v, max_value(s, alpha, beta, depth))
				else:
					v = min(v, max_value(s, alpha, beta, depth+1))
				if v <= alpha:
					return v
				beta = min(beta, v)
			return v

		# The default test cuts off at depth d or at a terminal state
		cutoff_test = (lambda pos,depth: depth>self.d or pos.terminal_test())
		
		action_states = self.successors(pos)

		for a,s in action_states:
			print a
			s.get_board()


		states  = [i[1] for i in action_states]
		actions = [i[0] for i in action_states]

		# if there's only 1 available action, just take it
		if len(actions) == 0:
			action=actions[0]
		else:
			Z = argmax(states,lambda s: min_value(s, -infinity, infinity, 0))
			action=actions[Z]
		return action


	def get_move(self,pos,tleft=100):
		branch_pos = copy.deepcopy(pos)
		return self.alphabeta_search(branch_pos)

