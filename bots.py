import random
from utils import *
from nn_methods import *
import datetime
from math import log, sqrt
import numpy as np

class RandomBot:
	
	def get_move(self, pos,board,nn, tleft=100):
		
		lmoves = pos.legal_moves(board)
		rm = random.randint(0, len(lmoves)-1)
		return lmoves[rm]


class AlphabetaBot:

	def __init__(self,d):
		#self.nn = nn
		self.d = d
		pass


	def alphabeta_search(self,pos,board,nn):
		"""Search board to determine best action; use alpha-beta pruning.
		This version cuts off search and uses an evaluation function."""

		def depth_eval_fn(board,nn):
			return predict_nn(nn,board)

		def max_value( board,alpha, beta, depth,a):
			# manual cutoff tests should be faster than reevaluation
			# it's mostly going to be depth!
			if depth>self.d:
				return depth_eval_fn(board,nn)
			if pos.terminal_test(board,a):
				return pos.terminal_util(board)

			v = -infinity
			GS = pos.successors(board)
			for (a, s) in GS:
				v = max(v, min_value(s, alpha, beta, depth+1,a))
				if v >= beta:
					return v
				alpha = max(alpha, v)
			return v

		def min_value(board, alpha, beta, depth,a):
			# manual cutoff tests should be faster than reevaluation
			if depth>self.d:
				return depth_eval_fn(board,nn)
			if pos.terminal_test(board,a):
				return pos.terminal_util(board)
				
			v = infinity
			GS = pos.successors(board)
			for (a, s) in GS:
				v = min(v, max_value(s, alpha, beta, depth+1,a))
				if v <= alpha:
					return v
				beta = min(beta, v)
			return v

		# The default test cuts off at depth d or at a terminal board
		
		action_boards = pos.successors(board)
		actions = [i[0] for i in action_boards]

		# if there's only 1 available action, just take it
		if len(actions) == 1:
			action=actions[0]
		else:
			Z = argmax(action_boards,lambda (a,s): min_value(s, -infinity, infinity, 0,a))
			action = actions[Z]
		return action


	def get_move(self,pos,board,nn,tleft=100):
		return self.alphabeta_search(pos,board,nn)



class MCTSBot:
	"""
	Credit here goes to Jeff Bradberry's excellent explanation of 
	MCTS, found here:
	https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
	"""

	def __init__(self, game, **kwargs):
		self.game = game
		self.boards = []

		self.wins = {}
		self.moves = {}

		self.C = kwargs.get('C',1.41)

		seconds = kwargs.get('time',30)
		self.calculation_time = datetime.timedelta(seconds=seconds)
		self.max_moves = kwargs.get('max_moves',100)

	def update(self,board):
		self.boards.append(board)
		pass

	def get_move(self,board):
		self.update(board)
		self.max_depth = 0
		board = self.boards[-1]
		player = board['next_turn']
		legal = self.game.legal_moves(board)

		if not legal:
			return
		if len(legal) == 1:
			return legal[0]

		games = 0

		begin = datetime.datetime.utcnow()
		while datetime.datetime.utcnow()-begin < self.calculation_time:
			self.run_sim()
			games += 1
		
		action_boards = self.game.successors(board)

		percent_wins,move = max(
			(float(self.wins.get((player,self.listify(S)),0))/
				self.moves.get((player,self.listify(S)),1),m)
			for m,S in action_boards)

		print games, datetime.datetime.utcnow() - begin
		# Display the stats for each possible play.
		for x in sorted(
			((100 * self.wins.get((player, self.listify(S)), 0) /
			  self.moves.get((player, self.listify(S)), 1),
			  self.wins.get((player, self.listify(S)), 0),
			  self.moves.get((player, self.listify(S)), 0), p)
			 for p, S in action_boards),
			reverse=True
		):
			print "{3}: {0:.2f}% ({1} / {2})".format(*x)

		print "Maximum depth searched:", self.max_depth

		return move

	def listify(self,board):
		return tuple(board['microboard']+
		board['macroboard']+
		board['win_macroboard'])


	def run_sim(self):
		moves, wins = self.moves, self.wins
		visited_boards = set()
		boards_copy = self.boards[:]
		board = boards_copy[-1]
		player = board['next_turn']

		expand = True
		for t in xrange(self.max_moves):
			if t > self.max_depth:
				self.max_depth = t
			legal = self.game.legal_moves(board)
			#move = choice(legal)
			action_boards = self.game.successors(board)
			
			if all(moves.get((player,self.listify(S))) for m,S in action_boards):
				log_total = log(
					sum(moves[(player,self.listify(S))] for p,S in action_boards))
				value,move,board = max(
					((wins[(player, self.listify(S))] / moves[(player, self.listify(S))]) + 
						self.C * sqrt(log_total / moves[(player,self.listify(S))]), p, S)
					for p,S in action_boards
					)
			else:
				move,board = action_boards[random.randint(0, len(legal)-1)]


			#board = self.game.make_move(board,move)
			boards_copy.append(board)

			if expand and (player,self.listify(board)) not in self.moves:
				expand = False
				moves[(player,self.listify(board))] = 0
				wins[(player,self.listify(board))] = 0
				# if t > self.max_depth:
				# 	self.max_depth = t

			visited_boards.add((player,self.listify(board)))

			player = board['next_turn']
			win_check = self.game.terminal_test(board,move)
			if not win_check:
				winner = 0
			else:
				winner = self.game.terminal_pid(board)
				break

		for player, board in visited_boards:
			if (player,board) not in self.moves:
				continue
			moves[(player,board)] += 1
			if player == winner:
				wins[(player,board)] += 1


class  PolicyBot:

	def __init__(self,sizes,act_func,act_grad,game,weights=None,biases=None):
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.game = game
		
		if biases is None:
			self.biases = [np.random.randn(y) for y in sizes[1:]]
			self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

		self.act_func = act_func
		self.act_grad = act_grad

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = self.act_func(np.dot(w, a))
		return a
	
	def listify(self, board):
		return np.asarray(board['microboard']+
		board['macroboard']+
		board['win_macroboard'])
	
	def softmax(self, x):
		tot = sum(x)
		return [y / tot for y in x]

	def get_move(self, game,board, *args):

		action_boards = self.game.successors(board)

		outputs = self.softmax([float(self.feedforward(self.listify(b))) 
			for a,b in action_boards])

		choice_int = np.random.choice(range(len(outputs)),1,p=outputs)
		return action_boards[choice_int][0]
		


if __name__ == '__main__':

	from game_rules import UTTT
	board = {'microboard': [0 for i in range(81)],
		'macroboard': [-1 for i in range(9)],
		'win_macroboard': [-1 for i in range(9)],
		'next_turn': 1}

	#board = boardObject()
	game = UTTT()
	
	def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_prime(z):
		return sigmoid(z)*(1-sigmoid(z))

	#bot = MCTSBot(game,time=2,max_moves=50)
	bot = PolicyBot([99,4,2,1],sigmoid,sigmoid_prime,game)
	#print bot.listify(board)

	for i in range(200):
		bot.get_move(board)
	# for w,b in zip(bot.weights,bot.biases):
	# 	print w.shape,b.shape











