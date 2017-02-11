import itertools
import copy
import numpy as np

class boardObject:

	# potentially, instead
	def __init__(self,
		board = [0 for i in range(81)],
		macroboard = [-1 for i in range(9)],
		win_macroboard = [-1 for i in range(9)],
		next_turn = 1):
		self.board = board
		self.macroboard = macroboard
		self.win_macroboard = win_macroboard
		self.next_turn = next_turn



class UTTT:
	""" Contains game rules and updates.

	UTTT is more complicated than TTT! We contain all methods for
	play in here. We differ from theaigames.com method by having
	a separate 'win_macroboard' from the macroboard. This allows us
	to separate information about macro win/loss/draws from the allowable
	moves. This is only serves to minimise how many evaluations we have to 
	make.

	For example, in determine_macroboard, we encode the logic that the player
	is not required to play in a region where there is already a win/loss/draw.

	In the aigames.com version, a 0 in the macroboard is used for both a draw 
	board, but also for macroboards where the player is not allowed to play.

	I also anticipate this win_macroboard may be a useful pre_compiled 
	abstraction for training.

	"""

	def __init__(self):
		self.win_combos = [range(3),range(3,6),range(6,9),
		[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
		self.win_combos2 = {0:((1,2),(3,6),(4,8)),
		1:((0,2),(4,7)),
		2:((0,1),(5,8),(4,6)),
		3:((0,6),(4,5)),
		4:((1,7),(3,5),(0,8),(2,6)),
		5:((3,4),(2,8)),
		6:((0,3),(7,8),(2,4)),
		7:((1,4),(6,8)),
		8:((6,7),(2,5))}
		# THis is for fast lookup
		self.mini_board_list = [self.get_mini_board(j) for j in range(9)]


	def parse_field(self, fstr):
		flist = fstr.replace(';', ',').split(',')
		self.board = [ int(f) for f in flist ]
	
	def parse_macroboard(self, mbstr):
		mblist = mbstr.replace(';', ',').split(',')
		self.macroboard = [ int(f) for f in mblist ]
	
	def is_legal(self,board, x, y):
		return board['microboard'][9*y+x] == 0

	def legal_moves(self,board):
		explore = []
		for j,k in enumerate(board['macroboard']):
			if k == -1:
				for i1 in range(3):
					for i2 in range(3):
						explore.append(((j%3)*3+i1,(j/3)*3+i2))
			else:
				continue

		return [(x, y) for x,y in explore if board['microboard'][9*y+x] == 0]			


	def deepish_copy(self,board):
		'''
		much, much faster than deepcopy, for a dict of the simple python types.
		'''
		out = dict().fromkeys(board)
		for k,v in board.iteritems():
			# try:
			#     out[k] = v.copy()   # dicts, sets
			# except AttributeError:
			try:
				out[k] = v[:]   # lists, tuples, strings, unicode
			except TypeError:
				out[k] = v      # ints	 
		return out
 

	def make_move(self,board, move):
		#board_copy = copy.deepcopy(board)

		board_copy = self.deepish_copy(board)
		pid = board_copy['next_turn']
		# here mbx/mby mean the maxi location
		# which of the 3by3s was played in?
		x=move[0]
		y=move[1]
		mbx = x / 3
		mby = y / 3
		j = mby * 3 + mbx
		board_copy['microboard'][9*y+x] = pid

		#Update situation for next player
		board_copy['next_turn'] = 3-pid
		board_copy['win_macroboard'][j] = self.determine_win_macroboard(board_copy,move,pid)
		# determine allowable next move:
		board_copy = self.determine_macroboard(board_copy,move)
		return board_copy
		


	def determine_macroboard(self,board,move):
		# here mbx,mby mean the mini location
		# where in the small sq did the move go?
		x = move[0]
		y = move[1]
		mbx = x % 3
		mby = y % 3
		j = mby * 3 + mbx
		if board['win_macroboard'][j] == -1:
			# faster to set all to 0, then put the one to -1 than list comp all.
			board['macroboard'] = [0]*9
			board['macroboard'][j] = -1
		else:
			# if the macro cell is won/lost/draw, then can go anywhere not
			# already won/lost/draw
			board['macroboard'] = [-1 if board['win_macroboard'][i] == -1 else 0 for i in range(9)]
		return board


	def get_mini_board(self,j):
		# j is the number 0:80 determining which cell was played.
		# mbx/y are for which of the 3b3s was played in.

		mbx, mby = j%3, j/3
		mini_board = [0]*9
		for i1 in range(3):
			for i2 in range(3):
				mini_board[i1*3+i2] = ((mby *3 + i1) * 9 + mbx * 3 + i2 )
		return mini_board


	def determine_win_macroboard(self,board,move,pid):
		#mini_board,mbx,mby = self.get_mini_board(move)
		x = move[0]
		y = move[1]
		mbx = x / 3
		mby = y / 3
		j = mby * 3 + mbx
		mini_board = self.mini_board_list[j]
		#x,y here are for getting z [0,9] which is where in the 3by3
		# the move was placed.
		x = move[0] % 3
		y = move[1] % 3
		z = y*3+x

		combos = self.win_combos2[z]
		for combo in combos:
			success = True
			for x in combo:
				if board['microboard'][mini_board[x]] != pid:
					success = False
					break
			#	else:
			#		continue
			if success:
				return pid

		success = True
		for x in xrange(9):
			if board['microboard'][mini_board[x]] == 0:
				success = False
				break
			#else:
			#	continue
		if success:
			return 0
		return -1


	def terminal_test(self,board,action):
		pid = 3-board['next_turn']

		x = action[0] / 3
		y = action[1] / 3
		z = y*3+x

		if board['win_macroboard'][z] == -1:
			return False
		else:
			if board['win_macroboard'][z] == pid:
				combos = self.win_combos2[z]
				for combo in combos:
					success = True
					for x in combo:
						if board['win_macroboard'][x] != pid:
							success = False
							break
					if success:
						return success

			success = True
			for x in xrange(9):
				if board['win_macroboard'][x] == -1:
					success = False
					break
				else:
					continue
			return success

			

	def terminal_pid(self,board):
		pid = 3-board['next_turn']
		# determine overall draw:
		#for pid in (1,2):
		for combos in self.win_combos:
			if all(board['win_macroboard'][x] == pid for x in combos):
				return pid
		if all(x != -1 for x in board['win_macroboard']):
			return 0
			

	def terminal_util(self,board):
		pid = 3-board['next_turn']
		outcome = self.terminal_pid(board)
		if outcome == pid:
			return 1
		elif outcome == 3 - pid:
			return -1
		else:
			return 0

	def successors(self,board):
		# worry about use of this will update pos in an irretrievable way
		# maybe need to make copy
		return [(move, self.make_move(board,move)) for move in self.legal_moves(board)]


	def play_game(self,record=False,verbose=False,*players,**kwargs):

		board = {'microboard': [0 for i in range(81)],
		'macroboard': [-1 for i in range(9)],
		'win_macroboard': [-1 for i in range(9)],
		'next_turn': 1}
		moves=[]

		# flip for who goes first:
		x = np.random.choice([1,2])
		for player in players:
			player.myid = x
			player.oppid = 3 - x
			x = 3 - x

		while True:
			for player in players:

				pid = player.myid

				if verbose:
					self.print_board_status(board)

				move = player.get_move(board)
				
				if record:
					moves.append((pid,move))

				if verbose:
					print 'player:',pid,'makes move:',move[0],move[1]
				
				board = self.make_move(board,move)

				if self.terminal_test(board,move):
					if verbose:
						print 'WINNER!'
					#outcome = [-1,-1]
					winner = self.terminal_pid(board)
					# if winner == 0:
					# 	outcome = [0,0]
					# else:
					# 	outcome[winner-1] = 1

					if record:
						return (winner,moves)
					else:
						return winner



	# ---- PRINTING FUNCTIONS START -----
	def print_board_status(self,board):
		print '_'*50
		print 'New go for:',board['next_turn']
		print 'board:'
		self.get_board(board)
		print 'move macroboard'
		self.get_macroboard(board)
		print 'current win macroboard'
		self.get_win_macroboard(board)
		print 'legal moves:',self.legal_moves(board)

	def get_board(self,board):
		X = board['microboard']
		for i in range(9):
			if i in (3,6):
				print '-'*30
			line=[]
			for y in range(3):
				for x in X[i*9 + y*3:i*9 + (y+1)*3]:
					line.append(' {} '.format(x))
				line.append('|')
			print ''.join(line)
		pass

	def get_macroboard(self,board):
		for row in range(3):
			print board['macroboard'][row*3:(row+1)*3]
		pass

	def get_win_macroboard(self,board):
		for row in range(3):
			print board['win_macroboard'][row*3:(row+1)*3]
		pass
	# ---- PRINTING FUNCTIONS END -----


if __name__ == '__main__':


	import bots
	
	game = UTTT()
	def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_prime(z):
		return sigmoid(z)*(1-sigmoid(z))

	bot1 = bots.PolicyBot([99,100,100,81],sigmoid,sigmoid_prime,game)
	bot2 = bots.PolicyBot([99,100,100,81],sigmoid,sigmoid_prime,game)

	bot1.myid = 1
	bot2.myid = 2
	bot1.oppid = 2
	bot2.oppid = 1

	outcome = game.play_game(True,bot1,bot2)



