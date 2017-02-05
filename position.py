import itertools
import copy

class StateObject:

	# potentially, instead
	def __init__(self,
		board = [0 for i in range(81)],
		macroboard = [-1 for i in range(9)],
		win_macroboard = [-1 for i in range(9)],
		internal_pid = 1):
		self.board = board
		self.macroboard = macroboard
		self.win_macroboard = win_macroboard
		self.internal_pid = internal_pid



class Position:
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
	state, but also for macroboards where the player is not allowed to play.

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
	
	def is_legal(self,state, x, y):
		return state['board'][9*y+x] == 0

	def legal_moves(self,state):
		explore = []
		for j,k in enumerate(state['macroboard']):
			if k == -1:
				for i1 in range(3):
					for i2 in range(3):
						explore.append(((j%3)*3+i1,(j/3)*3+i2))
			else:
				continue

		return [(x, y) for x,y in explore if state['board'][9*y+x] == 0]			


	def deepish_copy(self,state):
		'''
		much, much faster than deepcopy, for a dict of the simple python types.
		'''
		out = dict().fromkeys(state)
		for k,v in state.iteritems():
			# try:
			#     out[k] = v.copy()   # dicts, sets
			# except AttributeError:
			try:
				out[k] = v[:]   # lists, tuples, strings, unicode
			except TypeError:
				out[k] = v      # ints	 
		return out
 

	def make_move(self,state, move):
		#state_copy = copy.deepcopy(state)
		state_copy = self.deepish_copy(state)
		pid = state_copy['internal_pid']
		# here mbx/mby mean the maxi location
		# which of the 3by3s was played in?
		x=move[0]
		y=move[1]
		mbx = x / 3
		mby = y / 3
		j = mby * 3 + mbx
		state_copy['board'][9*y+x] = pid

		#Update situation for next player
		state_copy['internal_pid'] = 3-pid
		state_copy['win_macroboard'][j] = self.determine_win_macroboard(state_copy,move,pid)
		# determine allowable next move:
		state_copy = self.determine_macroboard(state_copy,move)
		return state_copy
		


	def determine_macroboard(self,state,move):
		# here mbx,mby mean the mini location
		# where in the small sq did the move go?
		x = move[0]
		y = move[1]
		mbx = x % 3
		mby = y % 3
		j = mby * 3 + mbx
		if state['win_macroboard'][j] == -1:
			# faster to set all to 0, then put the one to -1 than list comp all.
			state['macroboard'] = [0]*9
			state['macroboard'][j] = -1
		else:
			# if the macro cell is won/lost/draw, then can go anywhere not
			# already won/lost/draw
			state['macroboard'] = [-1 if state['win_macroboard'][i] == -1 else 0 for i in range(9)]
		return state


	def get_mini_board(self,j):
		# j is the number 0:80 determining which cell was played.
		# mbx/y are for which of the 3b3s was played in.

		mbx, mby = j%3, j/3
		mini_board = [0]*9
		for i1 in range(3):
			for i2 in range(3):
				mini_board[i1*3+i2] = ((mby *3 + i1) * 9 + mbx * 3 + i2 )
		return mini_board


	def determine_win_macroboard(self,state,move,pid):
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
				if state['board'][mini_board[x]] != pid:
					success = False
					break
			#	else:
			#		continue
			if success:
				return pid

		success = True
		for x in xrange(9):
			if state['board'][mini_board[x]] == 0:
				success = False
				break
			#else:
			#	continue
		if success:
			return 0
		return -1


	def terminal_test(self,state,action):
		pid = 3-state['internal_pid']

		x = action[0] / 3
		y = action[1] / 3
		z = y*3+x

		if state['win_macroboard'][z] == -1:
			return False
		else:
			if state['win_macroboard'][z] == pid:
				combos = self.win_combos2[z]
				for combo in combos:
					success = True
					for x in combo:
						if state['win_macroboard'][x] != pid:
							success = False
							break
					if success:
						return success

			success = True
			for x in xrange(9):
				if state['win_macroboard'][x] == -1:
					success = False
					break
				else:
					continue
			return success

			

	def terminal_state(self,state):
		pid = 3-state['internal_pid']
		# determine overall draw:
		#for pid in (1,2):
		for combos in self.win_combos:
			if all(state['win_macroboard'][x] == pid for x in combos):
				return pid
		if all(x != -1 for x in state['win_macroboard']):
			return 0
			

	def terminal_util(self,state):
		pid = 3-state['internal_pid']
		outcome = self.terminal_state(state)
		if outcome == pid:
			return 1
		elif outcome == 3 - pid:
			return -1
		else:
			return 0

	def successors(self,state):
		# worry about use of this will update pos in an irretrievable way
		# maybe need to make copy
		return [(move, self.make_move(state,move)) for move in self.legal_moves(state)]


	# ---- PRINTING FUNCTIONS START -----
	def get_board(self,state):
		X = state['board']
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

	def get_macroboard(self,state):
		for row in range(3):
			print state['macroboard'][row*3:(row+1)*3]
		pass

	def get_win_macroboard(self,state):
		for row in range(3):
			print state['win_macroboard'][row*3:(row+1)*3]
		pass
	# ---- PRINTING FUNCTIONS END -----

