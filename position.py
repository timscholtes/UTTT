import itertools

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
		self.board = [0 for i in range(81)]
		self.macroboard = [-1 for i in range(9)]
		self.win_macroboard = [-1 for i in range(9)]
		self.win_combos = [range(3),range(2,5),range(5,8),
		[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
	
	def parse_field(self, fstr):
		flist = fstr.replace(';', ',').split(',')
		self.board = [ int(f) for f in flist ]
	
	def parse_macroboard(self, mbstr):
		mblist = mbstr.replace(';', ',').split(',')
		self.macroboard = [ int(f) for f in mblist ]
	
	def is_legal(self, x, y):
		mbx, mby = x/3, y/3
		return self.macroboard[3*mby+mbx] == -1 and self.board[9*y+x] == 0

	def legal_moves(self):
		return [(x, y) for x in range(9) for y in range(9) if self.is_legal(x, y)]
		
	def make_move(self, x, y, pid):
		#mbx, mby = x/3, y/3
		#self.macroboard[3*mby+mbx] = -1
		self.board[9*y+x] = pid
		pass
		
	def get_board(self):
		X = self.board
		for i in range(9):
			if i in (3,6):
				print '-'*30
			line=[]
			for y in range(3):
				for x in X[i*9 + y*3:i*9 + (y+1)*3]:
					line.append(' {} '.format(x))
				line.append('|')
			print ''.join(line)
			# for x in X[i*9:(i+1)*9]:
			# 	line.append(' {} '.format(x))
			# print ''.join(line)
		pass

	def get_macroboard(self):
		for row in range(3):
			print self.macroboard[row*3:(row+1)*3]
		pass

	def get_win_macroboard(self):
		for row in range(3):
			print self.win_macroboard[row*3:(row+1)*3]
		pass
	
	def determine_macroboard(self,x,y,pid):
		# here mbx,mby mean the mini location -
		# where in the big sq did the move go?
		mbx = x % 3
		mby = y % 3
		j = mby * 3 + mbx
		if self.win_macroboard[j] == -1:
			self.macroboard = [-1 if i == j else 0 for i in range(9)]
		else:
			# if the macro cell is won/lost/draw, then can go anywhere not
			# already won/lost/draw
			self.macroboard = [-1 if self.win_macroboard[i] == -1 else 0 for i in range(9)]


	def determine_win_macroboard(self,x,y,pid):
		mbx, mby = x/3, y/3
		mini_board = [self.board[mby*27+mbx*3+0:mby*27+mbx*3+3],
			self.board[mby*27+mbx*3+9:mby*27+mbx*3+12],
				self.board[mby*27+mbx*3+18:mby*27+mbx*3+21]]
		mini_board = list(itertools.chain(*mini_board))
		for combo in self.win_combos:
			if all(mini_board[x] == pid for x in combo):
				self.win_macroboard[3*mby+mbx] = pid
				pass
		if all(mini_board[x] != 0 for x in range(9)):
			self.win_macroboard[3*mby+mbx] = 0
			pass

	def terminal_state(self,pid):
		for combos in self.win_combos:
			if all(self.win_macroboard[x] == pid for x in combos):
				return True

