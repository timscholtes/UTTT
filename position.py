import itertools

class Position:
	
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
		return [ (x, y) for x in range(9) for y in range(9) if self.is_legal(x, y) ]
		
	def make_move(self, x, y, pid):
		mbx, mby = x/3, y/3
		self.macroboard[3*mby+mbx] = -1
		self.board[9*y+x] = pid
		
	def get_board(self):
		line = []
		X = self.board
		for i in range(9):
			for x in X[i*9:(i+1)*9]:
				line.append(' {} '.format(x))
			print ''.join(line)
			line=[]
		pass

	def get_macroboard(self):
		for row in range(3):
			print self.macroboard[row*3:(row+1)*3]
		pass

	def get_win_macroboard(self):
		for row in range(3):
			print self.win_macroboard[row*3:(row+1)*3]
		pass
	
	def determine_win_macroboard(self,x,y,pid):
		mbx, mby = x/3, y/3
		mini_board = [self.board[mby*27+mbx*3+0:mby*27+mbx*3+3],
			self.board[mby*27+mbx*3+9:mby*27+mbx*3+12],
				self.board[mby*27+mbx*3+18:mby*27+mbx*3+21]]
		mini_board = list(itertools.chain(*mini_board))
		print 'mini'
		print mini_board
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

