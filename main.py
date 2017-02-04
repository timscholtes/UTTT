""" Run a game of UTTT

This starts from the default random bot from theaigames.com.

The parse_command function is for taking instruction from the
game server, and instructing the various module functions to operate
depending on the instruction.

For training, we will ignore this, as we don't want to have the overhead 
of passing instruction via string commands, rather we're going to use
a repeating while statement and player loop, with position class updates
to proceed with the game.

The function 'play_game' does exactly this. We will further develop logic 
for the learning process here.
"""

def parse_command(instr, bot, pos):
	if instr.startswith('action move'):
		time = int(instr.split(' ')[-1])
		x, y = bot.get_move(pos, time)
		pos.make_move()
		return 
		#return 'place_move %d %d\n' % (x, y)
	elif instr.startswith('update game field'):
		fstr = instr.split(' ')[-1]
		pos.parse_field(fstr)
	elif instr.startswith('update game macroboard'):
		mbstr = instr.split(' ')[-1]
		pos.parse_macroboard(mbstr)
	elif instr.startswith('update game move'):
		pos.nmove = int(instr.split(' ')[-1])
	elif instr.startswith('settings your_botid'):
		myid = int(instr.split(' ')[-1])
		bot.myid = myid
		bot.oppid = 1 if myid == 2 else 2
	elif instr.startswith('settings timebank'): 
		bot.timebank = int(instr.split(' ')[-1])
		print 'updating timebank'
	elif instr.startswith('settings time_per_move'): 
		bot.time_per_move = int(instr.split(' ')[-1])
	return ''


def play_game(pos,verbose=True,*players):
	""" Runs a game from scratch

	Runs a game from scratch given the position class, max counter and 
	a pair of players, which themselves are classes, with the method
	get_move(). They must be classes so that they can otherwise retrieve stored
	data, such as a neural net, etc.


	Args:
		pos: an instance of the Position() class
		verbose: Whether or not to print to console the state of the game each move.
		*players: the varaiable length (2 or more) player classes.
			For UTTT this should be just 2 players.

	Returns:
		The player id (pid) of the victorious player, or 0 for a draw.
	"""


	pid = 1
	tleft=1000
	while True:
		for player in players:
			if verbose:
				print '_'*50
				print 'New go for:',pid
				print 'board:'
				pos.get_board()
				print 'move macroboard'
				pos.get_macroboard()
				print 'current win macroboard'
				pos.get_win_macroboard()
				print 'legal moves:',pos.legal_moves()
				
			x,y = player.get_move(pos,tleft)
			if verbose:
				print 'player:',pid,'makes move:',x,y
			pos.make_move(x,y,pid)
			# update win macroboard
			pos.determine_win_macroboard(x,y,pid)
			# determine allowable next move:
			pos.determine_macroboard(x,y,pid)


			# check terminal state
			term = pos.terminal_state(pid)
			if term != -1:
				return term

			pid = 3 - pid





if __name__ == '__main__':
	import sys
	from position import Position
	from randombot import RandomBot


	pos = Position()
	bot1 = RandomBot()
	bot2 = RandomBot()
	
	outcome = play_game(pos,200,bot1,bot2)
	print 'winner is:',outcome


	






