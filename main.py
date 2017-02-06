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


def play_game(pos,nnets,verbose=False,*players):
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
	tleft=1000
	state = {'board': [0 for i in range(81)],
	'macroboard': [-1 for i in range(9)],
	'win_macroboard': [-1 for i in range(9)],
	'internal_pid': 1}

	while True:
		for player in players:

			pid = player.myid
			# ----- PRINTING START -----
			if verbose:
				print '_'*50
				print 'New go for:',pid
				print 'board:'
				pos.get_board(state)
				print 'move macroboard'
				pos.get_macroboard(state)
				print 'current win macroboard'
				pos.get_win_macroboard(state)
				print 'legal moves:',pos.legal_moves(state)
			# ----- PRINTING END -----

			move = player.get_move(pos,state,nnets[pid])
			if verbose:
				print 'player:',pid,'makes move:',move[0],move[1]
			# make the move - 
			# this will now update the game state too!
			# This was needed for game successor search.
			state = pos.make_move(state,move)
			# check terminal state

			if pos.terminal_test(state,move):
				if verbose:
					print 'WINNER!'
					print 'board:'
					pos.get_board(state)
					print 'move macroboard'
					pos.get_macroboard(state)
					print 'current win macroboard'
					pos.get_win_macroboard(state)
					print 'legal moves:',pos.legal_moves(state)
				outcome = [-1,-1]
				winner = pos.terminal_state(state)
				if winner == 0:
					outcome = [0,0]
				else:
					outcome[winner-1] = 1
				return outcome


if __name__ == '__main__':

	from position import Position
	from bots import RandomBot
	from bots import AlphabetaBot
	from nn_methods import *
	import time

	state = {'board': [0 for i in range(81)],
		'macroboard': [-1 for i in range(9)],
		'win_macroboard': [-1 for i in range(9)],
		'internal_pid': 1}
	#state = StateObject()
	pos = Position()
	bot1 = AlphabetaBot(2)
	bot2 = AlphabetaBot(2)
	
	bot1.myid = 1
	bot2.myid = 2
	bot1.oppid = 2
	bot2.oppid = 1

	nnets = {i: generate_player_nn() for i in range(1,3)}

	t0 = time.time()

	outcome = play_game(pos,nnets,True,bot1,bot2)
	t1 = time.time()
	print 'winner is:',outcome
	print t1-t0

