""" Run a game of UTTT

This starts from the default random bot from theaigames.com.

The parse_command function is for taking instruction from the
game server, and instructing the various module functions to operate
depending on the instruction.

For training, we will ignore this, as we don't want to have the overhead 
of passing instruction via string commands, rather we're going to use
a repeating while boardment and player loop, with gameition class updates
to proceed with the game.

The function 'play_game' does exactly this. We will further develop logic 
for the learning process here.
"""

def parse_command(instr, bot, game):
	if instr.startswith('action move'):
		time = int(instr.split(' ')[-1])
		x, y = bot.get_move(game, time)
		game.make_move()
		return 
		#return 'place_move %d %d\n' % (x, y)
	elif instr.startswith('update game field'):
		fstr = instr.split(' ')[-1]
		game.parse_field(fstr)
	elif instr.startswith('update game macroboard'):
		mbstr = instr.split(' ')[-1]
		game.parse_macroboard(mbstr)
	elif instr.startswith('update game move'):
		game.nmove = int(instr.split(' ')[-1])
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


def play_game(game,nnets,verbose=False,*players):
	""" Runs a game from scratch

	Runs a game from scratch given the gameition class, max counter and 
	a pair of players, which themselves are classes, with the method
	get_move(). They must be classes so that they can otherwise retrieve stored
	data, such as a neural net, etc.

	Args:
		game: an instance of the gameition() class
		verbose: Whether or not to print to console the board of the game each move.
		*players: the varaiable length (2 or more) player classes.
			For UTTT this should be just 2 players.

	Returns:
		The player id (pid) of the victorious player, or 0 for a draw.
	"""
	tleft=1000
	board = {'microboard': [0 for i in range(81)],
	'macroboard': [-1 for i in range(9)],
	'win_macroboard': [-1 for i in range(9)],
	'next_turn': 1}

	while True:
		for player in players:

			pid = player.myid

			if verbose:
				game.print_board_status(board)

			move = player.get_move(game,board,nnets[pid])
			if verbose:
				print 'player:',pid,'makes move:',move[0],move[1]
			
			board = game.make_move(board,move)

			if game.terminal_test(board,move):
				if verbose:
					print 'WINNER!'
				outcome = [-1,-1]
				winner = game.terminal_pid(board)
				if winner == 0:
					outcome = [0,0]
				else:
					outcome[winner-1] = 1
				return outcome


if __name__ == '__main__':

	from game_rules import UTTT
	import bots
	from nn_methods import *
	import time

	board = {'microboard': [0 for i in range(81)],
		'macroboard': [-1 for i in range(9)],
		'win_macroboard': [-1 for i in range(9)],
		'next_turn': 1}
	#board = boardObject()
	game = UTTT()
	# bot1 = bots.AlphabetaBot(3)
	# bot2 = bots.AlphabetaBot(3)
	
	def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_prime(z):
		return sigmoid(z)*(1-sigmoid(z))

	bot1 = bots.PolicyBot([99,4,2,1],sigmoid,sigmoid_prime,game)
	bot2 = bots.PolicyBot([99,4,2,1],sigmoid,sigmoid_prime,game)

	bot1.myid = 1
	bot2.myid = 2
	bot1.oppid = 2
	bot2.oppid = 1

	nnets = {i: generate_player_nn() for i in range(1,3)}

	t0 = time.time()

	outcome = play_game(game,nnets,False,bot1,bot2)
	t1 = time.time()
	print 'winner is:',outcome
	print t1-t0

