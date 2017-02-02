
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


def play_game(pos,max_counter,*players):
	pid = 1
	tleft=1000
	counter = 0
	while counter < max_counter:
		counter += 1
		print 'counter:',counter
		for player in players:
			print pid
			print pos.legal_moves()
			pos.get_board()
			pos.get_macroboard()
			pos.get_win_macroboard()
			x,y = player.get_move(pos,tleft)
			print 'player:',pid,'makes move:',x,y
			pos.make_move(x,y,pid)

			# determine allowable next move:
			mbx = x / 3
			mby = y / 3
			j = mby * 3 + mbx
			if pos.win_macroboard[j] == -1:
				pos.macroboard = [-1 if i == j else 0 for i in range(9)]
			else:
				pos.macroboard = [-1 for i in range(9)]
			pos.determine_win_macroboard(x,y,pid)

			# check terminal state
			term = pos.terminal_state(pid)
			if term:
				return pid

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


	






