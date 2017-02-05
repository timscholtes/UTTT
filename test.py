
def terminal_test(state):

	win_combos = [range(3),range(3,6),range(6,9),
		[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]


	for pid in (1,2):
		for combo in win_combos:
			print 'eval:',combo
			success = True
			for x in combo:
				if combo == [6,7,8]:
					print state['win_macroboard'][x],pid
				if state['win_macroboard'][x] != pid:
					success = False
					break
				print success

				if success:
					return success

state = {'board': [0 for i in range(81)],
	'macroboard': [-1 for i in range(9)],
	'win_macroboard': [-1 for i in range(9)],
	'internal_pid': 1}

for i in range(6,9):
	state['win_macroboard'][i] = 1

print state['win_macroboard']



print terminal_test(state)