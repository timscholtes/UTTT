## build a tournament
from nn_methods import *
import random
from main import *
import json
import multiprocessing as mp
import os
import time
import copy_reg
import types
# from position import Position
# from randombot import AlphabetaBot

def unwrap_self_f(arg, **kwarg):
    return C.piped_play_replay(*arg, **kwarg)
 

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

def generate_schedule(N_players,N_opponents):
	X = {i: random.sample([x for x in range(N_players) if x!=i],N_opponents) for i in range(N_players)}
	games_list = []
	for p1,opp_list in X.items():
		for p2 in opp_list:
			flip = random.random()
		# randomise start order
			if flip<0.5:
				Y = [p1,p2]
			else:
				Y = [p2,p1]
			games_list.append(Y)
	return games_list

def generate_scoreboard(N_players):
	return [0 for i in range(N_players)]



def play_tournament(schedule,players,d,verbose):
	score = generate_scoreboard(len(players))
	pos = Position()
	player1=AlphabetaBot(d)
	player2=AlphabetaBot(d)

	player1.myid = 1
	player2.myid = 2
	player1.oppid = 2
	player2.oppid = 1

	game_counter=0
	total_games = len(schedule)
	scores = []
	for match in schedule:
		p1 = match[0]
		p2 = match[1]
		game_counter+= 1
		if verbose:
			print('Game: ',game_counter,' out of ',total_games,': ',p1,' vs ',p2)
		nnets = {1: players[p1],2: players[p2]}
		scores.append(play_game(pos,nnets,verbose,player1,player2))
		#score[p1] += outcome[0]
		#score[p2] += outcome[1]
	return scores

def setup_play_game(input_list):
	game=input_list[0]
	match=input_list[1]
	players=input_list[2]
	d=input_list[3]
	p1 = match[0]
	p2 = match[1]
	player1=AlphabetaBot(d)
	player2=AlphabetaBot(d)

	player1.myid = 1
	player2.myid = 2
	player1.oppid = 2
	player2.oppid = 1

	nnets = {1: players[p1],2: players[p2]}
	outcome = play_game(game,nnets,False,player1,player2)
	return outcome

def reconcile_scores(schedule,scores,N_players):
	rec_scores = [0 for i in range(N_players)]
	for i in range(len(scores)):
		rec_scores[schedule[i][0]] += scores[i][0]
		rec_scores[schedule[i][1]] += scores[i][1]
	return rec_scores


def play_parallel_tourn(schedule,players,num_cores,d):
	pos = Position()

	inputs = [(pos,m,players,d) for m in schedule]
	pool = mp.Pool(processes=num_cores)
	scores = pool.map(setup_play_game,inputs)
	pool.close()
	pool.join()
	rec_scores = reconcile_scores(schedule,scores,len(players))
	return(rec_scores)



def cull(scores,k):
	return np.argsort(scores)[::-1][range(k)]

def log_progress():
	pass

def evolve(N_gen,N_players,matches_per_player,carry_forward,sigma,d,verbose=False):
	gen_counter = 1
	while gen_counter <= N_gen:
		print("Running generation ",gen_counter)
		if gen_counter == 1:
			prev_gen = None
		else:
			prev_gen = {i: gen[i] for i in prev_gen} 
		gen = regeneration(prev_gen=prev_gen,N_players=N_players,sigma=sigma)
		schedule = generate_schedule(N_players,matches_per_player)
		scores = play_tournament(schedule=schedule,players=gen,verbose=verbose,d=d)
		prev_gen = list(cull(scores,carry_forward))
		gen_counter += 1

	top_player = gen[list(cull(scores,1))[0]]

	#with open('data/top_player.txt', 'w') as f:
	#	json.dump(top_player,f)

	np.save('data/top_player/W1.npy',top_player['W1'])
	np.save('data/top_player/W2.npy',top_player['W2'])
	np.save('data/top_player/W3.npy',top_player['W3'])
	np.save('data/top_player/W4.npy',top_player['W4'])

	return top_player

def parallel_evolve(N_gen,N_players,matches_per_player,carry_forward,sigma,d,num_cores,verbose):
	gen_counter = 1
	spawn_ratio = int(N_players / carry_forward)
	while gen_counter <= N_gen:
		print("Running generation ",gen_counter)
		print(time.time())
		if gen_counter == 1:
			prev_gen = None
		else:
			prev_gen = {i: gen[i] for i in prev_gen} 
		gen = regeneration(prev_gen=prev_gen,N_players=N_players,sigma=sigma,spawn_ratio = spawn_ratio)
		schedule = generate_schedule(N_players,matches_per_player)
		scores = play_parallel_tourn(schedule=schedule,players=gen,num_cores=num_cores,d=d)
		prev_gen = list(cull(scores,carry_forward))

		if gen_counter % 10 == 0:
			top_player = gen[list(cull(scores,1))[0]]
			directory = 'data_'+str(d)+'/gen'+str(gen_counter)+'/'
			if not os.path.exists(directory):
				os.makedirs(directory)
			np.save(directory+'W1.npy',top_player['W1'])
			np.save(directory+'W2.npy',top_player['W2'])
			np.save(directory+'W3.npy',top_player['W3'])


		gen_counter += 1

	directory = 'data_'+str(d)+'/top_player/'
	if not os.path.exists(directory):
		os.makedirs(directory)

	top_player = gen[list(cull(scores,1))[0]]

	#with open('data/top_player.txt', 'w') as f:
	#	json.dump(top_player,f)

	np.save(directory+'W1.npy',top_player['W1'])
	np.save(directory+'W2.npy',top_player['W2'])
	np.save(directory+'W3.npy',top_player['W3'])


	return top_player


class Reinforce:

	"""
	for parallelism - could either parallelise the games,
	then parallelise the grad descent:
	OR
	parallelise the entire process, i.e. string together play
	and replay.

	In a parallel scheme, the updates will be kept separate.
	"""


	def __init__(self,eta,mini_batch_size,
		opponent_update_freq,nn_class,bot_class,game_class):
		self.mini_batch_size = mini_batch_size
		self.opponent_update_freq = opponent_update_freq
		self.nn = nn_class
		self.bot = bot_class
		self.game = game_class()
		self.eta = eta

		self.policies = [self.bot([99,100,100,81],self.sigmoid,self.sigmoid_prime,self.game)]
		self.main_policy = self.bot([99,100,100,81],self.sigmoid,self.sigmoid_prime,self.game)
	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_prime(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def tanh_prime(self,z):
		return 1-(np.tanh(z)**2)

	def batch_tournament(self):

		opponent = np.random.choice(self.policies)
		results=[]
		# first pass
		print 'running batch'
		for match in xrange(self.mini_batch_size):
			results.append(
				self.game.play_game(True,False,self.main_policy,opponent)
				)
		print 'finished first pass'

		return results


	def update_mini_batch(self,mini_batch):

		# with game results and moves, we can construct a vector of 
		# actual outcomes (0's and 1's) for winners, 0's and -1's for losers
		# versus feedforward probabilites

		#Because of my construction, the x input for each is the game board
		# each time.
		# loop through game results (replaying) and run mini_batch each time.
		# can be parallelised by working on each game separately.
		
		nabla_b = [np.zeros(b.shape) for b in self.main_policy.biases]
		nabla_w = [np.zeros(w.shape) for w in self.main_policy.weights]
		print 'replaying and updating'
		#for x, y in mini_batch:
		for winner,moves in mini_batch:
			if winner == 0:
				continue
			# reinitialize
			board = {'microboard': [0 for i in range(81)],
			'macroboard': [-1 for i in range(9)],
			'win_macroboard': [-1 for i in range(9)],
			'next_turn': int(moves[0][0])}

			for pid,move in moves:
				#board['next_turn'] = pid
				move_int = move[1] * 9 + move[0]
				dummy = (1 if winner == pid else -1)
				#action_boards = self.game.successors(board)
				y = [dummy if a == move_int else 0 for a in xrange(81)]
				# for a in lmoves:
				# 	if a == move:
				# 		y = (1 if winner == pid else -1)
				# 	else:
				# 		y = (-1 if winner == pid else 1)
				
				delta_nabla_b, delta_nabla_w = self.main_policy.backprop(board, y)
				nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
				nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

				board = self.game.make_move(board,move)

		self.main_policy.weights = [w-(self.eta/len(mini_batch))*nw 
						for w, nw in zip(self.main_policy.weights, nabla_w)]
		self.main_policy.biases = [b-(self.eta/len(mini_batch))*nb 
					   for b, nb in zip(self.main_policy.biases, nabla_b)]
		print 'updated mini_batch'
		pass

	def generational_update(self):

		for n in xrange(self.num_generations):
			print n
			results = self.batch_tournament()
			self.update_mini_batch(results)
		print 'DONE'


######### PARALLEL METHODS
	def piped_play_replay(self,opponent):
		record=True
		verbose=False
		
		
		out = self.game.play_game(record,verbose,
			self.main_policy,opponent)
		update = self.update_single_game(out)
		return update
		

	def parallel_play_replay(self,num_cores):
		
		pool = mp.Pool(processes = num_cores)
		input_list = np.random.choice(self.policies,self.mini_batch_size)
		res = pool.map(self.piped_play_replay,input_list)

		pool.close()
		pool.join()
		return res


	def update_single_game(self,game_outcome):

		# with game results and moves, we can construct a vector of 
		# actual outcomes (0's and 1's) for winners, 0's and -1's for losers
		# versus feedforward probabilites

		#Because of my construction, the x input for each is the game board
		# each time.
		# loop through game results (replaying) and run mini_batch each time.
		# can be parallelised by working on each game separately.
		
		nabla_b = [np.zeros(b.shape) for b in self.main_policy.biases]
		nabla_w = [np.zeros(w.shape) for w in self.main_policy.weights]
		#for x, y in mini_batch:
		winner = game_outcome[0]
		moves = game_outcome[1]	
	#	for winner,moves in game_outcome:
#		if winner == 0:
#			continue
		# reinitialize
		board = {'microboard': [0 for i in range(81)],
		'macroboard': [-1 for i in range(9)],
		'win_macroboard': [-1 for i in range(9)],
		'next_turn': int(moves[0][0])}

		for pid,move in moves:
			#board['next_turn'] = pid
			move_int = move[1] * 9 + move[0]
			dummy = (1 if winner == pid else -1)
			#action_boards = self.game.successors(board)
			y = [dummy if a == move_int else 0 for a in xrange(81)]
			# for a in lmoves:
			# 	if a == move:
			# 		y = (1 if winner == pid else -1)
			# 	else:
			# 		y = (-1 if winner == pid else 1)
			
			delta_nabla_b, delta_nabla_w = self.main_policy.backprop(board, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

			board = self.game.make_move(board,move)

		return nabla_w,nabla_b



	def generational_update_parallel(self,num_cores,gens):

		for n in xrange(gens):
			print n
			if n % self.opponent_update_freq == 0:
				pol_copy = copy.deepcopy(self.main_policy)
				self.policies.append(pol_copy)
			updates = self.parallel_play_replay(num_cores)
			updates_w = updates[0][0]
			updates_b = updates[0][1]
			for i in xrange(1,len(updates)):
				updates_w = updates_w + updates[i][0]
				updates_b = updates_b + updates[i][1]

			# updates_unzipped = zip(*updates)
			# delta_nabla_b = [np.zeros(b.shape) for b in self.main_policy.biases]
			# delta_nabla_w = [np.zeros(w.shape) for w in self.main_policy.weights]

			# nabla_b = [nb+dnb for nb, dnb in zip(delta_nabla_b, updates_unzipped[0])]
			# nabla_w = [nw+dnw for nw, dnw in zip(delta_nabla_w, updates_unzipped[1])]

			self.main_policy.weights = [w-(self.eta/self.mini_batch_size)*nw 
			for w, nw in zip(self.main_policy.weights, updates_w)]
			self.main_policy.biases = [b-(self.eta/self.mini_batch_size)*nb 
			for b, nb in zip(self.main_policy.biases, updates_b)]


		print 'DONE'


	def board_outcome_dataset_maker(self,N):
		dataset = []
		for n in xrange(N):
			dataset.append([self.game.play_game_interrupt(False,50,self.main_policy,self.main_policy)])
		
		return dataset





if __name__ == '__main__':
	num_cores = mp.cpu_count()

	# X = parallel_evolve(
	# 	N_gen=2,
	# 	N_players=4,
	# 	matches_per_player=1,
	# 	carry_forward=1,
	# 	sigma=0.05,
	# 	d=4,
	# 	num_cores=num_cores,
	# 	verbose=False)
	
	# #schedule = generate_schedule(4,2)
	# # gen = regeneration(None,2,4)
	# # scores = play_tournament(schedule,gen,4,False)
	# # #scores = play_parallel_tourn(schedule,gen,4,1)
	# # print scores

	from game_rules import UTTT
	import bots


	reinforce = Reinforce(
		eta=0.003,
		mini_batch_size=12,
		opponent_update_freq=500,
		nn_class=Network,
		bot_class=bots.PolicyBot,
		game_class=UTTT)

	# t1= time.time()
	# results = reinforce.batch_tournament()
	# reinforce.update_mini_batch(results)
	# t2 = time.time()
	# print t2-t1
	# # t1= time.time()
	# # reinforce.generational_update()
	# # t2 = time.time()
	# print t2-t1

	# t1 = time.time()
	# updates= reinforce.parallel_play_replay(4)
	# t2 = time.time()
	# print t2-t1

	t1 = time.time()
	reinforce.generational_update_parallel(4,2)
	t2 = time.time()
	print (t2-t1)/10

	t1 = time.time()
	dataset = reinforce.board_outcome_dataset_maker(1000)
	t2 = time.time()
	print (t2-t1)/10	
	# reinforce.opponent = np.random.choice(reinforce.policies)
	# out = reinforce.game.play_game(True,True,reinforce.main_policy,reinforce.opponent)
	# update = reinforce.update_single_game(out)

	#######


	#######
