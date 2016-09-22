import gym

class CartPole:

	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.env.reset()
		self.legal_actions = [0,1]

	def newGame(self):
		return self.env.reset()

	def next(self, action):
		nextstate, reward, done, info = self.env.step(action)

		if done:
			self.env.reset()
		return nextstate, reward, done


