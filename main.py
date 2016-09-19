
from dqn import *
from atari import Atari

T = 100000
UPDATE_TIME = 10000

if __name__ == '__main__':
    atari = Atari('breakout.bin')
    actions = atari.legal_actions
    dqn = DQN(actions)
    state = atari.newGame()

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for _ in range(T):
        action = dqn.selectAction(state)

        next_state, reward, game_over = atari.next(action)

        dqn.storeExperience(state, action, reward, next_state, game_over)

        minibatch = dqn.sampleExperiences()
        state_batch = [experience[0] for experience in minibatch]
        nextState_batch = [experience[3] for experience in minibatch]
        action_batch = [experience[1] for experience in minibatch]
        terminal_batch = [experience[4] for experience in minibatch]
        reward_batch = [experience[2] for experience in minibatch]

        y_batch = []
        Q_batch = sess.run(dqn.targetQNet.QValue, feed_dict = {self.stateInput: nextState_batch} )
        for i in range(len(minibatch)):
            terminal = terminal_batch[i]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_batch[i]))
        
        currentQ_batch = sess.run(dqn.currentQNet.QValue,
                                  feed_dict = {self.stateInput: state_batch })


        y_input = tf.placeholder("float", [None])
        action_input = tf.placeholder("float", [None, len(actions)])
        Q_action = tf.reduce_sum(tf.mul(currentQ_batch, action_input), reduction_indices=[1])

        loss = tf.reduce_mean(tf.square(y_input - Q_action))
        trainStep = tf.train.RMSPropOptimizer(RMS_LEARNING_RATE, RMS_DECAY, RMS_MOMENTUM, RMS_EPSILON).minimize(loss)
        
        sess.run(trainStep, feed_dict = {y_input: y_batch, action_input: action_batch})
 
        state = next_state
        if T % UPDATE_TIME == 0:
            sess.run(dqn.copyCurrentToTargetOperation())

