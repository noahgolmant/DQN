
from dqn import *
from cartpole import CartPole

T = 1000000
UPDATE_TIME = 100

def evaluate(index):
    game = CartPole()
    actions = game.legal_actions
    dqn = DQN(actions)
    dqn.epsilon = 0
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("networks")
    if checkpoint:
        saver.restore(sess, checkpoint.all_model_checkpoint_paths[index])
        print "Loaded: %s" % checkpoint.all_model_checkpoint_paths[index]
    rewards = []
    for episode in range(1000):
        state = game.newGame()
        totReward = 0
        for _ in range(200):
            if episode == 999:
                game.env.render()
            action = dqn.selectAction(state)
            actionNum = np.argmax(action)
            next_state, reward, game_over = game.next(actionNum)
            totReward += reward
            state = next_state
            if game_over:
                break  
        rewards.append(totReward)
    
    print "Average %s, best %s" % (sum(rewards) / len(rewards), max(rewards))

if __name__ == '__main__':
    game = CartPole()
    actions = game.legal_actions
    dqn = DQN(actions)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    state = game.newGame()
    #state = np.stack((state,state, state, state), axis=0)    
    for episode in range(T):
        action = dqn.selectAction(state)
        actionNum = np.argmax(action)

        next_state, reward, game_over = game.next(actionNum)
        # next_state = np.append(state[1:], next_state).reshape((4,4))#[:,:,1:]
        
        if game_over:
            dqn.storeExperience(state, action, 0, next_state, game_over)
            next_state = game.newGame()
        else:
            dqn.storeExperience(state, action, reward, next_state, game_over)

        minibatch = dqn.sampleExperiences()
        state_batch = [experience[0] for experience in minibatch]
        nextState_batch = [experience[3] for experience in minibatch]
        action_batch = [experience[1] for experience in minibatch]
        terminal_batch = [experience[4] for experience in minibatch]
        reward_batch = [experience[2] for experience in minibatch]

        y_batch = []
        Q_batch = sess.run(dqn.targetQNet.QValue, feed_dict = {dqn.targetQNet.stateInput: nextState_batch} )
        for i in range(len(minibatch)): 
            terminal = terminal_batch[i]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_batch[i]))
        
        currentQ_batch = sess.run(dqn.currentQNet.QValue,
                                  feed_dict = {dqn.currentQNet.stateInput: state_batch })

        sess.run(dqn.trainStep, feed_dict = {dqn.yInput: y_batch, dqn.actionInput: action_batch, dqn.currentQNet.stateInput: state_batch})
     
        state = next_state

        print "Time Step %s" % (episode) 

        if episode % UPDATE_TIME == 0:
            sess.run(dqn.copyCurrentToTargetOperation())

        if episode % 25000 == 0:
            saver.save(sess, 'networks/' + 'dqn', global_step= episode)
        if dqn.epsilon > FINAL_EPSILON:
            dqn.epsilon -= abs(FINAL_EPSILON - INITIAL_EPSILON) / 50000


