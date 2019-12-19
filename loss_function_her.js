const tf = require('@tensorflow/tfjs');
const { getStateTensor } = require('./game');

const STATE_INDEX = 0;
const DONE_INDEX = 3;
const ACTION_INDEX = 1;
const REWARD_INDEX = 2;
const NEXT_STATE_INDEX = 4;
const GOAL_INDEX = 5;


Array.prototype.pluck = function(key) {
    return this.map(item => item[key]);
}


// updating Q values
// Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])

module.exports = function getLossFunction(onlineNetwork, targetNetwork, batch, gamma) {
    return () => tf.tidy(() => {
        const stateTensor = getStateTensor(batch.pluck(STATE_INDEX));
        // console.log('state', stateTensor.arraySync());
        const actionTensor = tf.tensor1d(batch.pluck(ACTION_INDEX), 'int32');
        // console.log('online', this.onlineNetwork.apply(stateTensor, {training: true}).arraySync())
        // console.log('actions', tf.oneHot(actionTensor, NUM_ACTIONS).arraySync())

        // get current state action values
        // #Obtain the Q' values by feeding the new state through our network
        // Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
        // const qs = this.onlineNetwork.apply(stateTensor, {training: true}).mul(tf.oneHot(actionTensor, NUM_ACTIONS)).sum(-1);
        
        const rewardTensor = tf.tensor1d(batch.pluck(REWARD_INDEX));
        const nextStateTensor = getStateTensor(batch.pluck(NEXT_STATE_INDEX));
        const goalStateTensor = getStateTensor(batch.pluck(GOAL_INDEX));

        const stateGoalBatch = tf.concat([stateTensor, goalStateTensor], 1);
        const nonFinalNextStatesGoal = tf.concat([nextStateTensor, goalStateTensor], 1);

        const stateActionValues = onlineNetwork.apply(stateGoalBatch, {training: true}).mul(tf.oneHot(actionTensor, NUM_ACTIONS)).sum(-1);;
        // console.log(stateActionValues);

        // get next state values according to target network
        const nextMaxQTensor = targetNetwork.predict(nonFinalNextStatesGoal).max(-1);
        const doneMask = tf.scalar(1).sub(tf.tensor1d(batch.pluck(DONE_INDEX)).asType('float32'));
        const targetQs = rewardTensor.add(nextMaxQTensor.mul(gamma));
        // console.log('targetQs', targetQs.arraySync())
        // console.log('qs', qs.arraySync())
        return tf.losses.softmaxCrossEntropy(targetQs, stateActionValues);
    });
};