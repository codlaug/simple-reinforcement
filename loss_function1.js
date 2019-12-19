const tf = require('@tensorflow/tfjs');
const { getStateTensor, NUM_ACTIONS } = require('./game');

const STATE_INDEX = 0;
const DONE_INDEX = 3;
const ACTION_INDEX = 1;
const REWARD_INDEX = 2;
const NEXT_STATE_INDEX = 4;

Array.prototype.pluck = function(key) {
    return this.map(item => item[key]);
}

// updating Q values
// Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])

// Q values are the maximum expected future rewards for action at each state

module.exports = function getLossFunction(onlineNetwork, targetNetwork, batch, gamma) {
    return () => tf.tidy(() => {
        const stateTensor = getStateTensor(batch.pluck(STATE_INDEX));
        // console.log('state', stateTensor.arraySync());
        const actionTensor = tf.tensor1d(batch.pluck(ACTION_INDEX), 'int32');
        // console.log('online', this.onlineNetwork.apply(stateTensor, {training: true}).arraySync())
        // console.log('actions', tf.oneHot(actionTensor, NUM_ACTIONS).arraySync())

        // predicted future rewards for actions at state
        let qs = onlineNetwork.apply(stateTensor, {training: true});
        // console.log('qs', qs.arraySync()[0]);
        // console.log('multiple qs by', tf.oneHot(actionTensor, NUM_ACTIONS).arraySync());
        
        // predicted future rewards for action chosen at state
        qs = qs.mul(tf.oneHot(actionTensor, NUM_ACTIONS));
        // console.log('qs masked by actions taken at that state', qs.arraySync());
        // collapsed into just one value (from onehot)
        qs = qs.sum(-1);
        // console.log('qs -1 now', qs.arraySync());
        
        // rewards given at each state
        const rewardTensor = tf.tensor1d(batch.pluck(REWARD_INDEX));
        const nextStateTensor = getStateTensor(batch.pluck(NEXT_STATE_INDEX));
        // const goalStateTensor = getStateTensor(batch.pluck(GOAL_INDEX));


        // TD target is just the reward of taking that action at that state plus the discounted highest Q value for the next state

        // predicted future rewards for each action at the next state
        let nextMaxQTensor = targetNetwork.predict(nextStateTensor);
        // console.log('nextQs', nextMaxQTensor.arraySync());

        // best predicted future reward from nextState
        nextMaxQTensor = nextMaxQTensor.max(-1);

        // console.log('nextMaxQs', nextMaxQTensor.arraySync());
        // console.log('nextMaxQTensor', nextMaxQTensor.arraySync());
        // const doneMask = tf.scalar(1).sub(tf.tensor1d(batch.pluck(DONE_INDEX)).asType('float32'));
        // best predicted future reward added together with current reward
        const targetQs = rewardTensor.add(nextMaxQTensor.mul(gamma));
        // console.log('rewards', rewardTensor.arraySync());
        // console.log('targetQs', targetQs.arraySync());
        // console.log('qs', qs.arraySync())
        const loss = tf.losses.meanSquaredError(targetQs, qs);
        // console.log('loss', loss.arraySync());
        return loss;
    });
};