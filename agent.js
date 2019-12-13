
const tf = require('@tensorflow/tfjs');

const {createDeepQNetwork} = require('./dqn');
const {getRandomAction, NUM_ACTIONS, ALL_ACTIONS, getStateTensor} = require('./game');
const ReplayMemory = require('./replay_memory');

const STATE_INDEX = 0;
const DONE_INDEX = 3;
const ACTION_INDEX = 1;
const REWARD_INDEX = 2;
const NEXT_STATE_INDEX = 4;
const GOAL_INDEX = 5;

module.exports = class TradingAgent {
  /**
   * Constructor of SnakeGameAgent.
   *
   * @param {SnakeGame} game A game object.
   * @param {object} config The configuration object with the following keys:
   *   - `replayBufferSize` {number} Size of the replay memory. Must be a
   *     positive integer.
   *   - `epsilonInit` {number} Initial value of epsilon (for the epsilon-
   *     greedy algorithm). Must be >= 0 and <= 1.
   *   - `epsilonFinal` {number} The final value of epsilon. Must be >= 0 and
   *     <= 1.
   *   - `epsilonDecayFrames` {number} The # of frames over which the value of
   *     `epsilon` decreases from `episloInit` to `epsilonFinal`, via a linear
   *     schedule.
   */
  constructor(game, config) {

    this.game = game;

    this.epsilonInit = config.epsilonInit;
    this.epsilonFinal = config.epsilonFinal;
    this.epsilonDecayFrames = config.epsilonDecayFrames;
    this.epsilonIncrement_ = (this.epsilonFinal - this.epsilonInit) / this.epsilonDecayFrames;

    this.onlineNetwork = createDeepQNetwork(NUM_ACTIONS);
    this.targetNetwork = createDeepQNetwork(NUM_ACTIONS);

    // this.optimizer = tf.train.sgd(config.learningRate);

    this.replayBufferSize = config.replayBufferSize;
    this.replayMemory = new ReplayMemory(config.replayBufferSize);
    this.frameCount = 0;
    this.reset();
  }

  reset() {
    this.cumulativeReward_ = 0;
    this.tradesMade_ = 0;
    this.transitions = [];
    this.game.reset();
  }

  /**
   * Play one step of the game.
   *
   * @returns {number | null} If this step leads to the end of the game,
   *   the total reward from the game as a plain number. Else, `null`.
   */
  playStep() {
    this.epsilon = this.frameCount >= this.epsilonDecayFrames ?
        this.epsilonFinal :
        this.epsilonInit + this.epsilonIncrement_  * this.frameCount;
    this.frameCount++;

    // The epsilon-greedy algorithm.
    let action;
    const state = this.game.getState();
    if (Math.random() < this.epsilon) {
      // Pick an action at random.
      action = getRandomAction();
    } else {
      // Greedily pick an action based on online DQN output.
      tf.tidy(() => {
        const stateTensor = getStateTensor(state)
        const goal = state;
        goal.assets *= 2;
        goal.currency *= 2;
        const goalTensor = getStateTensor(goal);
        // console.log(stateTensor.arraySync())
        const prediction = this.onlineNetwork.predict(tf.concat([stateTensor, goalTensor], 1));
        // console.log(prediction.arraySync()[0])
        action = ALL_ACTIONS[prediction.argMax(-1).dataSync()[0]];
      });
    }

    const {state: nextState, reward, done, tradeMade} = this.game.step(action);
    const goal = nextState;
    if(typeof goal !== 'undefined') {
      goal.assets *= 2;
      goal.currency *= 2;
    }

    this.transitions.push([state, action, reward, false, nextState]);

    this.replayMemory.append([state, action, reward, done, nextState, goal]);
    // console.log([state, action, reward, done, nextState, goal])

    this.cumulativeReward_ += reward;
    if (tradeMade) {
      this.tradesMade_++;
    }
    const output = {
      action,
      cumulativeReward: this.cumulativeReward_,
      done,
      tradesMade: this.tradesMade_
    };
    
    if (done) {
      for(let i = 0; i < this.transitions.length; ++i) {
        const hindsightGoal = state;
        const hindsightReplay = this.transitions[i];
        hindsightReplay.push(hindsightGoal);
        if(typeof hindsightReplay[4] !== 'undefined' && hindsightReplay[4].assets === hindsightGoal.assets && hindsightReplay[4].currency === hindsightGoal.currency) {
          console.log('GOAL');
          hindsightReplay[2] = 10;
        }
        this.replayMemory.append(hindsightReplay);
      }
      this.reset();
    }
    return output;
  }

  /// HINDSIGHT EXPERINCE REPLAY

  /**
   * Perform training on a randomly sampled batch from the replay buffer.
   *
   * @param {number} batchSize Batch size.
   * @param {number} gamma Reward discount rate. Must be >= 0 and <= 1.
   * @param {tf.train.Optimizer} optimizer The optimizer object used to update
   *   the weights of the online network.
   */
  trainOnReplayBatch(batchSize, gamma, optimizer) {
    // Get a batch of examples from the replay buffer.
    const batch = this.replayMemory.sample(batchSize);
    // console.log('batch', batch.map(example => example[1]))
    const lossFunction = () => tf.tidy(() => {
      const stateTensor = getStateTensor(batch.map(example => example[STATE_INDEX]));
      // console.log('state', stateTensor.arraySync());
      const actionTensor = tf.tensor1d(batch.map(example => example[ACTION_INDEX]), 'int32');
      // console.log('online', this.onlineNetwork.apply(stateTensor, {training: true}).arraySync())
      // console.log('actions', tf.oneHot(actionTensor, NUM_ACTIONS).arraySync())

      // get current state action values
      // const qs = this.onlineNetwork.apply(stateTensor, {training: true}).mul(tf.oneHot(actionTensor, NUM_ACTIONS)).sum(-1);
      
      const rewardTensor = tf.tensor1d(batch.map(example => example[REWARD_INDEX]));
      const nextStateTensor = getStateTensor(batch.map(example => example[NEXT_STATE_INDEX]));
      const goalStateTensor = getStateTensor(batch.map(example => example[GOAL_INDEX]));

      const stateGoalBatch = tf.concat([stateTensor, goalStateTensor], 1);
      const nonFinalNextStatesGoal = tf.concat([nextStateTensor, goalStateTensor], 1);

      const stateActionValues = this.onlineNetwork.apply(stateGoalBatch, {training: true}).mul(tf.oneHot(actionTensor, NUM_ACTIONS)).sum(-1);;
      // console.log(stateActionValues);

      // get next state values according to target network
      const nextMaxQTensor = this.targetNetwork.predict(nonFinalNextStatesGoal).max(-1);
      const doneMask = tf.scalar(1).sub(tf.tensor1d(batch.map(example => example[DONE_INDEX])).asType('float32'));
      const targetQs = rewardTensor.add(nextMaxQTensor.mul(doneMask).mul(gamma));
      // console.log('targetQs', targetQs.arraySync())
      // console.log('qs', qs.arraySync())
      return tf.losses.softmaxCrossEntropy(targetQs, stateActionValues);
    });

    // Calculate the gradients of the loss function with repsect to the weights
    // of the online DQN.
    const grads = tf.variableGrads(lossFunction);
    // Use the gradients to update the online DQN's weights.
    optimizer.applyGradients(grads.grads);
    tf.dispose(grads);
    // TODO(cais): Return the loss value here?
  }
}