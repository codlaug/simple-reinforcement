const tf = require('@tensorflow/tfjs')
const { MinMaxScaler } = require('machinelearn/preprocessing')
// const data = require('./testdata.js')




const fs = require('fs');

const {mkdir} = require('shelljs');


const TradingAgent = require('./agent');
const {TradingGame} = require('./game');
const {copyWeights} = require('./dqn');
const IOHandler = require('./io_handler');

class MovingAverager {
  constructor(bufferLength) {
    this.buffer = [];
    for (let i = 0; i < bufferLength; ++i) {
      this.buffer.push(null);
    }
  }

  append(x) {
    this.buffer.shift();
    this.buffer.push(x);
  }

  average() {
    return this.buffer.reduce((x, prev) => x + prev) / this.buffer.length;
  }
}

/**
 * Train an agent to play the snake game.
 *
 * @param {SnakeGameAgent} agent The agent to train.
 * @param {number} batchSize Batch size for training.
 * @param {number} gamma Reward discount rate. Must be a number >= 0 and <= 1.
 * @param {number} learnigRate
 * @param {number} cumulativeRewardThreshold The threshold of moving-averaged
 *   cumulative reward from a single game. The training stops as soon as this
 *   threshold is achieved.
 * @param {number} maxNumFrames Maximum number of frames to train for.
 * @param {number} syncEveryFrames The frequency at which the weights are copied
 *   from the online DQN of the agent to the target DQN, in number of frames.
 * @param {string} savePath Path to which the online DQN of the agent will be
 *   saved upon the completion of the training.
 * @param {string} logDir Directory to which TensorBoard logs will be written
 *   during the training. Optional.
 */
async function train(agent, gamma, learningRate, cumulativeRewardThreshold, maxNumFrames, savePath) {

    const syncEveryFrames = 100;
    const batchSize = BATCH_SIZE;

  for (let i = 0; i < agent.replayBufferSize; ++i) {
    agent.playStep();
  }

  // Moving averager: cumulative reward across 100 most recent 100 episodes.
  const rewardAverager100 = new MovingAverager(100);
  // Moving averager: fruits eaten across 100 most recent 100 episodes.
  const eatenAverager100 = new MovingAverager(100);
  
  const optimizer = tf.train.sgd(learningRate);
  let tPrev = new Date().getTime();
  let frameCountPrev = agent.frameCount;
  let averageReward100Best = -Infinity;
  while (true) {
    agent.trainOnReplayBatch(batchSize, gamma, optimizer);
    const {cumulativeReward, done, tradesMade} = agent.playStep();
    if (done) {
      const t = new Date().getTime();
      const framesPerSecond = (agent.frameCount - frameCountPrev) / (t - tPrev) * 1e3;
      tPrev = t;
      frameCountPrev = agent.frameCount;

      rewardAverager100.append(cumulativeReward);
      eatenAverager100.append(tradesMade);
      const averageReward100 = rewardAverager100.average();
      const averageEaten100 = eatenAverager100.average();

      console.log(
          `Frame #${agent.frameCount}: ` +
          `cumulativeReward100=${averageReward100.toFixed(1)}; ` +
          `eaten100=${averageEaten100.toFixed(2)} ` +
          `(epsilon=${agent.epsilon.toFixed(3)}) ` +
          `(${framesPerSecond.toFixed(1)} frames/s)`);

      if (averageReward100 >= cumulativeRewardThreshold ||
          agent.frameCount >= maxNumFrames) {
        // TODO(cais): Save online network.
        break;
      }
      if (averageReward100 > averageReward100Best) {
        averageReward100Best = averageReward100;
        if (savePath != null) {
          if (!fs.existsSync(savePath)) {
            mkdir('-p', savePath);
          }
          await agent.onlineNetwork.save(IOHandler);
          console.log(`Saved DQN to ${savePath}`);
        }
      }
    }
    if (agent.frameCount % syncEveryFrames === 0) {
      copyWeights(agent.targetNetwork, agent.onlineNetwork);
      console.log('Sync\'ed weights from online network to target network');
    }
  }
}

const EPSILON_INIT = 0.5;
const EPSILON_FINAL = 0.01;
const EPSILON_DECAY_FRAMES = 1000

const BATCH_SIZE = 64;
const GAMMA = 0.99;
const LEARNING_RATE = 0.01;
const REWARD_THRESHOLD = 10000;
const MAX_FRAMES = 1000000

async function main() {

  const game = new TradingGame({});
  const agent = new TradingAgent(game, {
    replayBufferSize: 1000,
    epsilonInit: EPSILON_INIT,
    epsilonFinal: EPSILON_FINAL,
    epsilonDecayFrames: EPSILON_DECAY_FRAMES
  });

  await train(
      agent, GAMMA, LEARNING_RATE,
      REWARD_THRESHOLD, MAX_FRAMES,
      './models/dqn');
}

if (require.main === module) {
  main();
}
