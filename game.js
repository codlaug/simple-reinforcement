
const tf = require('@tensorflow/tfjs');


const ACTION_HOLD = 0;
const ACTION_BUY = 1;
const ACTION_SELL = 2;

const ALL_ACTIONS = [ACTION_HOLD, ACTION_BUY, ACTION_SELL];
const NUM_ACTIONS = ALL_ACTIONS.length;

module.exports.ALL_ACTIONS = ALL_ACTIONS
module.exports.NUM_ACTIONS = NUM_ACTIONS

const FEATURES = 5;
module.exports.FEATURES = FEATURES;
const LOOKBACKS = 6;
module.exports.LOOKBACKS = LOOKBACKS;



function getRandomInteger(min, max) {
  // Note that we don't reuse the implementation in the more generic
  // `getRandomIntegers()` (plural) below, for performance optimization.
  return Math.floor((max - min) * Math.random()) + min;
}

/**
 * Generate a random action among all possible actions.
 *
 * @return {0 | 1 | 2} Action represented as a number.
 */
module.exports.getRandomAction = function getRandomAction() {
  return getRandomInteger(0, NUM_ACTIONS);
}




module.exports.TradingGame = class TradingGame {
  /**
   * Constructor of SnakeGame.
   *
   * @param {object} args Configurations for the game. Fields include:
   *   - height {number} height of the board (positive integer).
   *   - width {number} width of the board (positive integer).
   *   - numFruits {number} number of fruits present on the screen
   *     at any given step.
   *   - initLen {number} initial length of the snake.
   */
  constructor(args) {
    if (args == null) {
      args = {};
    }
    

    this.data = []

    for(let i = 0; i < 40; ++i) {
      this.data[i] = Math.ceil(1+Math.cos(i))
    }

    this.reset();
  }

  /**
   * Reset the state of the game.
   *
   * @return {object} Initial state of the game.
   *   See the documentation of `getState()` for details.
   */
  reset() {
    this.currentIndex = 0;
    this.assets = 0;
    this.currency = 50;
    this.startingCurrency = this.currency;
    this.lastBuyPrice = null;
    return this.getState();
  }

  /**
   * Perform a step of the game.
   *
   * @param {0 | 1 | 2 | 3} action The action to take in the current step.
   *   The meaning of the possible values:
   *     0 - hold
   *     1 - buy
   *     2 - sell
   * @return {object} Object with the following keys:
   *   - `reward` {number} the reward value.
   *     - 0 if no fruit is eaten in this step
   *     - 1 if a fruit is eaten in this step
   *   - `state` New state of the game after the step.
   *   - `fruitEaten` {boolean} Whether a fruit is easten in this step.
   *   - `done` {boolean} whether the game has ended after this step.
   *     A game ends when the head of the snake goes off the board or goes
   *     over its own body.
   */
  step(action) {
    
    // Calculate the coordinates of the new head and check whether it has
    // gone off the board, in which case the game will end.
    let done = false;

    // Check if the head goes over the snake's body, in which case the
    // game will end.
    if(this.currentIndex >= this.data.length-2) {
        done = true;
    }
    

    if (done) {
      const endingBalance = (this.currency + (this.data[this.currentIndex] * this.assets)) - this.startingCurrency
      console.log('DONE~! REWARD:', endingBalance)
      return {reward: endingBalance, done};
    }

    const assetPrice = this.data[this.currentIndex]


    // Check if a fruit is eaten.
    let reward = 0;
    
    
    if(action === ACTION_BUY) {
      if(assetPrice === 1) {
        reward += 10;
      } else {
        console.log('wrong buy');
      }
      if(this.currency > 0) {
        const currencyToSpend = this.currency; // Math.min(10, this.currency);
        this.assets += currencyToSpend / assetPrice
        this.currency -= currencyToSpend
        
        this.lastBuyPrice = assetPrice;

        // console.log(`BUY @${assetPrice} with ${currencyToSpend} currency`, `assets: ${this.assets}`, `currency: ${this.currency}`)
      }
    } else if(action === ACTION_SELL) {
      if(assetPrice === 2) {
        reward += 10;
      } else {
        console.log('wrong sell');
      }
      if(this.assets > 0) {
        const assetsToSell = this.assets; // Math.min(15, this.assets);
        this.assets -= assetsToSell
        this.currency += assetsToSell * assetPrice
        
        // if(this.lastBuyPrice) {
        //     reward += this.data[this.currentIndex] - this.lastBuyPrice;
        //     this.lastBuyPrice = null;
        // }

        
        // console.log(`SELL @${assetPrice}`, `assets: ${this.assets}`, `currency: ${this.currency}`)
      }
    }

    this.currentIndex += 1

    const state = this.getState();
    return {reward, state, done, tradeMade: action !== ACTION_HOLD};
  }



  /**
   * Get plain JavaScript representation of the game state.
   *
   * @return An object with two keys:
   *   - s: {Array<[number, number]>} representing the squares occupied by
   *        the snake. The array is ordered in such a way that the first
   *        element corresponds to the head of the snake and the last
   *        element corresponds to the tail.
   *   - f: {Array<[number, number]>} representing the squares occupied by
   *        the fruit(s).
   */
  // getState() {
  //   return {
  //     assets: this.assets,
  //     currency: this.currency,
  //     price: this.data[this.currentIndex],
  //     nextPrice: this.data[this.currentIndex+1]
  //   }
  // }

  getState() {
    return {
      assets: this.assets,
      currency: this.currency,
      price: this.data[this.currentIndex],
      nextPrice: this.data[this.currentIndex+1],
      lastBuyPrice: this.lastBuyPrice
    }
  }
}

/**
 * Get the current state of the game as an image tensor.
 *
 * @param {object | object[]} state The state object as returned by
 *   `SnakeGame.getState()`, consisting of two keys: `s` for the snake and
 *   `f` for the fruit(s). Can also be an array of such state objects.
 * @param {number} h Height.
 * @param {number} w With.
 * @return {tf.Tensor} A tensor of shape [numExamples, height, width, 2] and
 *   dtype 'float32'
 *   - The first channel uses 0-1-2 values to mark the snake.
 *     - 0 means an empty square.
 *     - 1 means the body of the snake.
 *     - 2 means the head of the snake.
 *   - The second channel uses 0-1 values to mark the fruits.
 *   - `numExamples` is 1 if `state` argument is a single object or an
 *     array of a single object. Otherwise, it will be equal to the length
 *     of the state-object array.
 */

module.exports.getStateTensor = function getStateTensor(state) {
  if (!Array.isArray(state)) {
    state = [state];
  }
  const numExamples = state.length;
  // TODO(cais): Maintain only a single buffer for efficiency.
  const buffer = tf.buffer([numExamples, FEATURES]);

  for (let n = 0; n < numExamples; ++n) {
    if (state[n] == null) {
      continue;
    }

    buffer.set(state[n].assets, n, 3);
    buffer.set(state[n].currency, n, 4);
    buffer.set(state[n].price, n, 0);
    buffer.set(state[n].nextPrice, n, 1);
    buffer.set(state[n].lastBuyPrice, n, 2);
  }
  return buffer.toTensor();
}