class NeuralNetwork {
  static ActivationFunctions = {
    Sigmoid: (x) => 1 / (1 + Math.exp(-x)),
    ReLU: (x) => Math.max(0, x),
    Step: (x, limit) => (x > limit ? 1 : 0),
  };

  constructor(neuronCounts) {
    this.levels = [];
    for (let i = 0; i < neuronCounts.length - 1; i++) {
      this.levels.push(new Level(neuronCounts[i], neuronCounts[i + 1]));
    }
  }

  toJSON(){
    return {
      levels: this.levels.map(level => level.toJSON())
    }
  }

  static fromJSON(object){
    const network = new NeuralNetwork([]);
    network.levels = object.levels.map(level => Level.fromJSON(level));
    return network;
  }

  static fetch(uri){
    return new Promise((resolve, reject) => {
      fetch(uri).then(response => {
        if (!response.ok) throw new Error(`Modelo não encontrado em ${uri}`);
        return response.json();
      }).then(data => {
        const network = NeuralNetwork.fromJSON(data);
        resolve(network);
      }).catch(err => reject(err));
    });
  }

  static train(network, inputs, outputs, learningRate = 0.1) {
    if (inputs.length !== outputs.length) {
      throw new Error("Input and output devem ter o mesmo tamanho");
    }

    for (let i = 0; i < inputs.length; i++) {
      let predictedOutputs = this.feedForward(inputs[i], network);

      let errors = [];
      for (let j = 0; j < outputs[i].length; j++) {
        errors.push(outputs[i][j] - predictedOutputs[j]);
      }

      for (let k = network.levels.length - 1; k >= 0; k--) {
        let level = network.levels[k];
        let nextErrors = new Array(level.inputs.length).fill(0);

        for (let o = 0; o < level.outputs.length; o++) {
          let outputError = errors[o];
          level.biases[o] += outputError * learningRate;

          for (let i = 0; i < level.inputs.length; i++) {
            nextErrors[i] += outputError * level.weights[i][o];
            level.weights[i][o] += level.inputs[i] * outputError * learningRate;
          }
        }

        errors = nextErrors;
      }
    }
  }

  static feedForward(inputs, network) {
    let outputs = inputs;
    for (let i = 0; i < network.levels.length; i++) {
      outputs = Level.feedForward(outputs, network.levels[i]);
    }
    return outputs;
  }

  static mutate(network, amount = 1) {
    network.levels.forEach((level) => {
      for (let i = 0; i < level.biases.length; i++) {
        level.biases[i] = lerp(level.biases[i], Math.random() * 2 - 1, amount);
      }
      for (let i = 0; i < level.weights.length; i++) {
        for (let j = 0; j < level.weights[i].length; j++) {
          level.weights[i][j] = lerp(
            level.weights[i][j],
            Math.random() * 2 - 1,
            amount
          );
        }
      }
    });
  }
}

class Level {
  constructor(inputCount, outputCount) {
    this.inputs = new Array(inputCount);
    this.outputs = new Array(outputCount);
    this.biases = new Array(outputCount);

    this.weights = [];
    for (let i = 0; i < inputCount; i++) {
      this.weights[i] = new Array(outputCount);
    }

    Level.randomize(this);
  }

  static randomize(level) {
    for (let i = 0; i < level.inputs.length; i++) {
      for (let j = 0; j < level.outputs.length; j++) {
        level.weights[i][j] = Math.random() * 2 - 1;
      }
    }

    for (let i = 0; i < level.biases.length; i++) {
      level.biases[i] = Math.random() * 2 - 1;
    }
  }

  static feedForward(inputs, level) {
    level.inputs = inputs;

    for (let i = 0; i < level.outputs.length; i++) {
      let sum = 0;
      for (let j = 0; j < level.inputs.length; j++) {
        sum += level.inputs[j] * level.weights[j][i];
      }
      sum += level.biases[i];
      level.outputs[i] = NeuralNetwork.ActivationFunctions.Sigmoid(sum);
    }

    return level.outputs;
  }

  toJSON(){
    return {
      inputs: this.inputs,
      outputs: this.outputs,
      biases: this.biases,
      weights: this.weights
    }
  }

  static fromJSON(object){
    let level = new Level(object.inputs.length, object.outputs.length);
    level.inputs = object.inputs;
    level.outputs = object.outputs;
    level.biases = JSON.parse(JSON.stringify(object.biases));
    level.weights = JSON.parse(JSON.stringify(object.weights));
    return level;
  }
}

function lerp(A, B, t) {
  return A + (B - A) * t;
}
