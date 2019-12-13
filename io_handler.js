const fs = require('fs')
const path = require('path')

const ensureDirectoryExistence = function (filePath) {
  var dirname = path.dirname(filePath)
  console.log('ensureDirectoryExistence', dirname)
  if (fs.existsSync(dirname)) {
    return true
  }
  ensureDirectoryExistence(dirname)
  fs.mkdirSync(dirname)
}


function toArrayBuffer(buf) {
    if (Array.isArray(buf)) {
      // An Array of Buffers.
      let totalLength = 0;
      for (const buffer of buf) {
        totalLength += buffer.length;
      }
  
      const ab = new ArrayBuffer(totalLength);
      const view = new Uint8Array(ab);
      let pos = 0;
      for (const buffer of buf) {
        pos += buffer.copy(view, pos);
      }
      return ab;
    } else {
      // A single Buffer. Return a copy of the underlying ArrayBuffer slice.
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
  }

  function loadWeights(weightsManifest, filePath) {
    const dirName = path.dirname(filePath);
    const buffers = [];
    const weightSpecs = [];
    for (const group of weightsManifest) {
      for (const p of group.paths) {
        const weightFilePath = path.join(dirName, p);
        const buffer = fs.readFileSync(weightFilePath)
        buffers.push(buffer);
      }
      weightSpecs.push(...group.weights);
    }
    return [weightSpecs, toArrayBuffer(buffers)];
  }


module.exports = {
  save: (modelArtifacts) => {
    const filePath = './models/dqn'
    console.log(filePath)
    ensureDirectoryExistence(filePath+'/anything.json');
  
    const weightsBinPath = path.join(filePath, 'weights.bin');
    const weightsManifest = [{
      paths: ['weights.bin'],
      weights: modelArtifacts.weightSpecs
    }];
    const modelJSON = {
      modelTopology: modelArtifacts.modelTopology,
      weightsManifest,
      format: modelArtifacts.format,
      generatedBy: modelArtifacts.generatedBy,
      convertedBy: modelArtifacts.convertedBy
    };
    if (modelArtifacts.trainingConfig != null) {
      modelJSON.trainingConfig = modelArtifacts.trainingConfig;
    }
    if (modelArtifacts.userDefinedMetadata != null) {
      modelJSON.userDefinedMetadata = modelArtifacts.userDefinedMetadata;
    }
    const modelJSONPath = path.join(filePath, 'model.json');
    fs.writeFileSync(modelJSONPath, JSON.stringify(modelJSON), 'utf8');
    fs.writeFileSync(weightsBinPath, Buffer.from(modelArtifacts.weightData), 'binary');
    
    return {
      // TODO(cais): Use explicit tfc.io.ModelArtifactsInfo type below once it
      // is available.
      // tslint:disable-next-line:no-any
      modelArtifactsInfo: true // getModelArtifactsInfoForJSON(modelArtifacts)
    }
  },
  load: async () => {
    const filePath = 'brain/model.json';
    const info = fs.statSync(filePath)

    // `path` can be either a directory or a file. If it is a file, assume
    // it is model.json file.
    if (info.isFile()) {
      const modelJSON = JSON.parse(fs.readFileSync(filePath, 'utf8'));

      const modelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy
      };
      if (modelJSON.weightsManifest != null) {
        const [weightSpecs, weightData] = await loadWeights(modelJSON.weightsManifest, filePath);
        modelArtifacts.weightSpecs = weightSpecs;
        modelArtifacts.weightData = weightData;
      }
      if (modelJSON.trainingConfig != null) {
        modelArtifacts.trainingConfig = modelJSON.trainingConfig;
      }
      if (modelJSON.userDefinedMetadata != null) {
        modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
      }
      return modelArtifacts;
    } else {
      throw new Error(
          'The path to load from must be a file. Loading from a directory ' +
          'is not supported.');
    }
  }
}