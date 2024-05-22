const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

async function predictClassification(model, image) {
  try {
    // Decode the image, resize, normalize and add a batch dimension
    const tensor = tf.node.decodeImage(image, 3).resizeNearestNeighbor([224, 224]).expandDims().toFloat().div(tf.scalar(255.0));

    // Get the prediction and convert it to a data array
    const prediction = model.predict(tensor);
    const score = await prediction.data();

    // Assuming the model output is a binary classification
    const confidenceScore = score[0];
    const label = confidenceScore < 0.5 ? "Non-cancer" : "Cancer";

    let suggestion;

    if (label === "Cancer") {
      suggestion = "Segera periksa ke dokter!";
    } else {
      suggestion = "Anda sehat!";
    }

    return { label, suggestion, confidenceScore };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

module.exports = predictClassification;
