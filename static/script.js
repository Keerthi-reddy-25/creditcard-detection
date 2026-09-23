// script.js
// Handles: submitting the transaction form, reading the threshold slider
// and model dropdown, and dynamically refreshing prediction + metrics +
// confusion matrix.

const form = document.getElementById("txn-form");
const slider = document.getElementById("threshold-slider");
const thresholdValueEl = document.getElementById("threshold-value");
const modelSelect = document.getElementById("model-select");

const probabilityEl = document.getElementById("probability");
const predictionLabelEl = document.getElementById("prediction-label");

const accuracyEl = document.getElementById("accuracy");
const precisionEl = document.getElementById("precision");
const recallEl = document.getElementById("recall");

const tnEl = document.getElementById("tn");
const fpEl = document.getElementById("fp");
const fnEl = document.getElementById("fn");
const tpEl = document.getElementById("tp");

const dynamicExplanationEl = document.getElementById("dynamic-explanation");

// Cache the last predicted probability PER MODEL, so switching models or
// moving the threshold slider can instantly re-classify the current
// transaction on the client, without calling /predict again.
let lastProbabilities = {};

function currentThreshold() {
  return parseFloat(slider.value);
}

function currentModel() {
  return modelSelect.value;
}

function updateThresholdLabel() {
  thresholdValueEl.textContent = currentThreshold().toFixed(2);
}

function renderPrediction(probability, threshold) {
  const isFraud = probability >= threshold;
  probabilityEl.textContent = (probability * 100).toFixed(2) + "%";
  predictionLabelEl.textContent = isFraud ? "FRAUD" : "NOT FRAUD";
  predictionLabelEl.className = "badge " + (isFraud ? "fraud" : "not-fraud");
}

function getFormValues() {
  return {
    Amount: parseFloat(document.getElementById("Amount").value),
    Hour: parseInt(document.getElementById("Hour").value, 10),
    Distance_From_Home: parseFloat(document.getElementById("Distance_From_Home").value),
    Is_Foreign: document.getElementById("Is_Foreign").checked ? 1 : 0,
    Is_Online: document.getElementById("Is_Online").checked ? 1 : 0,
  };
}

// --- Predict fraud probability for the entered transaction, for BOTH
//     models at once, so switching the dropdown afterwards needs no
//     extra network call. ---
async function submitTransaction(e) {
  e.preventDefault();
  const base = getFormValues();
  const modelKeys = Array.from(modelSelect.options).map((o) => o.value);

  const results = await Promise.all(
    modelKeys.map((key) =>
      fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...base, threshold: currentThreshold(), model: key }),
      }).then((res) => res.json())
    )
  );

  modelKeys.forEach((key, i) => {
    lastProbabilities[key] = results[i].probability;
  });

  renderPrediction(lastProbabilities[currentModel()], currentThreshold());
}

// --- Fetch accuracy / precision / recall / confusion matrix for the
//     current threshold + selected model ---
async function refreshMetrics() {
  const threshold = currentThreshold();
  const model = currentModel();
  const res = await fetch(`/metrics?threshold=${threshold}&model=${model}`);
  const data = await res.json();

  accuracyEl.textContent = (data.accuracy * 100).toFixed(2) + "%";
  precisionEl.textContent = (data.precision * 100).toFixed(2) + "%";
  recallEl.textContent = (data.recall * 100).toFixed(2) + "%";

  tnEl.textContent = data.confusion_matrix.true_negative;
  fpEl.textContent = data.confusion_matrix.false_positive;
  fnEl.textContent = data.confusion_matrix.false_negative;
  tpEl.textContent = data.confusion_matrix.true_positive;

  updateDynamicExplanation(data);
}

function updateDynamicExplanation(data) {
  const { false_positive, false_negative } = data.confusion_matrix;
  dynamicExplanationEl.textContent =
    `${data.model_name} @ threshold ${data.threshold.toFixed(2)}: ` +
    `${false_positive} false positive(s) (genuine transactions wrongly flagged) ` +
    `and ${false_negative} false negative(s) (real fraud missed), giving ` +
    `${(data.precision * 100).toFixed(1)}% precision and ` +
    `${(data.recall * 100).toFixed(1)}% recall on the test set.`;
}

// --- When the slider OR model dropdown changes: refresh metrics, and
//     re-classify the current transaction using cached probabilities ---
function onControlsChange() {
  updateThresholdLabel();
  refreshMetrics();
  const cached = lastProbabilities[currentModel()];
  if (cached !== undefined) {
    renderPrediction(cached, currentThreshold());
  }
}

form.addEventListener("submit", submitTransaction);
slider.addEventListener("input", onControlsChange);
modelSelect.addEventListener("change", onControlsChange);

// Initial load
updateThresholdLabel();
refreshMetrics();
