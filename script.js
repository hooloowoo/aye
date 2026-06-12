const XOR_DATA = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 }
];

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoidFromOutput(y) {
  return y * (1 - y);
}

function tanh(x) {
  return Math.tanh(x);
}

function dtanhFromOutput(y) {
  return 1 - y * y;
}

function activate(x, isOutputLayer) {
  return isOutputLayer ? sigmoid(x) : tanh(x);
}

function activationDerivativeFromOutput(y, isOutputLayer) {
  return isOutputLayer ? dsigmoidFromOutput(y) : dtanhFromOutput(y);
}

function randWeight() {
  return (Math.random() * 2 - 1) * 0.9;
}

class TinyNN {
  constructor(inputSize, hiddenSize, hiddenLayers, outputSize, learningRate) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.hiddenLayers = hiddenLayers;
    this.outputSize = outputSize;
    this.learningRate = learningRate;
    this.reset();
  }

  reset() {
    this.epoch = 0;

    const layerSizes = [
      this.inputSize,
      ...Array.from({ length: this.hiddenLayers }, () => this.hiddenSize),
      this.outputSize
    ];

    this.weights = [];
    this.biases = [];

    for (let layer = 1; layer < layerSizes.length; layer += 1) {
      const prevSize = layerSizes[layer - 1];
      const currSize = layerSizes[layer];
      const scale = Math.sqrt(2 / (prevSize + currSize));

      this.weights.push(
        Array.from({ length: currSize }, () =>
          Array.from({ length: prevSize }, () => randWeight() * scale)
        )
      );
      this.biases.push(Array.from({ length: currSize }, () => randWeight() * scale));
    }

    this.lastForward = null;
  }

  resizeHidden(newHiddenSize, newHiddenLayers = this.hiddenLayers) {
    this.hiddenSize = newHiddenSize;
    this.hiddenLayers = newHiddenLayers;
    this.reset();
  }

  setLearningRate(lr) {
    this.learningRate = lr;
  }

  forward(inputs) {
    const activations = [[...inputs]];
    const zs = [];

    for (let layer = 0; layer < this.weights.length; layer += 1) {
      const isOutputLayer = layer === this.weights.length - 1;
      const prev = activations[layer];
      const z = this.weights[layer].map((row, i) => {
        const weighted = row.reduce((sum, w, j) => sum + w * prev[j], 0);
        return weighted + this.biases[layer][i];
      });
      const a = z.map((v) => activate(v, isOutputLayer));

      zs.push(z);
      activations.push(a);
    }

    this.lastForward = { activations, zs, inputs: [...inputs], output: activations[activations.length - 1] };
    return activations[activations.length - 1];
  }

  trainSample(inputs, target) {
    const output = this.forward(inputs)[0];
    const error = target - output;
    const layerCount = this.weights.length;
    const deltas = Array.from({ length: layerCount }, () => []);

    const outputLayerIndex = layerCount - 1;
    const outputActivations = this.lastForward.activations[outputLayerIndex + 1];

    deltas[outputLayerIndex] = outputActivations.map((a, i) => {
      const outError = (i === 0 ? target : 0) - a;
      return outError;
    });

    for (let layer = outputLayerIndex - 1; layer >= 0; layer -= 1) {
      const currentActivations = this.lastForward.activations[layer + 1];
      const nextWeights = this.weights[layer + 1];
      const nextDeltas = deltas[layer + 1];

      deltas[layer] = currentActivations.map((a, i) => {
        let propagated = 0;
        for (let k = 0; k < nextDeltas.length; k += 1) {
          propagated += nextWeights[k][i] * nextDeltas[k];
        }
        return propagated * activationDerivativeFromOutput(a, false);
      });
    }

    for (let layer = 0; layer < layerCount; layer += 1) {
      const prevActivations = this.lastForward.activations[layer];
      for (let i = 0; i < this.weights[layer].length; i += 1) {
        for (let j = 0; j < this.weights[layer][i].length; j += 1) {
          this.weights[layer][i][j] += this.learningRate * deltas[layer][i] * prevActivations[j];
        }
        this.biases[layer][i] += this.learningRate * deltas[layer][i];
      }
    }

    const loss = error * error;
    return { output, error, loss, target };
  }

  trainEpoch(samples) {
    let totalLoss = 0;
    const shuffled = [...samples].sort(() => Math.random() - 0.5);
    for (const sample of shuffled) {
      totalLoss += this.trainSample(sample.x, sample.y).loss;
    }
    this.epoch += 1;
    return totalLoss / samples.length;
  }

  evaluate(samples) {
    const snapshot = this.lastForward;
    let correct = 0;
    let totalLoss = 0;

    for (const sample of samples) {
      const prediction = this.forward(sample.x)[0];
      const cls = prediction >= 0.5 ? 1 : 0;
      if (cls === sample.y) {
        correct += 1;
      }
      const error = sample.y - prediction;
      totalLoss += error * error;
    }

    this.lastForward = snapshot;

    return {
      accuracy: correct / samples.length,
      loss: totalLoss / samples.length
    };
  }
}

const canvas = document.getElementById("nn-canvas");
const ctx = canvas.getContext("2d");

const hiddenSlider = document.getElementById("hidden-neurons");
const hiddenValue = document.getElementById("hidden-neurons-val");
const hiddenLayersSlider = document.getElementById("hidden-layers");
const hiddenLayersValue = document.getElementById("hidden-layers-val");
const lrSlider = document.getElementById("learning-rate");
const lrValue = document.getElementById("learning-rate-val");
const speedSlider = document.getElementById("speed");
const speedValue = document.getElementById("speed-val");
const presetSelect = document.getElementById("preset-select");

const stepBtn = document.getElementById("step-btn");
const epochBtn = document.getElementById("epoch-btn");
const hundredEpochBtn = document.getElementById("hundred-epoch-btn");
const playBtn = document.getElementById("play-btn");
const resetBtn = document.getElementById("reset-btn");
const probeBtn = document.getElementById("probe-btn");

const probeX1 = document.getElementById("probe-x1");
const probeX2 = document.getElementById("probe-x2");

const epochEl = document.getElementById("epoch-value");
const lossEl = document.getElementById("loss-value");
const accEl = document.getElementById("acc-value");
const predEl = document.getElementById("pred-value");
const sampleEl = document.getElementById("sample-value");
const relProbEl = document.getElementById("rel-prob");
const relPredClassEl = document.getElementById("rel-pred-class");
const relTargetClassEl = document.getElementById("rel-target-class");
const relMatchEl = document.getElementById("rel-match");
const statusPill = document.getElementById("status-pill");
const liveTip = document.getElementById("live-tip");

let timer = null;
let sampleCursor = 0;
let animationMs = Number(speedSlider.value);
let lastUpdate = null;
let currentSampleText = "[0,0] -> 0";
let currentSampleTarget = 0;

const PRESETS = {
  stable: { hiddenNeurons: 4, hiddenLayers: 2, learningRate: 0.22, speed: 220 },
  fast: { hiddenNeurons: 3, hiddenLayers: 1, learningRate: 0.45, speed: 140 },
  deep: { hiddenNeurons: 5, hiddenLayers: 3, learningRate: 0.15, speed: 240 }
};

const nn = new TinyNN(
  2,
  Number(hiddenSlider.value),
  Number(hiddenLayersSlider.value),
  1,
  Number(lrSlider.value)
);

function mix(a, b, t) {
  return a + (b - a) * t;
}

function getNodeLayout() {
  const padX = 120;
  const padY = 80;
  const xInput = padX;
  const xOutput = canvas.width - padX;

  const inputNodes = [0, 1].map((_, i) => ({
    x: xInput,
    y: mix(padY, canvas.height - padY, i / 1),
    r: 18
  }));

  const hiddenLayers = [];
  for (let l = 0; l < nn.hiddenLayers; l += 1) {
    const x = mix(xInput + 120, xOutput - 120, nn.hiddenLayers === 1 ? 0.5 : l / (nn.hiddenLayers - 1));
    hiddenLayers.push(
      Array.from({ length: nn.hiddenSize }, (_, i) => ({
        x,
        y: nn.hiddenSize === 1 ? canvas.height / 2 : mix(padY, canvas.height - padY, i / (nn.hiddenSize - 1)),
        r: 15
      }))
    );
  }

  const outputNodes = [{ x: xOutput, y: canvas.height / 2, r: 18 }];
  return { inputNodes, hiddenLayers, outputNodes };
}

function drawConnection(from, to, weight, pulse = 0) {
  const abs = Math.min(1, Math.abs(weight));
  const thickness = 1 + abs * 5;
  const positive = weight >= 0;

  const baseColor = positive ? [11, 122, 117] : [217, 79, 61];
  const alpha = 0.18 + abs * 0.55;

  ctx.strokeStyle = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, ${alpha.toFixed(3)})`;
  ctx.lineWidth = thickness;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();

  if (pulse > 0) {
    const px = mix(from.x, to.x, pulse);
    const py = mix(from.y, to.y, pulse);
    ctx.fillStyle = positive ? "rgba(11,122,117,0.85)" : "rgba(217,79,61,0.85)";
    ctx.beginPath();
    ctx.arc(px, py, 4 + abs * 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawNode(node, value, tone, label) {
  const raw = value ?? 0;
  const normalized = raw >= 0 && raw <= 1 ? raw : (raw + 1) / 2;
  const v = Math.max(0, Math.min(1, normalized));
  const glow = 10 + v * 18;

  ctx.save();
  ctx.shadowColor = `rgba(${tone.join(",")}, ${0.25 + v * 0.45})`;
  ctx.shadowBlur = glow;
  ctx.fillStyle = `rgba(${tone.join(",")}, ${0.28 + v * 0.55})`;
  ctx.beginPath();
  ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
  ctx.fill();

  ctx.shadowBlur = 0;
  ctx.lineWidth = 1.3;
  ctx.strokeStyle = "rgba(30,36,49,0.35)";
  ctx.stroke();

  ctx.fillStyle = "#1f2230";
  ctx.font = "600 13px IBM Plex Mono";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(value == null ? "?" : value.toFixed(2), node.x, node.y);

  ctx.fillStyle = "rgba(31,34,48,0.8)";
  ctx.font = "500 12px Space Grotesk";
  ctx.fillText(label, node.x, node.y + 33);
  ctx.restore();
}

function drawLabels() {
  ctx.fillStyle = "rgba(31, 34, 48, 0.78)";
  ctx.font = "700 12px Space Grotesk";
  ctx.fillText("Input Layer", 120, 26);
  ctx.fillText(`Hidden Layers (${nn.hiddenLayers})`, canvas.width / 2, 26);
  ctx.fillText("Output", canvas.width - 120, 26);
}

function render(pulse = 0) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const { inputNodes, hiddenLayers, outputNodes } = getNodeLayout();
  const forward = nn.lastForward;

  const allLayers = [inputNodes, ...hiddenLayers, outputNodes];
  for (let layer = 0; layer < allLayers.length - 1; layer += 1) {
    const fromNodes = allLayers[layer];
    const toNodes = allLayers[layer + 1];
    const weights = nn.weights[layer];

    for (let to = 0; to < toNodes.length; to += 1) {
      for (let from = 0; from < fromNodes.length; from += 1) {
        drawConnection(fromNodes[from], toNodes[to], weights[to][from], pulse);
      }
    }
  }

  drawLabels();

  inputNodes.forEach((node, i) => {
    drawNode(node, forward?.inputs?.[i], [11, 122, 117], `x${i + 1}`);
  });

  hiddenLayers.forEach((layerNodes, l) => {
    layerNodes.forEach((node, i) => {
      const val = forward?.activations?.[l + 1]?.[i];
      drawNode(node, val, [239, 179, 57], `h${l + 1}.${i + 1}`);
    });
  });

  outputNodes.forEach((node, i) => {
    const out = forward?.output?.[i];
    drawNode(node, out, [217, 79, 61], `y${i + 1}`);
  });
}

function updateRelationView(probability, targetClass) {
  if (probability == null || targetClass == null) {
    relProbEl.textContent = "-";
    relPredClassEl.textContent = "-";
    relTargetClassEl.textContent = "-";
    relMatchEl.textContent = "-";
    relMatchEl.className = "badge neutral";
    return;
  }

  const predictedClass = probability >= 0.5 ? 1 : 0;
  const isMatch = predictedClass === targetClass;

  relProbEl.textContent = probability.toFixed(3);
  relPredClassEl.textContent = String(predictedClass);
  relTargetClassEl.textContent = String(targetClass);
  relMatchEl.textContent = isMatch ? "MATCH" : "MISMATCH";
  relMatchEl.className = `badge ${isMatch ? "ok" : "bad"}`;
}

function updateMetrics(prediction = null, sampleText = null) {
  const stats = nn.evaluate(XOR_DATA);
  epochEl.textContent = String(nn.epoch);
  lossEl.textContent = stats.loss.toFixed(4);
  accEl.textContent = `${Math.round(stats.accuracy * 100)}%`;
  if (sampleText != null) {
    currentSampleText = sampleText;
  }
  sampleEl.textContent = currentSampleText;

  if (prediction == null) {
    predEl.textContent = "-";
    const inferred = nn.lastForward?.output?.[0];
    updateRelationView(inferred == null ? null : inferred, currentSampleTarget);
  } else {
    const asClass = prediction >= 0.5 ? 1 : 0;
    predEl.textContent = `${prediction.toFixed(3)} (class ${asClass})`;
    updateRelationView(prediction, currentSampleTarget);
  }
}

function setStatus(text, running) {
  statusPill.textContent = text;
  statusPill.style.background = running ? "#def4ea" : "#fffdf7";
  statusPill.style.borderColor = running ? "rgba(11, 122, 117, 0.35)" : "rgba(34, 40, 54, 0.16)";
}

function stepSingleSample() {
  const sample = XOR_DATA[sampleCursor % XOR_DATA.length];
  sampleCursor += 1;
  const report = nn.trainSample(sample.x, sample.y);
  nn.lastForward = nn.lastForward || { inputs: sample.x, activations: [], output: [report.output] };
  currentSampleTarget = sample.y;
  updateMetrics(report.output, `[${sample.x.join(",")}] -> ${sample.y}`);
  liveTip.textContent = `Sample [${sample.x.join(",")}] target ${sample.y} produced ${report.output.toFixed(3)} with loss ${report.loss.toFixed(4)}.`;
  return report.output;
}

function runEpoch() {
  const loss = nn.trainEpoch(XOR_DATA);
  updateMetrics();
  liveTip.textContent = `Completed epoch ${nn.epoch}. Mean loss = ${loss.toFixed(4)}.`;
}

function runManyEpochs(rounds) {
  let loss = 0;
  for (let i = 0; i < rounds; i += 1) {
    loss = nn.trainEpoch(XOR_DATA);
  }
  updateMetrics();
  liveTip.textContent = `Completed ${rounds} epochs. Last mean loss = ${loss.toFixed(4)}.`;
}

function animateOnce() {
  const start = performance.now();
  const duration = Math.max(160, animationMs);

  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    render(t);
    if (t < 1) {
      requestAnimationFrame(frame);
    }
  }

  requestAnimationFrame(frame);
}

function stopAuto() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  playBtn.textContent = "Auto Train";
  setStatus("Idle", false);
}

function startAuto() {
  stopAuto();
  setStatus("Training", true);
  playBtn.textContent = "Pause";

  timer = setInterval(() => {
    stepSingleSample();
    if (sampleCursor % XOR_DATA.length === 0) {
      nn.epoch += 1;
    }
    animateOnce();
  }, animationMs);
}

hiddenSlider.addEventListener("input", () => {
  hiddenValue.textContent = hiddenSlider.value;
});

hiddenLayersSlider.addEventListener("input", () => {
  hiddenLayersValue.textContent = hiddenLayersSlider.value;
});

function rebuildArchitecture() {
  nn.resizeHidden(Number(hiddenSlider.value), Number(hiddenLayersSlider.value));
  sampleCursor = 0;
  stopAuto();
  currentSampleText = "[0,0] -> 0";
  currentSampleTarget = 0;
  updateMetrics();
  render();
  liveTip.textContent = `Architecture updated to ${hiddenLayersSlider.value} hidden layers with ${hiddenSlider.value} neurons each.`;
}

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) {
    return;
  }

  hiddenSlider.value = String(preset.hiddenNeurons);
  hiddenLayersSlider.value = String(preset.hiddenLayers);
  lrSlider.value = String(preset.learningRate);
  speedSlider.value = String(preset.speed);

  hiddenValue.textContent = hiddenSlider.value;
  hiddenLayersValue.textContent = hiddenLayersSlider.value;
  lrValue.textContent = Number(lrSlider.value).toFixed(2);
  speedValue.textContent = `${speedSlider.value} ms`;

  animationMs = Number(speedSlider.value);
  nn.setLearningRate(Number(lrSlider.value));
  rebuildArchitecture();
  liveTip.textContent = `Preset "${name}" loaded. Architecture and training parameters updated.`;
}

hiddenSlider.addEventListener("change", rebuildArchitecture);
hiddenLayersSlider.addEventListener("change", rebuildArchitecture);

lrSlider.addEventListener("input", () => {
  const lr = Number(lrSlider.value);
  nn.setLearningRate(lr);
  lrValue.textContent = lr.toFixed(2);
  presetSelect.value = "custom";
});

speedSlider.addEventListener("input", () => {
  animationMs = Number(speedSlider.value);
  speedValue.textContent = `${animationMs} ms`;

  if (timer) {
    startAuto();
  }
  presetSelect.value = "custom";
});

presetSelect.addEventListener("change", () => {
  if (presetSelect.value === "custom") {
    return;
  }
  applyPreset(presetSelect.value);
});

stepBtn.addEventListener("click", () => {
  stopAuto();
  setStatus("Single Step", false);
  const output = stepSingleSample();
  animateOnce();
  lastUpdate = output;
});

epochBtn.addEventListener("click", () => {
  stopAuto();
  setStatus("Epoch Step", false);
  runEpoch();
  animateOnce();
});

hundredEpochBtn.addEventListener("click", () => {
  stopAuto();
  setStatus("Batch Epoch", false);
  runManyEpochs(100);
  animateOnce();
});

playBtn.addEventListener("click", () => {
  if (timer) {
    stopAuto();
    return;
  }
  startAuto();
});

resetBtn.addEventListener("click", () => {
  stopAuto();
  nn.reset();
  sampleCursor = 0;
  lastUpdate = null;
  updateMetrics();
  render();
  liveTip.textContent = "Model reset with fresh random weights. Start stepping again from scratch.";
});

probeBtn.addEventListener("click", () => {
  stopAuto();
  const x1 = probeX1.checked ? 1 : 0;
  const x2 = probeX2.checked ? 1 : 0;
  const target = x1 === x2 ? 0 : 1;
  currentSampleTarget = target;
  const pred = nn.forward([x1, x2])[0];
  updateMetrics(pred, `[${x1},${x2}] -> ${target}`);
  animateOnce();

  liveTip.textContent = `Forward pass for [${x1},${x2}] -> ${pred.toFixed(3)}. XOR target is ${target}.`;
  setStatus("Probing", false);
});

function bootstrap() {
  hiddenValue.textContent = hiddenSlider.value;
  hiddenLayersValue.textContent = hiddenLayersSlider.value;
  lrValue.textContent = Number(lrSlider.value).toFixed(2);
  speedValue.textContent = `${speedSlider.value} ms`;
  presetSelect.value = "custom";

  nn.forward([0, 0]);
  updateMetrics(null, currentSampleText);
  render();
}

bootstrap();
