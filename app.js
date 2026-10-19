/**
 * Galton-LLM — Simulación Quasi-Física de Máquina de Galton con Knobs y Backpropagation
 * En la parte superior hay una vía con dos carriles ligeramente oblicuos que se van separando:
 * las bolas pequeñas (rojas) caen antes por la abertura entre carriles y las grandes (violetas) caen más adelante,
 * haciendo que todos los knobs de todas las filas tengan influencia real.
 */

(() => {
  // 24 palabras posibles en la parte inferior (Vocabulario de salida del LLM)
  const VOCABULARY = [
    { word: "el", role: "conn" },
    { word: "la", role: "conn" },
    { word: "gato", role: "noun" },
    { word: "perro", role: "noun" },
    { word: "duerme", role: "verb" },
    { word: "corre", role: "verb" },
    { word: "alfombra", role: "noun" },
    { word: "cama", role: "noun" },
    { word: "casa", role: "noun" },
    { word: "sol", role: "noun" },
    { word: "luna", role: "noun" },
    { word: "brilla", role: "verb" },
    { word: "cielo", role: "noun" },
    { word: "mar", role: "noun" },
    { word: "noche", role: "noun" },
    { word: "ciudad", role: "noun" },
    { word: "bosque", role: "noun" },
    { word: "jardín", role: "noun" },
    { word: "red", role: "noun" },
    { word: "modelo", role: "noun" },
    { word: "datos", role: "noun" },
    { word: "código", role: "noun" },
    { word: "palabra", role: "noun" },
    { word: "ruido", role: "noun" }
  ];

  // 6 tamaños diferentes de bola (0 = Roja/Pequeña -> 5 = Violeta/Grande)
  const BALL_TIERS = [
    {
      tier: 0,
      name: "Roja",
      sampleDecimal: "0.15",
      radiusRatio: 0.0045,
      minRadius: 3.9,
      maxRadius: 4.7,
      mass: 0.75,
      restitution: 0.52,
      colorCore: "#fca5a5",
      colorMid: "#ef4444",
      colorEdge: "#991b1b",
      glow: "rgba(239, 68, 68, 0.55)"
    },
    {
      tier: 1,
      name: "Naranja",
      sampleDecimal: "0.30",
      radiusRatio: 0.0053,
      minRadius: 4.6,
      maxRadius: 5.4,
      mass: 1.1,
      restitution: 0.49,
      colorCore: "#fdba74",
      colorMid: "#f97316",
      colorEdge: "#9a3412",
      glow: "rgba(249, 115, 22, 0.55)"
    },
    {
      tier: 2,
      name: "Amarilla",
      sampleDecimal: "0.45",
      radiusRatio: 0.0061,
      minRadius: 5.3,
      maxRadius: 6.2,
      mass: 1.5,
      restitution: 0.45,
      colorCore: "#fde047",
      colorMid: "#eab308",
      colorEdge: "#854d0e",
      glow: "rgba(234, 179, 8, 0.55)"
    },
    {
      tier: 3,
      name: "Verde",
      sampleDecimal: "0.62",
      radiusRatio: 0.0069,
      minRadius: 6.0,
      maxRadius: 7.0,
      mass: 2.0,
      restitution: 0.42,
      colorCore: "#86efac",
      colorMid: "#22c55e",
      colorEdge: "#166534",
      glow: "rgba(34, 197, 94, 0.55)"
    },
    {
      tier: 4,
      name: "Azul",
      sampleDecimal: "0.78",
      radiusRatio: 0.0078,
      minRadius: 6.7,
      maxRadius: 7.9,
      mass: 2.6,
      restitution: 0.38,
      colorCore: "#93c5fd",
      colorMid: "#3b82f6",
      colorEdge: "#1e40af",
      glow: "rgba(59, 130, 246, 0.55)"
    },
    {
      tier: 5,
      name: "Violeta",
      sampleDecimal: "0.95",
      radiusRatio: 0.0087,
      minRadius: 7.5,
      maxRadius: 8.8,
      mass: 3.3,
      restitution: 0.35,
      colorCore: "#d8b4fe",
      colorMid: "#a855f7",
      colorEdge: "#581c87",
      glow: "rgba(168, 85, 247, 0.6)"
    }
  ];

  const CONNECTORS = new Set([
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "en", "de", "con", "por", "para", "sin", "sobre", "tras",
    "a", "al", "del", "y", "e", "o", "u", "que", "se", "su", "mi", "tu", "es", "son"
  ]);

  // Asociaciones semánticas de un "LLM entrenado"
  const SEMANTIC_RULES = [
    {
      match: ["gato", "duerme"],
      targets: [
        { word: "alfombra", weight: 1.0 },
        { word: "cama", weight: 0.65 },
        { word: "casa", weight: 0.35 }
      ]
    },
    {
      match: ["gato"],
      targets: [
        { word: "duerme", weight: 0.85 },
        { word: "alfombra", weight: 0.7 },
        { word: "noche", weight: 0.4 }
      ]
    },
    {
      match: ["sol", "brilla"],
      targets: [
        { word: "cielo", weight: 1.0 },
        { word: "mar", weight: 0.65 },
        { word: "ciudad", weight: 0.3 }
      ]
    },
    {
      match: ["sol"],
      targets: [
        { word: "brilla", weight: 0.9 },
        { word: "cielo", weight: 0.75 }
      ]
    },
    {
      match: ["red", "aprende"],
      targets: [
        { word: "datos", weight: 1.0 },
        { word: "código", weight: 0.65 },
        { word: "palabra", weight: 0.4 }
      ]
    },
    {
      match: ["perro", "corre"],
      targets: [
        { word: "bosque", weight: 1.0 },
        { word: "jardín", weight: 0.75 },
        { word: "casa", weight: 0.35 }
      ]
    },
    {
      match: ["perro"],
      targets: [
        { word: "corre", weight: 0.9 },
        { word: "jardín", weight: 0.7 }
      ]
    },
    {
      match: ["modelo", "predice"],
      targets: [
        { word: "palabra", weight: 1.0 },
        { word: "datos", weight: 0.55 },
        { word: "código", weight: 0.4 }
      ]
    },
    {
      match: ["luna", "ilumina"],
      targets: [
        { word: "noche", weight: 1.0 },
        { word: "ciudad", weight: 0.65 },
        { word: "mar", weight: 0.5 }
      ]
    },
    {
      match: ["luna"],
      targets: [
        { word: "noche", weight: 0.95 },
        { word: "brilla", weight: 0.7 },
        { word: "cielo", weight: 0.5 }
      ]
    }
  ];

  // Referencias al DOM
  const canvas = document.getElementById("galton-canvas");
  const ctx = canvas.getContext("2d");
  const presetPromptSelect = document.getElementById("preset-prompt-select");
  const promptInput = document.getElementById("prompt-input");
  const tokenCountBadge = document.getElementById("token-count-badge");
  const tokenChipsContainer = document.getElementById("token-chips-container");

  const btnDropBurst = document.getElementById("btn-drop-burst");
  const btnDropSingle = document.getElementById("btn-drop-single");
  const btnClearBalls = document.getElementById("btn-clear-balls");
  const btnAppendWinner = document.getElementById("btn-append-winner");

  const presetBtns = {
    trained: document.getElementById("preset-trained"),
    random: document.getElementById("preset-random"),
    neutral: document.getElementById("preset-neutral")
  };

  const ballsPerWordSlider = document.getElementById("balls-per-word-slider");
  const ballsPerWordVal = document.getElementById("balls-per-word-val");
  const temperatureSlider = document.getElementById("temperature-slider");
  const temperatureVal = document.getElementById("temperature-val");
  const knobStrengthSlider = document.getElementById("knob-strength-slider");
  const knobStrengthVal = document.getElementById("knob-strength-val");

  // Controles de Backpropagation
  const targetWordSelect = document.getElementById("target-word-select");
  const learningRateSlider = document.getElementById("learning-rate-slider");
  const learningRateVal = document.getElementById("learning-rate-val");
  const bpEpochCountEl = document.getElementById("bp-epoch-count");
  const bpLossValEl = document.getElementById("bp-loss-val");
  const bpTargetProbEl = document.getElementById("bp-target-prob");
  const btnBackpropStep = document.getElementById("btn-backprop-step");
  const btnBackpropNudgeOnly = document.getElementById("btn-backprop-nudge-only");
  const btnAutoTrain = document.getElementById("btn-auto-train");
  const bpFeedbackLog = document.getElementById("bp-feedback-log");

  const winnerWordEl = document.getElementById("winner-word");
  const winnerStatsEl = document.getElementById("winner-stats");
  const completedSentenceEl = document.getElementById("completed-sentence");
  const top3CandidatesEl = document.getElementById("top3-candidates");

  const simStatusDot = document.getElementById("sim-status-dot");
  const simStatusText = document.getElementById("sim-status-text");
  const metricActiveBalls = document.getElementById("metric-active-balls");
  const metricLandedBalls = document.getElementById("metric-landed-balls");
  const hoveredKnobPill = document.getElementById("hovered-knob-pill");

  // Estado de la simulación
  const state = {
    width: 960,
    height: 700,
    dpr: 1,
    mode: "trained", // 'trained' | 'random' | 'neutral' | 'custom'
    ballsPerWord: 6,
    temperature: 0.25,
    knobStrength: 1.0,
    learningRate: 0.22,
    backpropTargetBinIndex: 6, // 'alfombra'
    bpEpochs: 0,
    autoTraining: false,
    autoTrainPendingStep: false,
    knobs: [],
    balls: [],
    spawnQueue: [],
    spawnTimer: 0,
    landedHistory: [],
    bpPulses: [],
    railDropRings: [], // destellos cuando una bola se cuela entre los dos carriles
    bins: VOCABULARY.map((item, index) => ({
      index,
      word: item.word,
      role: item.role,
      count: 0,
      weightedScore: 0,
      settledBalls: [],
      xLeft: 0,
      xRight: 0,
      xCenter: 0,
      flash: 0
    })),
    hoveredKnob: null,
    draggingKnob: null,
    dragStartY: 0,
    dragStartX: 0,
    dragStartWeight: 0,
    hoveredBinIndex: -1,
    targetBinsInfo: [],
    boardBounds: {
      left: 40,
      right: 920,
      railStartX: 68,
      railEndX: 892,
      railStartY: 28,
      railEndY: 74,
      knobsTop: 118,
      knobsBottom: 465,
      binsTop: 488,
      binsBottom: 625
    }
  };

  /**
   * Convierte un número decimal en [0.05, 0.99] en uno de los 6 niveles de tamaño/color de bola (0=Roja a 5=Violeta)
   */
  function decimalToTierSpec(decimalVal) {
    const clamped = Math.max(0.05, Math.min(0.99, decimalVal));
    const idx = Math.max(0, Math.min(5, Math.floor(((clamped - 0.05) / 0.94) * 6)));
    return BALL_TIERS[idx];
  }

  /**
   * Genera la combinación determinista de bolas (vector de números decimales) para una palabra.
   * Cada palabra produce una combinación de diferentes tamaños de bola (de roja a violeta).
   */
  function getWordDecimalVector(rawWord, dimensionCount) {
    const clean = rawWord.toLowerCase().replace(/[.,;:!?¿¡"']/g, "").trim() || "x";
    const isConn = CONNECTORS.has(clean);

    let hash = 2166136261;
    for (let i = 0; i < clean.length; i++) {
      hash ^= clean.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }

    const components = [];
    for (let d = 0; d < dimensionCount; d++) {
      let h = Math.imul(hash ^ ((d + 1) * 0x9e3779b9), 0x85ebca6b);
      h ^= h >>> 13;
      const pseudoRand = (Math.abs(h) % 1000) / 1000;

      // Garantizar variedad de tamaños dentro de la misma palabra
      const baseBand = ((d * 2 + Math.abs(hash)) % 6) / 5;
      let rawDecimal = baseBand * 0.68 + pseudoRand * 0.32;

      if (isConn) {
        rawDecimal = rawDecimal * 0.8 + 0.07;
      } else {
        rawDecimal = rawDecimal * 0.88 + 0.1;
      }

      const decimal = Math.max(0.08, Math.min(0.98, Math.round(rawDecimal * 100) / 100));
      const spec = decimalToTierSpec(decimal);

      components.push({
        decimal,
        tier: spec.tier,
        spec
      });
    }

    return components;
  }

  /**
   * Devuelve la coordenada Y central de la vía oblicua en una posición horizontal X
   */
  function getRailCenterYAtX(x) {
    const { railStartX, railEndX, railStartY, railEndY } = state.boardBounds;
    const t = Math.max(0, Math.min(1, (x - railStartX) / Math.max(1, railEndX - railStartX)));
    return railStartY + t * (railEndY - railStartY);
  }

  /**
   * Devuelve la mitad de la separación (abertura) entre los dos carriles divergentes en la coordenada X
   */
  function getRailHalfGapAtX(x) {
    const { railStartX, railEndX } = state.boardBounds;
    const t = Math.max(0, Math.min(1, (x - railStartX) / Math.max(1, railEndX - railStartX)));
    // A la izquierda están muy juntos (2.8px de semihueco) y hacia la derecha se abren hasta 10.5px
    return 2.8 + t * 7.8;
  }

  /**
   * Calcula la coordenada X exacta en la que la separación de los dos carriles supera el tamaño de la bola
   * (las bolas rojas pequeñas caen al principio a la izquierda; las violetas grandes llegan hasta la derecha)
   */
  function computeDropXForBall(decimalVal, tokenIndex, totalTokens) {
    const { railStartX, railEndX } = state.boardBounds;
    const trackLen = railEndX - railStartX;

    // El progreso a lo largo de la vía depende directamente del diámetro/valor decimal de la bola
    const normalizedSize = Math.max(0.04, Math.min(0.96, (decimalVal - 0.06) / 0.92));
    // Pequeño desfase según posición de la palabra y temperatura para repartir sobre todos los knobs vecinos
    const wordOffset = totalTokens > 1 ? ((tokenIndex / (totalTokens - 1)) - 0.5) * 0.045 : 0;
    const jitter = (Math.random() - 0.5) * (0.025 + state.temperature * 0.035);

    const tDrop = Math.max(0.04, Math.min(0.96, normalizedSize + wordOffset + jitter));
    return railStartX + tDrop * trackLen;
  }

  function initTargetWordSelect() {
    targetWordSelect.innerHTML = "";
    VOCABULARY.forEach((item, index) => {
      const opt = document.createElement("option");
      opt.value = String(index);
      opt.textContent = `🎯 “${item.word}” (Cubeta #${index + 1})`;
      targetWordSelect.appendChild(opt);
    });
    targetWordSelect.value = String(state.backpropTargetBinIndex);
  }

  function setBackpropTargetBin(binIndex, announce = true) {
    if (binIndex < 0 || binIndex >= VOCABULARY.length) return;
    state.backpropTargetBinIndex = binIndex;
    targetWordSelect.value = String(binIndex);
    updatePredictionUI();

    if (announce) {
      const targetWord = VOCABULARY[binIndex].word;
      bpFeedbackLog.innerHTML = `Palabra objetivo fijada en <strong>“${escapeHtml(targetWord)}”</strong>. Pulsa <em>1 Paso de Backprop</em> para reajustar los knobs hacia ella o gíralos a mano.`;
    }
  }

  function parsePromptTokens(text, customDimCount) {
    const dim = customDimCount || state.ballsPerWord;
    const rawParts = text
      .trim()
      .split(/\s+/)
      .map((w) => w.replace(/[.,;:!?¿¡"']/g, "").trim())
      .filter(Boolean);

    return rawParts.map((word, idx) => {
      const lower = word.toLowerCase();
      const vector = getWordDecimalVector(lower, dim);
      return {
        id: `${idx}-${lower}`,
        word: lower,
        vector
      };
    });
  }

  function computeTrainedTargetBins(tokens) {
    const words = tokens.map((t) => t.word);
    const wordSet = new Set(words);
    const lastWord = words[words.length - 1] || "";

    for (const rule of SEMANTIC_RULES) {
      const allMatched = rule.match.every((m) => wordSet.has(m));
      if (allMatched) {
        const filtered = rule.targets.filter((t) => t.word !== lastWord);
        const list = filtered.length > 0 ? filtered : rule.targets;
        return list
          .map((t) => ({
            binIndex: VOCABULARY.findIndex((v) => v.word === t.word),
            weight: t.weight,
            word: t.word
          }))
          .filter((t) => t.binIndex >= 0);
      }
    }

    if (lastWord === "la" || lastWord === "una") {
      const femCandidates = ["alfombra", "cama", "noche", "ciudad", "palabra", "casa"];
      const hash = words.join("").length % femCandidates.length;
      const primary = femCandidates[hash];
      const secondary = femCandidates[(hash + 2) % femCandidates.length];
      return [
        { binIndex: VOCABULARY.findIndex((v) => v.word === primary), weight: 1.0, word: primary },
        { binIndex: VOCABULARY.findIndex((v) => v.word === secondary), weight: 0.55, word: secondary }
      ];
    }

    if (lastWord === "el" || lastWord === "un" || lastWord === "al" || lastWord === "del") {
      const mascCandidates = ["cielo", "bosque", "mar", "jardín", "código", "modelo"];
      const hash = words.join("").length % mascCandidates.length;
      const primary = mascCandidates[hash];
      const secondary = mascCandidates[(hash + 1) % mascCandidates.length];
      return [
        { binIndex: VOCABULARY.findIndex((v) => v.word === primary), weight: 1.0, word: primary },
        { binIndex: VOCABULARY.findIndex((v) => v.word === secondary), weight: 0.55, word: secondary }
      ];
    }

    if (!CONNECTORS.has(lastWord)) {
      return [
        { binIndex: VOCABULARY.findIndex((v) => v.word === "duerme"), weight: 0.9, word: "duerme" },
        { binIndex: VOCABULARY.findIndex((v) => v.word === "brilla"), weight: 0.65, word: "brilla" },
        { binIndex: VOCABULARY.findIndex((v) => v.word === "corre"), weight: 0.5, word: "corre" }
      ];
    }

    return [
      { binIndex: VOCABULARY.findIndex((v) => v.word === "palabra"), weight: 1.0, word: "palabra" },
      { binIndex: VOCABULARY.findIndex((v) => v.word === "datos"), weight: 0.6, word: "datos" }
    ];
  }

  /**
   * Calcula el peso ideal w* en [-0.95, 0.95] para que un knob situado en knob.x
   * guíe tanto las bolas pequeñas (que caen por la izquierda) como las grandes (por la derecha) hacia targetX
   */
  function computeIdealKnobWeightForTargetX(knob, targetX) {
    const usableW = state.boardBounds.right - state.boardBounds.left;
    const layerProgress = knob.row / 7;
    const dx = targetX - knob.x;
    const normalizedDelta = dx / (usableW * 0.25);
    const focusGain = 0.78 + layerProgress * 0.45;
    const rawWeight = Math.tanh(normalizedDelta * focusGain);
    return Math.max(-0.95, Math.min(0.95, rawWeight));
  }

  /**
   * Construye la geometría de la vía oblicua superior de 2 carriles divergentes,
   * las 8 filas escalonadas de ancho completo (92 knobs) y las 24 cubetas inferiores.
   */
  function layoutBoard(preserveWeights = false) {
    const W = state.width;
    const H = state.height;

    const marginX = Math.max(20, W * 0.028);
    const left = marginX;
    const right = W - marginX;
    const usableW = right - left;

    // Vía superior con ligera pendiente hacia la derecha (escalada proporcionalmente a la altura H)
    const railStartX = left + usableW * 0.045;
    const railEndX = right - usableW * 0.045;
    const railStartY = Math.max(22, Math.min(34, H * 0.048));
    const railEndY = railStartY + Math.max(24, Math.min(42, H * 0.058));

    const knobsTop = railEndY + Math.max(24, Math.min(38, H * 0.055));
    const binsBottom = H - Math.max(54, Math.min(72, H * 0.11));
    const binsTop = binsBottom - Math.max(76, Math.min(125, H * 0.18));
    const knobsBottom = binsTop - Math.max(20, Math.min(32, H * 0.045));

    state.boardBounds = {
      left,
      right,
      railStartX,
      railEndX,
      railStartY,
      railEndY,
      knobsTop,
      knobsBottom,
      binsTop,
      binsBottom
    };

    // Actualizar las 24 cubetas inferiores
    const numBins = VOCABULARY.length;
    const binWidth = usableW / numBins;
    state.bins.forEach((bin, i) => {
      bin.xLeft = left + i * binWidth;
      bin.xRight = bin.xLeft + binWidth;
      bin.xCenter = bin.xLeft + binWidth * 0.5;
    });

    // 8 filas escalonadas de ancho completo (alternando 11 y 12 knobs = 92 knobs).
    const numRows = 8;
    const rowSpacingY = (knobsBottom - knobsTop) / (numRows - 1);
    const knobRadius = Math.max(6.5, Math.min(10.5, usableW * 0.0094, rowSpacingY * 0.25));

    const previousKnobs = preserveWeights ? state.knobs : null;
    const newKnobs = [];
    let globalIdx = 0;

    const baseSpan = usableW * 0.86;
    const stepX = baseSpan / 10; // distancia entre knobs de una fila de 11

    for (let r = 0; r < numRows; r++) {
      const isOddRow = r % 2 === 1;
      const count = isOddRow ? 12 : 11;
      const tRow = numRows === 1 ? 0 : r / (numRows - 1);
      const y = knobsTop + tRow * (knobsBottom - knobsTop);

      const rowSpan = (count - 1) * stepX;
      const rowLeft = W * 0.5 - rowSpan * 0.5;

      for (let c = 0; c < count; c++) {
        const x = rowLeft + c * stepX;
        const prev = previousKnobs && previousKnobs[globalIdx];
        const prevW = prev ? prev.weight : 0;

        newKnobs.push({
          id: globalIdx,
          row: r,
          col: c,
          x,
          y,
          radius: knobRadius,
          weight: prevW,
          displayWeight: prevW,
          hitEnergy: 0,
          bpGlow: 0,
          bpDelay: 0,
          lastDelta: 0
        });
        globalIdx++;
      }
    }

    state.knobs = newKnobs;
    if (!preserveWeights) {
      applyKnobPreset(state.mode);
    }
  }

  function applyKnobPreset(mode) {
    state.mode = mode;

    Object.entries(presetBtns).forEach(([key, btn]) => {
      if (btn) btn.classList.toggle("active", key === mode);
    });

    const tokens = parsePromptTokens(promptInput.value);
    state.targetBinsInfo = computeTrainedTargetBins(tokens);

    if (mode === "neutral") {
      state.knobs.forEach((knob) => {
        knob.weight = 0;
        knob.lastDelta = 0;
      });
      return;
    }

    if (mode === "random") {
      state.knobs.forEach((knob) => {
        const rand = Math.random() * 2 - 1;
        knob.weight = Math.round(rand * 100) / 100;
        knob.lastDelta = 0;
      });
      return;
    }

    if (mode === "trained") {
      const primaryTarget = state.targetBinsInfo[0] || { binIndex: 6, weight: 1.0 };
      const secondaryTarget = state.targetBinsInfo[1] || primaryTarget;

      const primaryX = state.bins[primaryTarget.binIndex]?.xCenter ?? state.width * 0.5;
      const secondaryX = state.bins[secondaryTarget.binIndex]?.xCenter ?? primaryX;

      state.knobs.forEach((knob) => {
        const targetX = knob.col % 5 === 0 && knob.row >= 3 ? secondaryX : primaryX;
        const idealW = computeIdealKnobWeightForTargetX(knob, targetX);
        knob.weight = Math.round(idealW * 100) / 100;
        knob.lastDelta = 0;
      });
    }
  }

  /**
   * Ejecuta un paso de Backpropagation:
   * Reajusta ligeramente los knobs que intervinieron en la caída (en cualquiera de las zonas de tamaño)
   * y los que forman el cauce hacia la palabra objetivo (y*).
   */
  function performBackpropStep(launchAfter = true) {
    const targetBin = state.bins[state.backpropTargetBinIndex];
    if (!targetBin) return;

    const targetX = targetBin.xCenter;
    const usableW = state.boardBounds.right - state.boardBounds.left;
    const lr = state.learningRate;

    const knobStats = new Map();

    for (const record of state.landedHistory) {
      for (const kId of record.touchedKnobs) {
        if (!knobStats.has(kId)) {
          knobStats.set(kId, { touches: 0, massSum: 0, sumLandedX: 0 });
        }
        const st = knobStats.get(kId);
        st.touches += 1;
        st.massSum += record.mass;
        st.sumLandedX += record.x * record.mass;
      }
    }

    state.mode = "custom";
    Object.values(presetBtns).forEach((btn) => btn && btn.classList.remove("active"));

    let adjustedCount = 0;
    let sumAbsDelta = 0;

    state.knobs.forEach((knob) => {
      const layerProgress = knob.row / 7;
      const touchInfo = knobStats.get(knob.id);
      const wasTouched = Boolean(touchInfo && touchInfo.touches > 0);

      // En las filas inferiores miramos además la cercanía al embudo convergente hacia targetX
      const distToTargetX = Math.abs(knob.x - targetX);
      const convergenceWidth = usableW * (0.45 - layerProgress * 0.22);
      const inConvergenceZone = distToTargetX <= convergenceWidth;

      if (!wasTouched && !inConvergenceZone) return;

      const idealW = computeIdealKnobWeightForTargetX(knob, targetX);
      let deltaW = 0;

      if (wasTouched) {
        const avgLandedX = touchInfo.sumLandedX / touchInfo.massSum;
        const empiricalPush = Math.tanh((targetX - avgLandedX) / (usableW * 0.22));
        const structuralDelta = idealW - knob.weight;
        const credit = Math.min(
          1.4,
          0.6 + (touchInfo.touches / Math.max(1, state.landedHistory.length)) * 1.5
        );
        deltaW = lr * credit * (0.65 * structuralDelta + 0.45 * empiricalPush);
      } else if (inConvergenceZone) {
        const zoneFactor = 1 - (distToTargetX / convergenceWidth) * 0.6;
        deltaW = lr * 0.55 * zoneFactor * (idealW - knob.weight);
      }

      deltaW = Math.max(-0.28, Math.min(0.28, deltaW));

      if (Math.abs(deltaW) >= 0.008) {
        const nextW = Math.max(-0.96, Math.min(0.96, knob.weight + deltaW));
        const actualDelta = nextW - knob.weight;
        knob.weight = Math.round(nextW * 100) / 100;
        knob.lastDelta = actualDelta;
        knob.bpDelay = (7 - knob.row) * 0.045;
        knob.bpGlow = 1.0;

        adjustedCount++;
        sumAbsDelta += Math.abs(actualDelta);
      }
    });

    spawnBackpropPulses(targetBin);

    state.bpEpochs += 1;
    bpEpochCountEl.textContent = String(state.bpEpochs);

    const avgDelta = adjustedCount > 0 ? (sumAbsDelta / adjustedCount).toFixed(2) : "0.00";
    bpFeedbackLog.innerHTML = `<strong>Época #${state.bpEpochs}:</strong> Reajustados <strong>${adjustedCount} knobs</strong> (Δ medio: ±${avgDelta}) hacia <strong>“${escapeHtml(targetBin.word)}”</strong>.`;

    if (launchAfter) {
      clearBallsAndBins(true);
      triggerBurstDrop();
    }
  }

  function spawnBackpropPulses(targetBin) {
    const { binsTop } = state.boardBounds;
    const activeKnobs = state.knobs.filter((k) => k.bpGlow > 0.2);
    const sampled = activeKnobs.filter((_, idx) => idx % 2 === 0).slice(0, 32);

    sampled.forEach((knob) => {
      state.bpPulses.push({
        startX: targetBin.xCenter + (Math.random() - 0.5) * 18,
        startY: binsTop,
        endX: knob.x,
        endY: knob.y,
        progress: -knob.bpDelay * 1.2,
        speed: 2.1 + Math.random() * 0.6,
        delta: knob.lastDelta
      });
    });
  }

  function renderTokenChips() {
    const tokens = parsePromptTokens(promptInput.value);
    if (tokenCountBadge) {
      tokenCountBadge.textContent = `${tokens.length} ${tokens.length === 1 ? "palabra" : "palabras"} · ${tokens.length * state.ballsPerWord} bolas`;
    }
    tokenChipsContainer.innerHTML = "";

    tokens.forEach((tok) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "token-chip";
      const decimalsStr = tok.vector.map((c) => c.decimal.toFixed(2)).join(", ");
      chip.title = `Vector de "${tok.word}": [${decimalsStr}] — Haz clic para lanzar solo esta palabra por los carriles`;

      const textSpan = document.createElement("span");
      textSpan.textContent = tok.word;

      const ballsWrap = document.createElement("span");
      ballsWrap.className = "token-chip-balls";

      tok.vector.forEach((comp) => {
        const dot = document.createElement("span");
        dot.className = "token-chip-dot";
        const px = 5 + comp.tier * 1.2;
        dot.style.width = `${px}px`;
        dot.style.height = `${px}px`;
        dot.style.background = comp.spec.colorMid;
        dot.style.boxShadow = `0 0 4px ${comp.spec.glow}`;
        dot.title = `${comp.decimal.toFixed(2)} (${comp.spec.name})`;
        ballsWrap.appendChild(dot);
      });

      chip.append(textSpan, ballsWrap);
      chip.addEventListener("click", () => {
        enqueueWordVectors([tok]);
      });
      tokenChipsContainer.appendChild(chip);
    });

    const suggestedTargets = computeTrainedTargetBins(tokens);
    state.targetBinsInfo = suggestedTargets;

    if (state.mode === "trained") {
      applyKnobPreset("trained");
    }
    updatePredictionUI();
  }

  function enqueueWordVectors(tokens) {
    if (!tokens.length) return;

    const maxDim = Math.max(...tokens.map((t) => t.vector.length));
    for (let d = 0; d < maxDim; d++) {
      tokens.forEach((tok, idx) => {
        const comp = tok.vector[d];
        if (!comp) return;
        state.spawnQueue.push({
          word: tok.word,
          decimal: comp.decimal,
          tier: comp.tier,
          spec: comp.spec,
          tokenIndex: idx,
          totalTokens: tokens.length
        });
      });
    }
    updateStatusUI();
  }

  /**
   * Instancia una bola en el inicio izquierdo de la vía superior de dos carriles divergentes
   */
  function spawnBallFromQueue(item) {
    const { railStartX, railStartY } = state.boardBounds;
    const usableW = state.boardBounds.right - state.boardBounds.left;

    const r = Math.max(
      item.spec.minRadius,
      Math.min(item.spec.maxRadius, usableW * item.spec.radiusRatio)
    );

    const dropX = computeDropXForBall(item.decimal, item.tokenIndex, item.totalTokens);

    state.balls.push({
      word: item.word,
      decimal: item.decimal,
      tier: item.tier,
      spec: item.spec,
      radius: r,
      mass: item.spec.mass,
      restitution: item.spec.restitution,
      x: railStartX,
      y: railStartY,
      vx: 320 + Math.random() * 45, // velocidad de rodadura a lo largo de los dos carriles oblicuos
      vy: 0,
      onRails: true,
      dropX,
      angle: 0,
      vAngle: 0,
      trail: [],
      touchedKnobs: new Set(),
      stuckTimer: 0,
      settled: false,
      binIndex: -1,
      age: 0
    });
  }

  function clearBallsAndBins(preserveEpochs = false) {
    state.balls = [];
    state.spawnQueue = [];
    state.landedHistory = [];
    state.railDropRings = [];
    state.bins.forEach((bin) => {
      bin.count = 0;
      bin.weightedScore = 0;
      bin.settledBalls = [];
      bin.flash = 0;
    });
    if (!preserveEpochs) {
      state.bpEpochs = 0;
      bpEpochCountEl.textContent = "0";
    }
    updatePredictionUI();
    updateStatusUI();
  }

  function updatePredictionUI() {
    const promptText = promptInput.value.trim();
    const totalLanded = state.bins.reduce((acc, b) => acc + b.count, 0);

    metricActiveBalls.textContent = String(state.balls.length + state.spawnQueue.length);
    metricLandedBalls.textContent = String(totalLanded);

    const targetBin = state.bins[state.backpropTargetBinIndex];
    if (totalLanded === 0 || !targetBin) {
      bpTargetProbEl.textContent = "0%";
      bpLossValEl.textContent = "—";
    } else {
      const targetHitRatio = targetBin.count / totalLanded;
      const targetPct = Math.round(targetHitRatio * 100);
      bpTargetProbEl.textContent = `${targetPct}%`;

      let distSum = 0;
      state.bins.forEach((b) => {
        if (b.count > 0) {
          distSum += b.count * Math.abs(b.index - state.backpropTargetBinIndex);
        }
      });
      const avgBinDist = distSum / totalLanded;
      const loss = (avgBinDist / 8) + (1 - targetHitRatio) * 0.45;
      bpLossValEl.textContent = loss.toFixed(2);
    }

    if (totalLanded === 0) {
      winnerWordEl.textContent = "—";
      winnerStatsEl.textContent = "Lanza bolas para ver distribución";
      completedSentenceEl.innerHTML = `<span class="prompt-part">${escapeHtml(promptText || "…")}</span> <span class="predicted-part">…</span>`;
      btnAppendWinner.disabled = true;
      top3CandidatesEl.innerHTML = "";
      return;
    }

    const sorted = [...state.bins].sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return b.weightedScore - a.weightedScore;
    });

    const winner = sorted[0];
    const pct = Math.round((winner.count / totalLanded) * 100);

    winnerWordEl.textContent = `“${winner.word}”`;
    winnerStatsEl.textContent = `${winner.count}/${totalLanded} bolas (${pct}% · Σ ${winner.weightedScore.toFixed(1)})`;

    completedSentenceEl.innerHTML = `<span class="prompt-part">${escapeHtml(promptText)}</span> <span class="predicted-part">${escapeHtml(winner.word)}</span>`;
    btnAppendWinner.disabled = false;

    const top3 = sorted.slice(0, 3).filter((b) => b.count > 0);
    top3CandidatesEl.innerHTML = top3
      .map((item, idx) => {
        const itemPct = Math.round((item.count / totalLanded) * 100);
        const isTarget = item.index === state.backpropTargetBinIndex;
        return `
          <div class="top3-row">
            <strong>#${idx + 1} ${escapeHtml(item.word)} ${isTarget ? "🎯" : ""}</strong>
            <div class="top3-bar-wrap">
              <div class="top3-bar-fill" style="width: ${itemPct}%"></div>
            </div>
            <span>${item.count} (${itemPct}%)</span>
          </div>
        `;
      })
      .join("");
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function updateStatusUI() {
    const activeCount = state.balls.length + state.spawnQueue.length;
    metricActiveBalls.textContent = String(activeCount);

    if (activeCount > 0) {
      simStatusDot.classList.add("busy");
      simStatusText.textContent = `Rodando por los carriles y cayendo según calibre (${activeCount} bolas en curso)…`;
    } else {
      simStatusDot.classList.remove("busy");
      simStatusText.textContent = "Listo — Haz clic en una palabra inferior para fijarla como objetivo 🎯 o gira cualquier knob";
    }
  }

  /**
   * Paso de simulación quasi-física:
   * Fase 1: La bola rueda por la vía superior de dos carriles divergentes hasta que la abertura supera su diámetro.
   * Fase 2: La bola se cuela entre los dos carriles y cae a través de las 8 filas de knobs hacia las 24 cubetas.
   */
  function stepPhysics(dt) {
    if (state.spawnQueue.length > 0) {
      state.spawnTimer -= dt;
      if (state.spawnTimer <= 0) {
        const nextItem = state.spawnQueue.shift();
        spawnBallFromQueue(nextItem);
        state.spawnTimer = 0.065;
        updateStatusUI();
      }
    }

    for (const knob of state.knobs) {
      knob.displayWeight += (knob.weight - knob.displayWeight) * Math.min(1, dt * 14);
      if (knob.hitEnergy > 0) {
        knob.hitEnergy = Math.max(0, knob.hitEnergy - dt * 2.8);
      }
      if (knob.bpDelay > 0) {
        knob.bpDelay = Math.max(0, knob.bpDelay - dt);
      } else if (knob.bpGlow > 0) {
        knob.bpGlow = Math.max(0, knob.bpGlow - dt * 1.15);
      }
    }

    if (state.bpPulses.length > 0) {
      state.bpPulses = state.bpPulses.filter((p) => {
        p.progress += dt * p.speed;
        return p.progress <= 1.05;
      });
    }

    if (state.railDropRings.length > 0) {
      state.railDropRings = state.railDropRings.filter((ring) => {
        ring.life -= dt * 2.4;
        return ring.life > 0;
      });
    }

    for (const bin of state.bins) {
      if (bin.flash > 0) {
        bin.flash = Math.max(0, bin.flash - dt * 2.2);
      }
    }

    if (state.balls.length === 0 && state.spawnQueue.length === 0) {
      if (state.autoTraining && state.landedHistory.length > 0 && !state.autoTrainPendingStep) {
        state.autoTrainPendingStep = true;
        setTimeout(() => {
          state.autoTrainPendingStep = false;
          if (!state.autoTraining) return;

          const total = state.bins.reduce((a, b) => a + b.count, 0);
          const targetCount = state.bins[state.backpropTargetBinIndex]?.count || 0;
          const sorted = [...state.bins].sort((a, b) => b.count - a.count);
          const isTargetWinner =
            sorted[0]?.index === state.backpropTargetBinIndex &&
            total > 0 &&
            targetCount / total >= 0.45;

          if (isTargetWinner) {
            stopAutoTrain(
              `✅ ¡Convergencia alcanzada en ${state.bpEpochs} épocas! La palabra objetivo “${VOCABULARY[state.backpropTargetBinIndex].word}” ya gana con el ${Math.round((targetCount / total) * 100)}% de las bolas.`
            );
          } else {
            performBackpropStep(true);
          }
        }, 420);
      }
      return;
    }

    if (state.balls.length === 0) return;

    const subSteps = 3;
    const subDt = dt / subSteps;
    const gravity = 490;
    const { left, right, binsTop } = state.boardBounds;

    const remainingBalls = [];

    for (const ball of state.balls) {
      ball.age += dt;

      if (ball.age > 0.02) {
        ball.trail.push({ x: ball.x, y: ball.y });
        if (ball.trail.length > 8) ball.trail.shift();
      }

      // Fase 1: Rodando por los dos carriles oblicuos superiores hasta llegar al ancho de su calibre
      if (ball.onRails) {
        ball.x += ball.vx * dt;
        ball.y = getRailCenterYAtX(ball.x);
        ball.angle += ball.vx * 0.06 * dt;

        if (ball.x >= ball.dropX) {
          // ¡La abertura entre los dos carriles ya es mayor que el diámetro de esta bola! Cae hacia los knobs
          ball.x = ball.dropX;
          ball.y = getRailCenterYAtX(ball.dropX) + 4;
          ball.onRails = false;
          ball.vx = 18 + (Math.random() - 0.5) * (20 + state.temperature * 28);
          ball.vy = 52 + Math.random() * 18;

          state.railDropRings.push({
            x: ball.x,
            y: ball.y - 3,
            radius: ball.radius,
            color: ball.spec.colorMid,
            life: 1.0
          });
        }

        remainingBalls.push(ball);
        continue;
      }

      // Fase 2: Caída quasi-física entre las 8 filas de knobs
      const speed = Math.hypot(ball.vx, ball.vy);
      if (speed < 42 || Math.abs(ball.vy) < 18) {
        ball.stuckTimer += dt;
        if (ball.stuckTimer > 0.12) {
          ball.vy += 85;
          ball.vx += (Math.random() < 0.5 ? -1 : 1) * (35 + Math.random() * 25);
          ball.stuckTimer = 0;
        }
      } else {
        ball.stuckTimer = Math.max(0, ball.stuckTimer - dt * 2);
      }

      let landedInBin = false;

      for (let s = 0; s < subSteps; s++) {
        const airDrag = 0.995 + (ball.tier / 5) * 0.0035;
        ball.vy += gravity * subDt;
        ball.vx *= airDrag;
        ball.vy *= airDrag;

        ball.x += ball.vx * subDt;
        ball.y += ball.vy * subDt;
        ball.angle += ball.vAngle * subDt;

        // Límites laterales del tablero
        if (ball.x - ball.radius < left) {
          ball.x = left + ball.radius;
          ball.vx = Math.abs(ball.vx) * 0.6;
        } else if (ball.x + ball.radius > right) {
          ball.x = right - ball.radius;
          ball.vx = -Math.abs(ball.vx) * 0.6;
        }

        // Colisión quasi-física con los Knobs
        for (const knob of state.knobs) {
          const dy = ball.y - knob.y;
          if (Math.abs(dy) > ball.radius + knob.radius + 6) continue;
          const dx = ball.x - knob.x;
          if (Math.abs(dx) > ball.radius + knob.radius + 6) continue;

          const distSq = dx * dx + dy * dy;
          const minDist = ball.radius + knob.radius;

          if (distSq < minDist * minDist && distSq > 0.0001) {
            const dist = Math.sqrt(distSq);
            const nx = dx / dist;
            const ny = dy / dist;

            const overlap = minDist - dist + 0.4;
            ball.x += nx * overlap;
            ball.y += ny * overlap;

            const vn = ball.vx * nx + ball.vy * ny;
            if (vn < 0) {
              const restitution = ball.restitution;
              ball.vx -= (1 + restitution) * vn * nx;
              ball.vy -= (1 + restitution) * vn * ny;
            }

            const w = knob.weight;
            const massStability = 0.82 + (ball.tier / 5) * 0.42;
            const thermalScatter =
              (Math.random() - 0.5) * 2 * state.temperature * (135 / Math.sqrt(ball.mass));
            const knobDeflectionVx = w * 185 * state.knobStrength * massStability;

            ball.vx = ball.vx * 0.3 + knobDeflectionVx + thermalScatter;

            if (Math.abs(ball.vx) < 22) {
              ball.vx += (nx >= 0 ? 1 : -1) * (25 + Math.random() * 15);
            }

            if (ball.vy < 48 && ny < 0) {
              const maxUpBounce = -56 + ball.tier * 6;
              ball.vy = Math.max(maxUpBounce, ball.vy * 0.5);
            }

            ball.vAngle = ball.vx * 0.04;
            ball.touchedKnobs.add(knob.id);
            knob.hitEnergy = 1.0;
          }
        }

        // Llegada a las 24 cubetas inferiores
        if (ball.y + ball.radius >= binsTop + 10) {
          const usableW = right - left;
          const relX = Math.max(0, Math.min(usableW - 0.01, ball.x - left));
          const binIdx = Math.floor((relX / usableW) * VOCABULARY.length);
          const targetBin = state.bins[binIdx];

          if (targetBin) {
            targetBin.count += 1;
            targetBin.weightedScore += ball.decimal;
            targetBin.flash = 1.0;

            state.landedHistory.push({
              binIndex: binIdx,
              x: targetBin.xCenter,
              mass: ball.mass,
              decimal: ball.decimal,
              touchedKnobs: Array.from(ball.touchedKnobs)
            });

            if (targetBin.settledBalls.length < 20) {
              targetBin.settledBalls.push({
                tier: ball.tier,
                decimal: ball.decimal,
                spec: ball.spec,
                word: ball.word,
                radius: Math.max(3.8, Math.min(7.2, ball.radius * 0.82)),
                jitterX: (Math.random() - 0.5) * 5
              });
            }
          }

          landedInBin = true;
          break;
        }
      }

      if (!landedInBin) {
        remainingBalls.push(ball);
      } else {
        updatePredictionUI();
      }
    }

    state.balls = remainingBalls;
    updateStatusUI();
  }

  /**
   * Dibuja todo el tablero Galton-LLM en el Canvas
   */
  function drawBoard() {
    const W = state.width;
    const H = state.height;
    ctx.clearRect(0, 0, W, H);

    const { left, knobsTop, knobsBottom, binsTop, binsBottom } = state.boardBounds;

    drawLayerGuides(left, knobsTop, knobsBottom);
    drawDivergingRails();
    drawBackpropTargetGuide(knobsTop, binsTop);
    drawVocabularyBins(binsTop, binsBottom);
    drawBackpropPulses();

    for (const knob of state.knobs) {
      drawSingleKnob(knob);
    }

    for (const ball of state.balls) {
      drawFallingBall(ball);
    }
  }

  function drawLayerGuides(left, knobsTop, knobsBottom) {
    ctx.save();
    ctx.font = "600 10px system-ui, sans-serif";
    ctx.fillStyle = "rgba(148, 163, 184, 0.42)";
    ctx.textAlign = "left";

    const labels = [
      { y: knobsTop, text: "FILA 1 · ENTRADA DISTRIBUIDA POR CALIBRE DECIMAL" },
      { y: (knobsTop + knobsBottom) * 0.5, text: "FILAS 2–7 · MATRIZ DE PESOS (92 KNOBS)" },
      { y: knobsBottom, text: "FILA 8 · PROYECCIÓN A LAS 24 PALABRAS" }
    ];

    labels.forEach((lbl) => {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
      ctx.setLineDash([4, 6]);
      ctx.moveTo(left, lbl.y);
      ctx.lineTo(state.boardBounds.right, lbl.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(lbl.text, left + 4, lbl.y - 6);
    });
    ctx.restore();
  }

  /**
   * Dibuja la vía superior con dos carriles ligeramente oblicuos que se van separando de izquierda a derecha,
   * junto con las marcas de calibre donde caen las bolas de los 6 tamaños (de Roja a Violeta).
   */
  function drawDivergingRails() {
    const { railStartX, railEndX, railStartY, railEndY } = state.boardBounds;
    const startHalfGap = getRailHalfGapAtX(railStartX);
    const endHalfGap = getRailHalfGapAtX(railEndX);

    ctx.save();

    // Sombra/hueco interior entre los dos carriles divergentes (forma de cuña que se abre hacia la derecha)
    const wedgeGrad = ctx.createLinearGradient(railStartX, 0, railEndX, 0);
    wedgeGrad.addColorStop(0, "rgba(239, 68, 68, 0.14)");
    wedgeGrad.addColorStop(0.25, "rgba(249, 115, 22, 0.14)");
    wedgeGrad.addColorStop(0.45, "rgba(234, 179, 8, 0.14)");
    wedgeGrad.addColorStop(0.65, "rgba(34, 197, 94, 0.14)");
    wedgeGrad.addColorStop(0.82, "rgba(59, 130, 246, 0.14)");
    wedgeGrad.addColorStop(1, "rgba(168, 85, 247, 0.18)");

    ctx.beginPath();
    ctx.moveTo(railStartX, railStartY - startHalfGap);
    ctx.lineTo(railEndX, railEndY - endHalfGap);
    ctx.lineTo(railEndX, railEndY + endHalfGap);
    ctx.lineTo(railStartX, railStartY + startHalfGap);
    ctx.closePath();
    ctx.fillStyle = wedgeGrad;
    ctx.fill();

    // Marcas de calibre de los 6 tamaños a lo largo de la vía
    BALL_TIERS.forEach((tierSpec, idx) => {
      const tZone = 0.09 + (idx / 5) * 0.82;
      const gx = railStartX + tZone * (railEndX - railStartX);
      const gy = getRailCenterYAtX(gx);
      const hGap = getRailHalfGapAtX(gx);

      // Línea vertical punteada hacia la primera fila de knobs
      ctx.beginPath();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = tierSpec.glow;
      ctx.lineWidth = 1.2;
      ctx.moveTo(gx, gy + hGap + 2);
      ctx.lineTo(gx, state.boardBounds.knobsTop - 14);
      ctx.stroke();
      ctx.setLineDash([]);

      // Etiqueta del calibre encima de la vía
      ctx.font = "700 9.5px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = tierSpec.colorCore;
      ctx.fillText(`↓ ${tierSpec.sampleDecimal} (${tierSpec.name})`, gx, gy - hGap - 6);
    });

    // Dibujar los dos carriles metálicos divergentes (Carril Superior y Carril Inferior)
    ctx.lineWidth = 3.0;
    ctx.lineCap = "round";

    // Carril 1 (superior)
    ctx.beginPath();
    ctx.strokeStyle = "#94a3b8";
    ctx.moveTo(railStartX - 12, railStartY - startHalfGap);
    ctx.lineTo(railEndX + 8, railEndY - endHalfGap);
    ctx.stroke();

    // Carril 2 (inferior, divergiendo más a medida que baja hacia la derecha)
    ctx.beginPath();
    ctx.strokeStyle = "#cbd5e1";
    ctx.moveTo(railStartX - 12, railStartY + startHalfGap);
    ctx.lineTo(railEndX + 8, railEndY + endHalfGap);
    ctx.stroke();

    // Boquilla de entrada a la izquierda de los carriles
    ctx.fillStyle = "#1e293b";
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(railStartX - 28, railStartY - 12, 22, 24, 5);
    ctx.fill();
    ctx.stroke();

    ctx.font = "700 9px system-ui, sans-serif";
    ctx.fillStyle = "#38bdf8";
    ctx.textAlign = "left";
    ctx.fillText("VÍA DE 2 CARRILES DIVERGENTES →", railStartX - 28, railStartY - 16);

    // Anillos de destello donde las bolas acaban de colarse entre los dos carriles
    for (const ring of state.railDropRings) {
      ctx.beginPath();
      ctx.arc(ring.x, ring.y, ring.radius + (1 - ring.life) * 10, 0, Math.PI * 2);
      ctx.strokeStyle = ring.color;
      ctx.globalAlpha = ring.life * 0.75;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.restore();
  }

  function drawBackpropTargetGuide(knobsTop, binsTop) {
    const targetBin = state.bins[state.backpropTargetBinIndex];
    if (!targetBin) return;

    const { railStartX, railEndX } = state.boardBounds;
    ctx.save();

    // Suave abanico convergente desde la vía de carriles hacia la cubeta objetivo y*
    const grad = ctx.createLinearGradient(0, knobsTop, 0, binsTop);
    grad.addColorStop(0, "rgba(167, 139, 250, 0.015)");
    grad.addColorStop(1, "rgba(167, 139, 250, 0.08)");

    ctx.beginPath();
    ctx.moveTo(railStartX, knobsTop);
    ctx.lineTo(railEndX, knobsTop);
    ctx.lineTo(targetBin.xRight + 8, binsTop);
    ctx.lineTo(targetBin.xLeft - 8, binsTop);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.restore();
  }

  function drawBackpropPulses() {
    if (state.bpPulses.length === 0) return;
    ctx.save();

    for (const p of state.bpPulses) {
      if (p.progress < 0 || p.progress > 1) continue;
      const t = p.progress;
      const x = p.startX + (p.endX - p.startX) * t;
      const y = p.startY + (p.endY - p.startY) * t;

      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = p.delta >= 0 ? "#f472b6" : "#c084fc";
      ctx.shadowColor = "#e879f9";
      ctx.shadowBlur = 10;
      ctx.fill();
    }

    ctx.restore();
  }

  function drawSingleKnob(knob) {
    ctx.save();
    ctx.translate(knob.x, knob.y);

    const r = knob.radius;
    const w = knob.displayWeight;
    const maxTiltRad = 0.98;
    const tiltAngle = w * maxTiltRad;

    const isHovered = state.hoveredKnob === knob || state.draggingKnob === knob;
    const hasBpGlow = knob.bpDelay <= 0 && knob.bpGlow > 0.02;

    let rimColor = "#64748b";
    if (w > 0.08) {
      rimColor = "#f59e0b"; // desvío derecha
    } else if (w < -0.08) {
      rimColor = "#38bdf8"; // desvío izquierda
    }

    if (hasBpGlow) {
      ctx.beginPath();
      ctx.arc(0, 0, r + 5 + knob.bpGlow * 5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(192, 132, 252, ${knob.bpGlow * 0.42})`;
      ctx.fill();
    }

    if (knob.hitEnergy > 0.02 || isHovered) {
      const glowRadius = r + 4 + knob.hitEnergy * 5;
      ctx.beginPath();
      ctx.arc(0, 0, glowRadius, 0, Math.PI * 2);
      ctx.fillStyle = isHovered
        ? "rgba(56, 189, 248, 0.32)"
        : w >= 0
          ? `rgba(245, 158, 11, ${knob.hitEnergy * 0.35})`
          : `rgba(56, 189, 248, ${knob.hitEnergy * 0.35})`;
      ctx.fill();
    }

    ctx.save();
    ctx.rotate(tiltAngle);

    ctx.beginPath();
    ctx.roundRect(-r * 1.12, -r - 3.0, r * 2.24, 4.0, 2.0);
    ctx.fillStyle = hasBpGlow ? "#e879f9" : rimColor;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    const dialGrad = ctx.createRadialGradient(-r * 0.3, -r * 0.3, r * 0.1, 0, 0, r);
    dialGrad.addColorStop(0, "#334155");
    dialGrad.addColorStop(1, "#0f172a");
    ctx.fillStyle = dialGrad;
    ctx.fill();

    ctx.lineWidth = isHovered || hasBpGlow ? 2.2 : 1.5;
    ctx.strokeStyle = isHovered ? "#f8fafc" : hasBpGlow ? "#e879f9" : rimColor;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, 1);
    ctx.lineTo(0, -r + 2);
    ctx.lineWidth = 2.1;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#f8fafc";
    ctx.stroke();

    ctx.restore();

    ctx.beginPath();
    ctx.arc(0, 0, 1.8, 0, Math.PI * 2);
    ctx.fillStyle = "#94a3b8";
    ctx.fill();

    if ((hasBpGlow && Math.abs(knob.lastDelta) >= 0.01) || state.draggingKnob === knob) {
      const val = state.draggingKnob === knob ? knob.weight : knob.lastDelta;
      const sign = val > 0 ? "+" : "";
      ctx.font = "700 9px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = state.draggingKnob === knob ? "#38bdf8" : "#f5d0fe";
      ctx.fillText(`${sign}${val.toFixed(2)}`, 0, r + 10);
    }

    ctx.restore();
  }

  function drawFallingBall(ball) {
    ctx.save();

    if (ball.trail.length > 1) {
      ctx.beginPath();
      for (let i = 0; i < ball.trail.length; i++) {
        const pt = ball.trail[i];
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = ball.spec.glow;
      ctx.lineWidth = Math.max(1.5, ball.radius * 0.5);
      ctx.lineCap = "round";
      ctx.stroke();
    }

    const r = ball.radius;
    const grad = ctx.createRadialGradient(
      ball.x - r * 0.32,
      ball.y - r * 0.32,
      r * 0.12,
      ball.x,
      ball.y,
      r
    );
    grad.addColorStop(0, ball.spec.colorCore);
    grad.addColorStop(0.65, ball.spec.colorMid);
    grad.addColorStop(1, ball.spec.colorEdge);

    ctx.beginPath();
    ctx.arc(ball.x, ball.y, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.lineWidth = 1.1;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
    ctx.stroke();

    ctx.font = "700 9.5px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "#f8fafc";
    ctx.shadowColor = "rgba(0, 0, 0, 0.9)";
    ctx.shadowBlur = 4;
    ctx.fillText(`${ball.word} ${ball.decimal.toFixed(2)}`, ball.x, ball.y - r - 3.5);

    ctx.restore();
  }

  function drawVocabularyBins(binsTop, binsBottom) {
    ctx.save();

    const totalLanded = state.bins.reduce((acc, b) => acc + b.count, 0);
    let maxCount = 0;
    let winnerIndex = -1;

    state.bins.forEach((b) => {
      if (b.count > maxCount) {
        maxCount = b.count;
        winnerIndex = b.index;
      }
    });

    const binHeight = binsBottom - binsTop;

    state.bins.forEach((bin, idx) => {
      const binW = bin.xRight - bin.xLeft;
      const isWinner = winnerIndex === idx && maxCount > 0;
      const isHovered = state.hoveredBinIndex === idx;
      const isBpTarget = state.backpropTargetBinIndex === idx;

      ctx.fillStyle = isWinner
        ? "rgba(16, 185, 129, 0.18)"
        : isBpTarget
          ? "rgba(167, 139, 250, 0.16)"
          : isHovered
            ? "rgba(56, 189, 248, 0.14)"
            : "rgba(15, 23, 42, 0.65)";
      ctx.fillRect(bin.xLeft + 1, binsTop, binW - 2, binHeight);

      if (bin.count > 0 && maxCount > 0) {
        const fillRatio = bin.count / Math.max(4, maxCount);
        const barH = Math.min(binHeight - 6, fillRatio * (binHeight - 16));
        const barGrad = ctx.createLinearGradient(0, binsBottom - barH, 0, binsBottom);
        if (isWinner) {
          barGrad.addColorStop(0, "rgba(16, 185, 129, 0.55)");
          barGrad.addColorStop(1, "rgba(5, 150, 105, 0.25)");
        } else if (isBpTarget) {
          barGrad.addColorStop(0, "rgba(192, 132, 252, 0.55)");
          barGrad.addColorStop(1, "rgba(124, 58, 237, 0.25)");
        } else {
          barGrad.addColorStop(0, "rgba(56, 189, 248, 0.4)");
          barGrad.addColorStop(1, "rgba(37, 99, 235, 0.15)");
        }
        ctx.fillStyle = barGrad;
        ctx.fillRect(bin.xLeft + 2, binsBottom - barH, binW - 4, barH);
      }

      let currentY = binsBottom - 6;
      bin.settledBalls.forEach((sb, bIdx) => {
        const colOffset = (bIdx % 2 === 0 ? -1 : 1) * Math.min(binW * 0.18, 4.0);
        const bx = bin.xCenter + colOffset + sb.jitterX * 0.25;
        const by = currentY - sb.radius * 0.6;
        currentY -= sb.radius * 1.25;

        if (by > binsTop + 6) {
          ctx.beginPath();
          ctx.arc(bx, by, sb.radius, 0, Math.PI * 2);
          ctx.fillStyle = sb.spec.colorMid;
          ctx.fill();
          ctx.lineWidth = 0.8;
          ctx.strokeStyle = sb.spec.colorCore;
          ctx.stroke();
        }
      });

      ctx.beginPath();
      ctx.moveTo(bin.xLeft, binsTop);
      ctx.lineTo(bin.xLeft, binsBottom);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = isWinner ? "#10b981" : isBpTarget ? "#a78bfa" : "#334155";
      ctx.stroke();

      if (isBpTarget) {
        ctx.font = "700 10px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = "#e879f9";
        ctx.fillText("🎯 y*", bin.xCenter, binsTop - (bin.count > 0 ? 17 : 5));
      }

      if (bin.count > 0) {
        ctx.font = "700 11px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = isWinner ? "#34d399" : isBpTarget ? "#e879f9" : "#bae6fd";
        ctx.fillText(String(bin.count), bin.xCenter, binsTop - 5);
      }

      ctx.save();
      ctx.translate(bin.xCenter, binsBottom + 12);
      ctx.rotate(Math.PI / 4.2);
      ctx.font = isWinner || isBpTarget
        ? "800 12px system-ui, sans-serif"
        : "600 11px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = isWinner
        ? "#34d399"
        : isBpTarget
          ? "#e879f9"
          : isHovered
            ? "#38bdf8"
            : "#94a3b8";
      ctx.fillText(isBpTarget ? `${bin.word} 🎯` : bin.word, 0, 4);
      ctx.restore();
    });

    const lastBin = state.bins[state.bins.length - 1];
    if (lastBin) {
      ctx.beginPath();
      ctx.moveTo(lastBin.xRight, binsTop);
      ctx.lineTo(lastBin.xRight, binsBottom);
      ctx.moveTo(state.boardBounds.left, binsBottom);
      ctx.lineTo(lastBin.xRight, binsBottom);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#475569";
      ctx.stroke();
    }

    const bpBin = state.bins[state.backpropTargetBinIndex];
    if (bpBin) {
      ctx.save();
      ctx.strokeStyle = "#c084fc";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(bpBin.xLeft + 1, binsTop, (bpBin.xRight - bpBin.xLeft) - 2, binHeight);
      ctx.restore();
    }

    if (totalLanded > 0 && winnerIndex >= 0) {
      const wBin = state.bins[winnerIndex];
      ctx.save();
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2.2;
      ctx.strokeRect(wBin.xLeft + 1, binsTop, (wBin.xRight - wBin.xLeft) - 2, binHeight);
      ctx.restore();
    }

    ctx.restore();
  }

  function getCanvasCoords(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((evt.clientX - rect.left) / rect.width) * state.width,
      y: ((evt.clientY - rect.top) / rect.height) * state.height
    };
  }

  function findKnobAt(x, y) {
    for (const knob of state.knobs) {
      const dx = x - knob.x;
      const dy = y - knob.y;
      const hitR = knob.radius + 7;
      if (dx * dx + dy * dy <= hitR * hitR) {
        return knob;
      }
    }
    return null;
  }

  function findBinAt(x, y) {
    if (y < state.boardBounds.binsTop) return -1;
    for (const bin of state.bins) {
      if (x >= bin.xLeft && x <= bin.xRight) {
        return bin.index;
      }
    }
    return -1;
  }

  function updateHoveredKnobUI(knob) {
    if (!knob) {
      hoveredKnobPill.innerHTML = `Knob: <strong>ninguno</strong>`;
      return;
    }
    const sign = knob.weight > 0 ? "+" : "";
    const dir = knob.weight > 0.05 ? "→ Der" : knob.weight < -0.05 ? "← Izq" : "Centro";
    hoveredKnobPill.innerHTML = `Knob F${knob.row + 1}·N${knob.col + 1}: <strong>${sign}${knob.weight.toFixed(2)} (${dir})</strong>`;
  }

  canvas.addEventListener("pointerdown", (evt) => {
    const pos = getCanvasCoords(evt);
    const knob = findKnobAt(pos.x, pos.y);

    if (knob) {
      state.draggingKnob = knob;
      state.dragStartX = pos.x;
      state.dragStartY = pos.y;
      state.dragStartWeight = knob.weight;
      canvas.setPointerCapture(evt.pointerId);
      updateHoveredKnobUI(knob);
      return;
    }

    const binIdx = findBinAt(pos.x, pos.y);
    if (binIdx >= 0) {
      setBackpropTargetBin(binIdx, true);
    }
  });

  canvas.addEventListener("pointermove", (evt) => {
    const pos = getCanvasCoords(evt);

    if (state.draggingKnob) {
      const deltaX = (pos.x - state.dragStartX) / 55;
      const deltaY = -(pos.y - state.dragStartY) / 55;
      const combinedDelta = Math.abs(deltaX) > Math.abs(deltaY) ? deltaX : deltaY;

      const newW = Math.max(-0.98, Math.min(0.98, state.dragStartWeight + combinedDelta));
      const k = state.draggingKnob;
      k.weight = Math.round(newW * 100) / 100;
      k.hitEnergy = 0.6;
      state.mode = "custom";
      Object.values(presetBtns).forEach((btn) => btn && btn.classList.remove("active"));
      updateHoveredKnobUI(k);
      bpFeedbackLog.innerHTML = `Ajuste manual en <strong>Knob F${k.row + 1}·N${k.col + 1}</strong> → peso <strong>${k.weight > 0 ? "+" : ""}${k.weight.toFixed(2)}</strong>. Lanza la frase para ver cómo cambia la caída hacia <strong>“${escapeHtml(VOCABULARY[state.backpropTargetBinIndex].word)}”</strong>.`;
      return;
    }

    const knob = findKnobAt(pos.x, pos.y);
    state.hoveredKnob = knob;
    state.hoveredBinIndex = knob ? -1 : findBinAt(pos.x, pos.y);

    canvas.style.cursor = knob ? "ew-resize" : state.hoveredBinIndex >= 0 ? "pointer" : "default";
    updateHoveredKnobUI(knob);
  });

  window.addEventListener("pointerup", () => {
    state.draggingKnob = null;
  });

  canvas.addEventListener(
    "wheel",
    (evt) => {
      const pos = getCanvasCoords(evt);
      const knob = findKnobAt(pos.x, pos.y);
      if (!knob) return;

      evt.preventDefault();
      const step = evt.deltaY < 0 ? 0.08 : -0.08;
      knob.weight = Math.max(-0.98, Math.min(0.98, Math.round((knob.weight + step) * 100) / 100));
      knob.hitEnergy = 0.7;
      state.mode = "custom";
      Object.values(presetBtns).forEach((btn) => btn && btn.classList.remove("active"));
      updateHoveredKnobUI(knob);
      bpFeedbackLog.innerHTML = `Ajuste manual con rueda en <strong>Knob F${knob.row + 1}·N${knob.col + 1}</strong> → peso <strong>${knob.weight > 0 ? "+" : ""}${knob.weight.toFixed(2)}</strong>.`;
    },
    { passive: false }
  );

  presetPromptSelect.addEventListener("change", () => {
    stopAutoTrain();
    promptInput.value = presetPromptSelect.value;
    clearBallsAndBins();
    renderTokenChips();
    if (state.targetBinsInfo.length > 0) {
      setBackpropTargetBin(state.targetBinsInfo[0].binIndex, false);
    }
    triggerBurstDrop();
  });

  promptInput.addEventListener("input", () => {
    renderTokenChips();
  });

  promptInput.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter") {
      evt.preventDefault();
      triggerBurstDrop();
    }
  });

  function triggerBurstDrop() {
    const tokens = parsePromptTokens(promptInput.value);
    if (!tokens.length) return;
    if (state.mode === "trained") {
      applyKnobPreset("trained");
    }
    enqueueWordVectors(tokens);
  }

  btnDropBurst.addEventListener("click", () => {
    triggerBurstDrop();
  });

  btnDropSingle.addEventListener("click", () => {
    const tokens = parsePromptTokens(promptInput.value, 3);
    if (!tokens.length) return;
    if (state.mode === "trained") {
      applyKnobPreset("trained");
    }
    enqueueWordVectors(tokens);
  });

  btnClearBalls.addEventListener("click", () => {
    stopAutoTrain();
    clearBallsAndBins();
  });

  btnAppendWinner.addEventListener("click", () => {
    stopAutoTrain();
    const sorted = [...state.bins].sort((a, b) => b.count - a.count);
    const winner = sorted[0];
    if (!winner || winner.count === 0) return;

    promptInput.value = `${promptInput.value.trim()} ${winner.word}`.trim();
    clearBallsAndBins();
    renderTokenChips();
    if (state.targetBinsInfo.length > 0) {
      setBackpropTargetBin(state.targetBinsInfo[0].binIndex, false);
    }
    triggerBurstDrop();
  });

  Object.entries(presetBtns).forEach(([modeKey, btn]) => {
    btn.addEventListener("click", () => {
      stopAutoTrain();
      applyKnobPreset(modeKey);
      clearBallsAndBins();
      triggerBurstDrop();
    });
  });

  ballsPerWordSlider.addEventListener("input", () => {
    state.ballsPerWord = Number(ballsPerWordSlider.value);
    ballsPerWordVal.textContent = String(state.ballsPerWord);
    renderTokenChips();
  });

  temperatureSlider.addEventListener("input", () => {
    state.temperature = Number(temperatureSlider.value);
    temperatureVal.textContent = state.temperature.toFixed(2);
  });

  knobStrengthSlider.addEventListener("input", () => {
    state.knobStrength = Number(knobStrengthSlider.value);
    if (knobStrengthVal) {
      knobStrengthVal.textContent = `${state.knobStrength.toFixed(1)}×`;
    }
  });

  targetWordSelect.addEventListener("change", () => {
    stopAutoTrain();
    setBackpropTargetBin(Number(targetWordSelect.value), true);
  });

  learningRateSlider.addEventListener("input", () => {
    state.learningRate = Number(learningRateSlider.value);
    learningRateVal.textContent = state.learningRate.toFixed(2);
  });

  btnBackpropStep.addEventListener("click", () => {
    stopAutoTrain();
    performBackpropStep(true);
  });

  btnBackpropNudgeOnly.addEventListener("click", () => {
    stopAutoTrain();
    performBackpropStep(false);
  });

  function stopAutoTrain(customMessage) {
    if (!state.autoTraining && !customMessage) return;
    state.autoTraining = false;
    state.autoTrainPendingStep = false;
    btnAutoTrain.classList.remove("active-loop");
    btnAutoTrain.textContent = "🔁 Auto-Entrenar";
    if (customMessage) {
      bpFeedbackLog.innerHTML = customMessage;
    }
  }

  btnAutoTrain.addEventListener("click", () => {
    if (state.autoTraining) {
      stopAutoTrain("⏹ Auto-entrenamiento pausado por el usuario.");
      return;
    }
    state.autoTraining = true;
    btnAutoTrain.classList.add("active-loop");
    btnAutoTrain.textContent = "⏹ Detener Auto-Entreno";
    performBackpropStep(true);
  });

  const canvasWrapper = document.getElementById("canvas-wrapper");
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const rect = entry.contentRect;
      if (rect.width < 100 || rect.height < 100) continue;

      const dpr = window.devicePixelRatio || 1;
      state.dpr = dpr;
      state.width = Math.round(rect.width);
      state.height = Math.round(rect.height);

      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      layoutBoard(state.knobs.length > 0);
    }
  });
  resizeObserver.observe(canvasWrapper);

  let lastTime = performance.now();
  function animate(now) {
    const dt = Math.min(0.035, (now - lastTime) / 1000);
    lastTime = now;

    stepPhysics(dt);
    drawBoard();

    requestAnimationFrame(animate);
  }

  initTargetWordSelect();
  renderTokenChips();
  layoutBoard(false);
  requestAnimationFrame((t) => {
    lastTime = t;
    triggerBurstDrop();
    requestAnimationFrame(animate);
  });
})();
