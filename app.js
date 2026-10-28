/**
 * Galton-LLM — Quasi-Physical Galton Machine Simulation with Knobs & Backpropagation
 * At the top, two diverging oblique rails guide the balls:
 * smaller balls (Red) drop earlier through the gap between rails while larger balls (Violet) drop further right,
 * giving all knobs across all rows real influence over the output distribution.
 */

(() => {
  // Default bottom output vocabulary (24 basic English words)
  const DEFAULT_VOCABULARY = [
    { word: "the", role: "conn" },
    { word: "a", role: "conn" },
    { word: "is", role: "verb" },
    { word: "in", role: "conn" },
    { word: "on", role: "conn" },
    { word: "to", role: "conn" },
    { word: "has", role: "verb" },
    { word: "cat", role: "noun" },
    { word: "dog", role: "noun" },
    { word: "fish", role: "noun" },
    { word: "boy", role: "noun" },
    { word: "girl", role: "noun" },
    { word: "man", role: "noun" },
    { word: "sun", role: "noun" },
    { word: "sat", role: "verb" },
    { word: "ran", role: "verb" },
    { word: "big", role: "noun" },
    { word: "red", role: "noun" },
    { word: "blue", role: "noun" },
    { word: "hot", role: "noun" },
    { word: "mat", role: "noun" },
    { word: "hat", role: "noun" },
    { word: "bag", role: "noun" },
    { word: "bed", role: "noun" }
  ];

  let VOCABULARY = DEFAULT_VOCABULARY.map((item) => ({ ...item }));

  // 6 ball size/color tiers (0 = Red/Small -> 5 = Violet/Large)
  const BALL_TIERS = [
    {
      tier: 0,
      name: "Red",
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
      name: "Orange",
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
      name: "Yellow",
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
      name: "Green",
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
      name: "Blue",
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
      name: "Violet",
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
    "the", "a", "an", "in", "on", "with", "through", "by", "for", "from",
    "to", "of", "at", "into", "over", "after", "and", "or", "that", "its",
    "my", "your"
  ]);

  // DOM References
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
    random: document.getElementById("preset-random"),
    neutral: document.getElementById("preset-neutral")
  };

  const ballsPerWordSlider = document.getElementById("balls-per-word-slider");
  const ballsPerWordVal = document.getElementById("balls-per-word-val");
  const temperatureSlider = document.getElementById("temperature-slider");
  const temperatureVal = document.getElementById("temperature-val");
  const knobStrengthSlider = document.getElementById("knob-strength-slider");
  const knobStrengthVal = document.getElementById("knob-strength-val");
  const knobResistanceSlider = document.getElementById("knob-resistance-slider");
  const knobResistanceVal = document.getElementById("knob-resistance-val");
  const simSpeedSlider = document.getElementById("sim-speed-slider");
  const simSpeedVal = document.getElementById("sim-speed-val");

  // Backpropagation Controls
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
  const bpMemoryBadge = document.getElementById("bp-memory-badge");
  const bpMemoryStrip = document.getElementById("bp-memory-strip");

  const winnerWordEl = document.getElementById("winner-word");
  const winnerStatsEl = document.getElementById("winner-stats");
  const completedSentenceEl = document.getElementById("completed-sentence");
  const top3CandidatesEl = document.getElementById("top3-candidates");

  const simStatusDot = document.getElementById("sim-status-dot");
  const simStatusText = document.getElementById("sim-status-text");
  const metricActiveBalls = document.getElementById("metric-active-balls");
  const metricLandedBalls = document.getElementById("metric-landed-balls");
  const hoveredKnobPill = document.getElementById("hovered-knob-pill");

  // Vocabulary Dialog Controls
  const btnOpenVocab = document.getElementById("btn-open-vocab");
  const vocabCountBadge = document.getElementById("vocab-count-badge");
  const vocabDialog = document.getElementById("vocab-dialog");
  const vocabForm = document.getElementById("vocab-form");
  const vocabTextarea = document.getElementById("vocab-textarea");
  const vocabDialogCount = document.getElementById("vocab-dialog-count");
  const vocabDialogChips = document.getElementById("vocab-dialog-chips");
  const btnResetVocab = document.getElementById("btn-reset-vocab");
  const btnCancelVocab = document.getElementById("btn-cancel-vocab");
  const btnCloseVocabX = document.getElementById("btn-close-vocab-x");

  // Training Rules Dialog Controls
  const rulesDialog = document.getElementById("rules-dialog");
  const rulesForm = document.getElementById("rules-form");
  const rulesTextarea = document.getElementById("rules-textarea");
  const rulesDialogCount = document.getElementById("rules-dialog-count");
  const rulesDialogChips = document.getElementById("rules-dialog-chips");
  const btnAddAllExamples = document.getElementById("btn-add-all-examples");
  const btnClearRules = document.getElementById("btn-clear-rules");
  const btnCancelRules = document.getElementById("btn-cancel-rules");
  const btnCloseRulesX = document.getElementById("btn-close-rules-x");

  // Simulation State
  const state = {
    width: 960,
    height: 700,
    dpr: 1,
    customCanvasSize: null, // null = auto-fit window, or { width, height }
    rowCounts: [22, 23, 22, 23, 22, 23, 22, 23],
    customRowCounts: false,
    mode: "neutral", // 'neutral' | 'random' | 'custom'
    ballsPerWord: 6,
    temperature: 0.25,
    knobStrength: 1.0,
    knobResistance: 0.75,
    simSpeed: 1.0,
    learningRate: 0.22,
    backpropTargetBinIndex: 20, // 'mat'
    bpEpochs: 0,
    autoTraining: false,
    autoTrainPendingStep: false,
    autoTrainIndex: 0,
    autoTrainPassSuccessCount: 0,
    trainedPrompts: new Map(), // Trained prompt memory: promptKey -> { promptKey, targetBinIndex, epochs, converged }
    knobs: [],
    balls: [],
    spawnQueue: [],
    spawnTimer: 0,
    landedHistory: [],
    bpPulses: [],
    railDropRings: [],
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
   * Normaliza un texto de prompt para usarlo como clave de contexto y memoria
   */
  function getNormalizedPromptKey(text) {
    return (text || "")
      .trim()
      .toLowerCase()
      .split(/\s+/)
      .map((w) => w.replace(/[.,;:!?¿¡"']/g, "").trim())
      .filter(Boolean)
      .join(" ");
  }

  /**
   * Atención Contextual Causal (Self-Attention simplificado):
   * Calcula una firma determinista que depende de toda la frase y muy especialmente
   * de la última palabra (Query) y la longitud de la secuencia. Así, "el gato duerme en"
   * y "el gato duerme en la" proyectan sus bolas por subcanales distintos del carril.
   */
  function computePromptContextSignature(words) {
    if (!words || words.length === 0) {
      return { hash: 0, attentionCenter: 0.5, subChannelShift: 0 };
    }
    const lastWord = words[words.length - 1];
    let h = 2166136261;
    for (let wIdx = 0; wIdx < words.length; wIdx++) {
      const w = words[wIdx];
      const posWeight = (wIdx + 1) * (wIdx === words.length - 1 ? 37 : 13);
      for (let i = 0; i < w.length; i++) {
        h ^= w.charCodeAt(i) * posWeight;
        h = Math.imul(h, 16777619);
      }
    }
    // Incorporar longitud exacta y última palabra con fuerte peso causal
    h ^= words.length * 0x9e3779b9;
    h = Math.imul(h, 0x85ebca6b);
    for (let i = 0; i < lastWord.length; i++) {
      h ^= (lastWord.charCodeAt(i) + i * 29) * 0x27d4eb2d;
      h = Math.imul(h, 16777619);
    }
    h ^= h >>> 15;
    const absH = Math.abs(h);

    // Centro de atención sobre la vía en [0.16, 0.84] y desfase fino de subcanal
    const attentionCenter = 0.16 + ((absH % 1000) / 1000) * 0.68;
    const subChannelShift = (((absH >>> 8) % 200) / 200 - 0.5) * 0.085;
    return {
      hash: absH,
      attentionCenter,
      subChannelShift
    };
  }

  /**
   * Convierte un número decimal en [0.05, 0.99] en uno de los 6 niveles de tamaño/color de bola (0=Roja a 5=Violeta)
   */
  function decimalToTierSpec(decimalVal) {
    const clamped = Math.max(0.05, Math.min(0.99, decimalVal));
    const idx = Math.max(0, Math.min(5, Math.floor(((clamped - 0.05) / 0.94) * 6)));
    return BALL_TIERS[idx];
  }

  /**
   * Generates the deterministic combination of balls (decimal vector) for a word,
   * modulated by the prompt's Contextual Attention (contextSig) across the full [0.06, 0.98] range
   * so all 6 ball tiers (Red=0 through Violet=5) appear uniformly.
   */
  function getWordDecimalVector(rawWord, dimensionCount, contextSig, wordIdx = 0, totalWords = 1) {
    const clean = rawWord.toLowerCase().replace(/[.,;:!?¿¡"']/g, "").trim() || "x";

    let wordHash = 2166136261;
    for (let i = 0; i < clean.length; i++) {
      wordHash ^= clean.charCodeAt(i);
      wordHash = Math.imul(wordHash, 16777619);
    }

    const ctxHash = contextSig ? contextSig.hash : 0;
    const ctxPhase = ((Math.abs(wordHash ^ Math.imul(ctxHash, 2654435761) ^ Math.imul(wordIdx + 1, 374761393))) % 1000) / 1000;

    const components = [];
    for (let d = 0; d < dimensionCount; d++) {
      let h = Math.imul(
        wordHash ^ Math.imul(ctxHash, 31) ^ Math.imul(wordIdx + 1, 0x85ebca6b) ^ Math.imul(d + 1, 0x9e3779b9),
        0xc2b2ae35
      );
      h ^= h >>> 13;
      const pseudoRand = (Math.abs(h) % 1000) / 1000;

      // Stride 5 is coprime with 6, visiting all 6 size bands (0..5) evenly
      const bandSlot = (d * 5 + Math.abs(wordHash)) % 6;
      // Modular wraparound on [0, 1) preserves uniform coverage across all 6 tiers (including Red & Violet)
      const u = (((bandSlot + 0.5 + (pseudoRand - 0.5) * 1.35) / 6) + ctxPhase + 10) % 1.0;
      const rawDecimal = 0.06 + u * 0.92;

      const decimal = Math.max(0.06, Math.min(0.98, Math.round(rawDecimal * 100) / 100));
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
   * Calcula la coordenada X exacta en la que la separación de los dos carriles supera el tamaño de la bola,
   * incorporando el desplazamiento de subcanal de la Atención Contextual del prompt.
   */
  function computeDropXForBall(decimalVal, tokenIndex, totalTokens, contextSig) {
    const { railStartX, railEndX } = state.boardBounds;
    const trackLen = railEndX - railStartX;

    // El progreso a lo largo de la vía depende del diámetro/valor decimal de la bola y del subcanal contextual
    const normalizedSize = Math.max(0.04, Math.min(0.96, (decimalVal - 0.06) / 0.92));
    const wordOffset = totalTokens > 1 ? ((tokenIndex / (totalTokens - 1)) - 0.5) * 0.038 : 0;
    const ctxShift = contextSig ? contextSig.subChannelShift : 0;
    // Inferencia determinista: sin jitter aleatorio. La aleatoriedad solo se usa al muestrear la palabra ganadora.

    const tDrop = Math.max(0.04, Math.min(0.96, normalizedSize + wordOffset + ctxShift));
    return railStartX + tDrop * trackLen;
  }

  function computeRecommendedRowCounts(width, height) {
    if (width >= 800 && height >= 600) {
      return [22, 23, 22, 23, 22, 23, 22, 23];
    }
    const baseKnobs =
      width >= 800 ? 22 : Math.max(8, Math.min(22, Math.floor((width * 0.88) / 32)));
    const numRows =
      height >= 600 ? 8 : Math.max(4, Math.min(8, Math.floor((height * 0.5) / 38)));
    const counts = [];
    for (let r = 0; r < numRows; r++) {
      counts.push(r % 2 === 1 ? baseKnobs + 1 : baseKnobs);
    }
    return counts;
  }

  function parseRowCountsString(rawText) {
    const parts = (rawText || "")
      .split(/[,;\s]+/)
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => Number.isFinite(n) && n >= 2);
    if (parts.length === 0) return null;
    return parts.slice(0, 16).map((n) => Math.max(2, Math.min(45, n)));
  }

  function initTargetWordSelect() {
    targetWordSelect.innerHTML = "";
    VOCABULARY.forEach((item, index) => {
      const opt = document.createElement("option");
      opt.value = String(index);
      opt.textContent = `🎯 “${item.word}” (Bin #${index + 1})`;
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
      bpFeedbackLog.innerHTML = `Target word set to <strong>“${escapeHtml(targetWord)}”</strong>. Click <em>⚡ Backprop</em> to adjust knobs toward it.${renderTrainedMemoryStripHtml()}`;
    }
  }

  function parsePromptTokens(text, customDimCount) {
    const dim = customDimCount || state.ballsPerWord;
    const rawParts = text
      .trim()
      .split(/\s+/)
      .map((w) => w.replace(/[.,;:!?¿¡"']/g, "").trim().toLowerCase())
      .filter(Boolean);

    const contextSig = computePromptContextSignature(rawParts);

    return rawParts.map((lower, idx) => {
      const vector = getWordDecimalVector(lower, dim, contextSig, idx, rawParts.length);
      return {
        id: `${idx}-${lower}`,
        word: lower,
        vector,
        contextSig
      };
    });
  }

  /**
   * Computes the ideal weight w* in [-0.95, 0.95] so a knob at knob.x
   * steers balls toward targetX
   */
  function computeIdealKnobWeightForTargetX(knob, targetX) {
    const usableW = state.boardBounds.right - state.boardBounds.left;
    const maxRowIdx = Math.max(1, (state.rowCounts?.length || 8) - 1);
    const layerProgress = knob.row / maxRowIdx;
    const dx = targetX - knob.x;
    const normalizedDelta = dx / (usableW * 0.25);
    const focusGain = 0.78 + layerProgress * 0.45;
    const rawWeight = Math.tanh(normalizedDelta * focusGain);
    return Math.max(-0.95, Math.min(0.95, rawWeight));
  }

  /**
   * Builds the geometry of the top diverging rails, the staggered rows of knobs
   * according to state.rowCounts, and the bottom vocabulary bins.
   */
  function layoutBoard(preserveWeights = false) {
    const W = state.width;
    const H = state.height;

    const marginX = Math.max(20, W * 0.028);
    const left = marginX;
    const right = W - marginX;
    const usableW = right - left;

    // Top oblique rail track scaled to canvas height H
    const railStartX = left + usableW * 0.045;
    const railEndX = right - usableW * 0.045;
    const railStartY = Math.max(22, Math.min(34, H * 0.048));
    const railEndY = railStartY + Math.max(24, Math.min(42, H * 0.058));

    const knobsTop = railEndY + Math.max(24, Math.min(38, H * 0.055));
    const binsBottom = H - Math.max(54, Math.min(72, H * 0.11));
    const binsTop = binsBottom - Math.max(76, Math.min(125, H * 0.18));
    const knobsBottom = binsTop - Math.max(20, Math.min(32, H * 0.045));

    const rowCounts =
      Array.isArray(state.rowCounts) && state.rowCounts.length > 0
        ? state.rowCounts
        : [22, 23, 22, 23, 22, 23, 22, 23];
    const numRows = rowCounts.length;
    const maxRowCount = Math.max(...rowCounts);
    const minRowCount = Math.min(...rowCounts);
    const refCount = maxRowCount - minRowCount === 1 ? minRowCount : maxRowCount;

    const rowSpacingY =
      numRows > 1 ? (knobsBottom - knobsTop) / (numRows - 1) : knobsBottom - knobsTop;
    const baseSpan = usableW * 0.90;
    const stepX = baseSpan / Math.max(1, refCount - 1);
    const knobRadius = Math.max(4.2, Math.min(7.4, stepX * 0.195, rowSpacingY * 0.22));

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
      binsBottom,
      knobStepX: stepX,
      knobRadius
    };

    // Update bottom vocabulary bins
    const numBins = VOCABULARY.length;
    const binWidth = usableW / numBins;
    state.bins.forEach((bin, i) => {
      bin.xLeft = left + i * binWidth;
      bin.xRight = bin.xLeft + binWidth;
      bin.xCenter = bin.xLeft + binWidth * 0.5;
    });

    const previousKnobs = preserveWeights ? state.knobs : null;
    const newKnobs = [];
    let globalIdx = 0;

    for (let r = 0; r < numRows; r++) {
      const count = rowCounts[r];
      const tRow = numRows === 1 ? 0.5 : r / (numRows - 1);
      const y = knobsTop + tRow * (knobsBottom - knobsTop);

      const rowSpan = (count - 1) * stepX;
      // If adjacent rows have the same parity, stagger odd rows by ±0.25 * stepX
      const prevCount = r > 0 ? rowCounts[r - 1] : count + 1;
      const sameParityAsPrev = count % 2 === prevCount % 2;
      const parityStagger = sameParityAsPrev ? (r % 2 === 1 ? 0.25 : -0.25) * stepX : 0;
      const rowLeft = W * 0.5 - rowSpan * 0.5 + parityStagger;

      for (let c = 0; c < count; c++) {
        const x = rowLeft + c * stepX;
        const prev = previousKnobs && previousKnobs[globalIdx];
        const prevBaseW = prev ? (prev.baseWeight ?? prev.weight) : 0;

        newKnobs.push({
          id: globalIdx,
          row: r,
          col: c,
          x,
          y,
          radius: knobRadius,
          baseWeight: prevBaseW,
          weight: prevBaseW,
          displayWeight: prevBaseW,
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

  /**
   * Returns all knobs to their post-Backprop base weight (clearing temporary ball collision nudges).
   */
  function resetKnobsToBaseWeights() {
    for (const knob of state.knobs) {
      knob.weight = knob.baseWeight;
    }
  }

  function updateMemoryBadgeUI() {
    if (bpMemoryBadge) {
      const n = state.trainedPrompts.size;
      bpMemoryBadge.textContent = `🧠 ${n} ${n === 1 ? "rule" : "rules"}`;
    }
    renderTrainedMemoryStripHtml();
  }

  function renderTrainedMemoryStripHtml() {
    if (!bpMemoryStrip) return "";
    if (state.trainedPrompts.size === 0) {
      bpMemoryStrip.innerHTML = `<span class="bp-memory-empty">No rules yet</span>`;
      return "";
    }
    const activeKey = getNormalizedPromptKey(promptInput.value);
    const chips = [];
    for (const [pKey, entry] of state.trainedPrompts.entries()) {
      const targetW = VOCABULARY[entry.targetBinIndex]?.word || "?";
      const shortPrompt = pKey.length > 20 ? `…${pKey.slice(-19)}` : pKey;
      const isCurrent = pKey === activeKey;
      const statusPrefix = entry.converged ? "✓ " : isCurrent ? "● " : "";
      chips.push(
        `<button type="button" class="bp-memory-chip${isCurrent ? " active-rule" : ""}" data-prompt="${escapeHtml(pKey)}" title="Click to test trained rule: “${escapeHtml(pKey)}” → “${escapeHtml(targetW)}”">${statusPrefix}“${escapeHtml(shortPrompt)}” → <strong>${escapeHtml(targetW)}</strong></button>`
      );
    }
    bpMemoryStrip.innerHTML = chips.join("");
    return "";
  }

  function applyKnobPreset(mode, clearMemory = false) {
    state.mode = mode;

    Object.entries(presetBtns).forEach(([key, btn]) => {
      if (btn) btn.classList.toggle("active", key === mode);
    });

    if (clearMemory && (mode === "neutral" || mode === "random")) {
      state.trainedPrompts.clear();
      updateMemoryBadgeUI();
    }

    if (mode === "random") {
      state.knobs.forEach((knob) => {
        const rand = Math.random() * 2 - 1;
        const w = Math.round(rand * 100) / 100;
        knob.baseWeight = w;
        knob.weight = w;
        knob.lastDelta = 0;
      });
      return;
    }

    // Default / "neutral": all knobs at 0°
    state.knobs.forEach((knob) => {
      knob.baseWeight = 0;
      knob.weight = 0;
      knob.lastDelta = 0;
    });
  }

  /**
   * Executes one step of Backpropagation on the shared physical knobs.
   */
  function performBackpropStep(launchAfter = true) {
    const targetBin = state.bins[state.backpropTargetBinIndex];
    if (!targetBin) return;

    const promptKey = getNormalizedPromptKey(promptInput.value) || "default";
    const tokens = parsePromptTokens(promptInput.value);
    const { railStartX, railEndX, left, right } = state.boardBounds;
    const usableW = right - left;
    const targetX = targetBin.xCenter;
    const lr = state.learningRate;

    // 1. Return all knobs to their post-Backprop base state before applying gradient update
    resetKnobsToBaseWeights();

    let memEntry = state.trainedPrompts.get(promptKey);
    if (!memEntry) {
      memEntry = {
        promptKey,
        targetBinIndex: state.backpropTargetBinIndex,
        epochs: 0
      };
      state.trainedPrompts.set(promptKey, memEntry);
    } else if (memEntry.targetBinIndex !== state.backpropTargetBinIndex) {
      memEntry.targetBinIndex = state.backpropTargetBinIndex;
    }
    memEntry.epochs += 1;
    updateMemoryBadgeUI();

    let sumDropX = 0;
    let countDropX = 0;
    tokens.forEach((tok, idx) => {
      tok.vector.forEach((comp) => {
        const normalizedSize = Math.max(0.04, Math.min(0.96, (comp.decimal - 0.06) / 0.92));
        const wordOffset = tokens.length > 1 ? ((idx / (tokens.length - 1)) - 0.5) * 0.038 : 0;
        const ctxShift = tok.contextSig ? tok.contextSig.subChannelShift : 0;
        const tDrop = Math.max(0.04, Math.min(0.96, normalizedSize + wordOffset + ctxShift));
        sumDropX += railStartX + tDrop * (railEndX - railStartX);
        countDropX++;
      });
    });
    const promptDropCenterX = countDropX > 0 ? sumDropX / countDropX : (railStartX + railEndX) * 0.5;

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
    const maxRowIdx = Math.max(1, (state.rowCounts?.length || 8) - 1);

    state.knobs.forEach((knob) => {
      const layerProgress = knob.row / maxRowIdx;
      const touchInfo = knobStats.get(knob.id);
      const wasTouched = Boolean(touchInfo && touchInfo.touches > 0);

      const corridorCenterX =
        promptDropCenterX * (1 - layerProgress * 0.85) + targetX * (layerProgress * 0.85);
      const corridorHalfWidth = usableW * (0.18 - layerProgress * 0.035);
      const distToCorridor = Math.abs(knob.x - corridorCenterX);
      const distToTargetX = Math.abs(knob.x - targetX);
      const inActiveCorridor =
        distToCorridor <= corridorHalfWidth ||
        (layerProgress >= 0.5 && distToTargetX <= usableW * 0.16);

      if (!wasTouched && !inActiveCorridor) return;

      const currentBase = knob.baseWeight;
      const idealW = computeIdealKnobWeightForTargetX(knob, targetX);
      let deltaW = 0;

      if (wasTouched) {
        const avgLandedX = touchInfo.sumLandedX / touchInfo.massSum;
        const empiricalPush = Math.tanh((targetX - avgLandedX) / (usableW * 0.2));
        const structuralDelta = idealW - currentBase;
        const credit = Math.min(
          1.45,
          0.65 + (touchInfo.touches / Math.max(1, state.landedHistory.length)) * 1.5
        );
        deltaW = lr * credit * (0.7 * structuralDelta + 0.45 * empiricalPush);
      } else if (inActiveCorridor) {
        const zoneFactor = Math.max(0.35, 1 - (distToCorridor / Math.max(1, corridorHalfWidth)) * 0.55);
        deltaW = lr * 0.65 * zoneFactor * (idealW - currentBase);
      }

      deltaW = Math.max(-0.32, Math.min(0.32, deltaW));

      if (Math.abs(deltaW) >= 0.006) {
        const nextW = Math.max(-0.96, Math.min(0.96, currentBase + deltaW));
        const actualDelta = nextW - currentBase;
        const roundedW = Math.round(nextW * 100) / 100;
        knob.baseWeight = roundedW;
        knob.weight = roundedW;
        knob.lastDelta = actualDelta;
        knob.bpDelay = (maxRowIdx - knob.row) * 0.045;
        knob.bpGlow = 1.0;

        adjustedCount++;
        sumAbsDelta += Math.abs(actualDelta);
      }
    });

    // 2. Leave all knobs in their post-Backprop state for the next launch
    resetKnobsToBaseWeights();

    spawnBackpropPulses(targetBin);

    state.bpEpochs += 1;
    bpEpochCountEl.textContent = String(state.bpEpochs);

    const avgDelta = adjustedCount > 0 ? (sumAbsDelta / adjustedCount).toFixed(2) : "0.00";
    bpFeedbackLog.innerHTML = `<strong>Epoch #${state.bpEpochs}:</strong> ${adjustedCount} knobs (Δ ±${avgDelta}) for <strong>“${escapeHtml(promptKey)}” → “${escapeHtml(targetBin.word)}”</strong>.${renderTrainedMemoryStripHtml()}`;

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
      tokenCountBadge.textContent = `${tokens.length} ${tokens.length === 1 ? "word" : "words"} · ${tokens.length * state.ballsPerWord} balls`;
    }
    tokenChipsContainer.innerHTML = "";

    tokens.forEach((tok) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "token-chip";
      const decimalsStr = tok.vector.map((c) => c.decimal.toFixed(2)).join(", ");
      chip.title = `Contextual vector for "${tok.word}": [${decimalsStr}] — Click to launch only this word along the rails`;

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
        if (state.balls.length === 0 && state.spawnQueue.length === 0) {
          resetKnobsToBaseWeights();
        }
        enqueueWordVectors([tok]);
      });
      tokenChipsContainer.appendChild(chip);
    });

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
          totalTokens: tokens.length,
          contextSig: tok.contextSig
        });
      });
    }
    updateStatusUI();
  }

  /**
   * Instancia una bola en el inicio izquierdo de la vía superior de dos carriles divergentes
   */
  function spawnBallFromQueue(item) {
    const { railStartX, railStartY, knobStepX, knobRadius } = state.boardBounds;
    const usableW = state.boardBounds.right - state.boardBounds.left;

    // Espacio libre entre las superficies de dos knobs vecinos en una misma fila
    const clearGap = Math.max(14, (knobStepX || 32) - 2 * (knobRadius || 6));
    const maxSafeRadiusForTier = clearGap * (0.21 + (item.tier / 5) * 0.21);

    const baseR = Math.max(
      item.spec.minRadius,
      Math.min(item.spec.maxRadius, usableW * item.spec.radiusRatio)
    );
    const r = Math.min(baseR, maxSafeRadiusForTier);

    const dropX = computeDropXForBall(
      item.decimal,
      item.tokenIndex,
      item.totalTokens,
      item.contextSig
    );

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
      vx: 342.5, // velocidad de rodadura (fija, determinista) a lo largo de los dos carriles oblicuos
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
    state.spawnTimer = 0;
    state.landedHistory = [];
    state.railDropRings = [];
    resetKnobsToBaseWeights();
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

    if (metricActiveBalls) {
      metricActiveBalls.textContent = String(state.balls.length + state.spawnQueue.length);
    }
    if (metricLandedBalls) {
      metricLandedBalls.textContent = String(totalLanded);
    }

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
      if (winnerWordEl) winnerWordEl.textContent = "—";
      if (winnerStatsEl) winnerStatsEl.textContent = "Launch balls to see distribution";
      completedSentenceEl.innerHTML = `<span class="prompt-part">${escapeHtml(promptText || "…")}</span> <span class="predicted-part">…</span>`;
      btnAppendWinner.disabled = true;
      top3CandidatesEl.innerHTML = [0, 1, 2]
        .map(
          (idx) => `
          <div class="top3-row">
            <strong>#${idx + 1} —</strong>
            <div class="top3-bar-wrap">
              <div class="top3-bar-fill" style="width: 0%"></div>
            </div>
            <span>0 (0% · 🎲0%)</span>
          </div>
        `
        )
        .join("");
      return;
    }

    const sorted = [...state.bins].sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return b.weightedScore - a.weightedScore;
    });

    const winner = sorted[0];
    const pct = Math.round((winner.count / totalLanded) * 100);

    if (winnerWordEl) winnerWordEl.textContent = `“${winner.word}”`;
    if (winnerStatsEl) {
      winnerStatsEl.textContent = `${winner.count}/${totalLanded} balls (${pct}% · Σ ${winner.weightedScore.toFixed(1)})`;
    }

    completedSentenceEl.innerHTML = `<span class="prompt-part">${escapeHtml(promptText)}</span> <span class="predicted-part">${escapeHtml(winner.word)}</span>`;
    btnAppendWinner.disabled = false;

    const top3Active = sorted.slice(0, 3).filter((b) => b.count > 0);
    const samplingProbs = computeTop3SamplingProbs(top3Active);
    top3CandidatesEl.innerHTML = [0, 1, 2]
      .map((idx) => {
        const item = top3Active[idx];
        if (!item) {
          return `
            <div class="top3-row">
              <strong>#${idx + 1} —</strong>
              <div class="top3-bar-wrap">
                <div class="top3-bar-fill" style="width: 0%"></div>
              </div>
              <span>0 (0% · 🎲0%)</span>
            </div>
          `;
        }
        const itemPct = Math.round((item.count / totalLanded) * 100);
        const samplePct = Math.round((samplingProbs[idx] || 0) * 100);
        const isTarget = item.index === state.backpropTargetBinIndex;
        return `
          <div class="top3-row" title="Balls: ${item.count}/${totalLanded} (${itemPct}%) · Sampling probability at Temperature ${state.temperature.toFixed(2)}: ${samplePct}%">
            <strong>#${idx + 1} ${escapeHtml(item.word)} ${isTarget ? "🎯" : ""}</strong>
            <div class="top3-bar-wrap">
              <div class="top3-bar-fill" style="width: ${itemPct}%"></div>
            </div>
            <span>${item.count} (${itemPct}% · 🎲${samplePct}%)</span>
          </div>
        `;
      })
      .join("");
  }

  /**
   * Computes sampling probabilities (Softmax with Temperature) across the Top-3 candidate words
   */
  function computeTop3SamplingProbs(top3) {
    if (!top3 || top3.length === 0) return [];
    if (top3.length === 1) return [1.0];
    const temp = Math.max(0.02, state.temperature);
    const scale = Math.max(0.07, temp * 1.35);
    const totalTop3Balls = top3.reduce((acc, b) => acc + b.count, 0) || 1;
    const logits = top3.map((b) => {
      const share = b.count / totalTop3Balls;
      const avgDec = b.weightedScore / Math.max(1, b.count);
      return (Math.log(Math.max(0.01, share)) + avgDec * 0.12) / scale;
    });
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((e) => e / sumExp);
  }

  function sampleWinnerFromTop3() {
    const sorted = [...state.bins].sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return b.weightedScore - a.weightedScore;
    });
    const top3 = sorted.slice(0, 3).filter((b) => b.count > 0);
    if (!top3.length) return null;
    const probs = computeTop3SamplingProbs(top3);
    const r = Math.random();
    let accum = 0;
    for (let i = 0; i < top3.length; i++) {
      accum += probs[i];
      if (r <= accum) {
        return { chosen: top3[i], rank: i + 1, prob: probs[i] };
      }
    }
    return { chosen: top3[0], rank: 1, prob: probs[0] || 1 };
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
    if (metricActiveBalls) {
      metricActiveBalls.textContent = String(activeCount);
    }
    if (!simStatusDot || !simStatusText) return;

    if (activeCount > 0) {
      simStatusDot.classList.add("busy");
      simStatusText.textContent = `Rolling along rails and dropping by caliber (${activeCount} balls active)…`;
    } else {
      simStatusDot.classList.remove("busy");
      simStatusText.textContent = "Ready — Click a bottom word to set it as target 🎯 or drag any knob";
    }
  }

  /**
   * Quasi-physical simulation step:
   * Phase 1: Ball rolls along the top diverging rails until the gap exceeds its diameter.
   * Phase 2: Ball drops through the rows of knobs toward the bottom vocabulary bins.
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

          const activeKey = getNormalizedPromptKey(promptInput.value) || "default";
          const activeEntry = state.trainedPrompts.get(activeKey);

          if (isTargetWinner) {
            if (activeEntry) activeEntry.converged = true;
            state.autoTrainPassSuccessCount += 1;
            renderTrainedMemoryStripHtml();
          } else {
            if (activeEntry) activeEntry.converged = false;
            state.autoTrainPassSuccessCount = 0;
            for (const entry of state.trainedPrompts.values()) {
              if (entry !== activeEntry) entry.converged = false;
            }
            performBackpropStep(false);
          }

          const keys = Array.from(state.trainedPrompts.keys());
          if (keys.length > 0 && state.autoTrainPassSuccessCount >= keys.length) {
            stopAutoTrain(
              keys.length === 1
                ? `✅ Convergence reached in ${state.bpEpochs} epochs! “${VOCABULARY[state.backpropTargetBinIndex].word}” wins with ${Math.round((targetCount / total) * 100)}%.`
                : `✅ Convergence completed across all ${keys.length} rules in memory after ${state.bpEpochs} epochs!`
            );
            return;
          }

          if (keys.length > 1) {
            const currentIdx = Math.max(0, keys.indexOf(activeKey));
            state.autoTrainIndex = (currentIdx + 1) % keys.length;
            const nextKey = keys[state.autoTrainIndex];
            const nextEntry = state.trainedPrompts.get(nextKey);
            promptInput.value = nextKey;
            if (nextEntry && nextEntry.targetBinIndex < VOCABULARY.length) {
              setBackpropTargetBin(nextEntry.targetBinIndex, false);
            }
            renderTokenChips();
            renderTrainedMemoryStripHtml();
          }

          clearBallsAndBins(true);
          triggerBurstDrop();
        }, Math.max(45, Math.round(440 / Math.max(0.2, state.simSpeed))));
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
          ball.vx = 18;
          ball.vy = 61;

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
          // Empujón determinista: aleja la bola del knob más cercano (desempate por la inclinación del knob)
          let nearestKnob = null;
          let nearestDistSq = Infinity;
          for (const knob of state.knobs) {
            const kdx = ball.x - knob.x;
            const kdy = ball.y - knob.y;
            const d2 = kdx * kdx + kdy * kdy;
            if (d2 < nearestDistSq) {
              nearestDistSq = d2;
              nearestKnob = knob;
            }
          }
          let pushDir = 1;
          if (nearestKnob) {
            const kdx = ball.x - nearestKnob.x;
            if (kdx !== 0) pushDir = kdx > 0 ? 1 : -1;
            else if (nearestKnob.weight !== 0) pushDir = nearestKnob.weight > 0 ? 1 : -1;
          }
          ball.vx += pushDir * 47.5;
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
            // Sin dispersión térmica aleatoria: la Temperatura solo afecta al muestreo de la palabra ganadora.
            const knobDeflectionVx = w * 185 * state.knobStrength * massStability;

            ball.vx = ball.vx * 0.3 + knobDeflectionVx;

            if (Math.abs(ball.vx) < 22) {
              ball.vx += (nx >= 0 ? 1 : -1) * 32.5;
            }

            if (ball.vy < 48 && ny < 0) {
              const maxUpBounce = -56 + ball.tier * 6;
              ball.vy = Math.max(maxUpBounce, ball.vy * 0.5);
            }

            // Al golpear el knob, la bola lo mueve según su tamaño/masa y la resistencia configurada de los knobs
            const isFirstTouchOnKnob = !ball.touchedKnobs.has(knob.id);
            const yieldFactor = Math.max(0, (1 - state.knobResistance) / 0.25); // 1.0 al 75% de resistencia, 0.0 al 100%, 4.0 al 0%
            if (isFirstTouchOnKnob && yieldFactor > 0) {
              const baseSizeNudge = 0.014 + (ball.tier / 5) * 0.042; // 0.014 (roja pequeña) a 0.056 (violeta grande)
              const sizeNudge = baseSizeNudge * yieldFactor;
              const torqueDir = nx >= 0 ? 1 : -1;
              const maxTempOffset = Math.min(0.65, 0.18 * yieldFactor);
              const currentOffset = knob.weight - knob.baseWeight;
              const nextOffset = Math.max(
                -maxTempOffset,
                Math.min(maxTempOffset, currentOffset + torqueDir * sizeNudge)
              );
              knob.weight = Math.max(
                -0.98,
                Math.min(0.98, Math.round((knob.baseWeight + nextOffset) * 1000) / 1000)
              );
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
                jitterX: (((targetBin.settledBalls.length * 7) % 5) - 2) // desfase visual determinista
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

    const { knobsTop, binsTop, binsBottom } = state.boardBounds;

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
    const baseW = typeof knob.baseWeight === "number" ? knob.baseWeight : w;
    const maxTiltRad = 0.98;
    const tiltAngle = w * maxTiltRad;
    const baseTiltAngle = baseW * maxTiltRad;

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

    // Si las bolas han movido un poco el knob respecto a su estado base post-Backprop, dibujar marca testigo
    if (Math.abs(w - baseW) > 0.01) {
      ctx.save();
      ctx.rotate(baseTiltAngle);
      ctx.beginPath();
      ctx.moveTo(0, -1);
      ctx.lineTo(0, -r + 2.2);
      ctx.lineWidth = 1.3;
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(192, 132, 252, 0.72)";
      ctx.stroke();
      ctx.restore();
    }

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
    let bestKnob = null;
    let bestDistSq = Infinity;
    for (const knob of state.knobs) {
      const dx = x - knob.x;
      const dy = y - knob.y;
      const hitR = knob.radius + 5.5;
      const distSq = dx * dx + dy * dy;
      if (distSq <= hitR * hitR && distSq < bestDistSq) {
        bestDistSq = distSq;
        bestKnob = knob;
      }
    }
    return bestKnob;
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
    if (!hoveredKnobPill) return;
    if (!knob) {
      hoveredKnobPill.innerHTML = `Knob: <strong>none</strong>`;
      return;
    }
    const sign = knob.weight > 0 ? "+" : "";
    const dir = knob.weight > 0.05 ? "→ Right" : knob.weight < -0.05 ? "← Left" : "Center";
    hoveredKnobPill.innerHTML = `Knob R${knob.row + 1}·N${knob.col + 1}: <strong>${sign}${knob.weight.toFixed(2)} (${dir})</strong>`;
  }

  canvas.addEventListener("pointerdown", (evt) => {
    const pos = getCanvasCoords(evt);
    const knob = findKnobAt(pos.x, pos.y);

    if (knob) {
      state.draggingKnob = knob;
      state.dragStartX = pos.x;
      state.dragStartY = pos.y;
      state.dragStartWeight = knob.baseWeight;
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
      const roundedW = Math.round(newW * 100) / 100;
      k.baseWeight = roundedW;
      k.weight = roundedW;
      k.hitEnergy = 0.6;
      state.mode = "custom";
      Object.values(presetBtns).forEach((btn) => btn && btn.classList.remove("active"));
      updateHoveredKnobUI(k);
      bpFeedbackLog.innerHTML = `Manual adjustment on <strong>Knob R${k.row + 1}·N${k.col + 1}</strong> → weight <strong>${k.weight > 0 ? "+" : ""}${k.weight.toFixed(2)}</strong>.${renderTrainedMemoryStripHtml()}`;
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
      const roundedW = Math.max(-0.98, Math.min(0.98, Math.round((knob.baseWeight + step) * 100) / 100));
      knob.baseWeight = roundedW;
      knob.weight = roundedW;
      knob.hitEnergy = 0.7;
      state.mode = "custom";
      Object.values(presetBtns).forEach((btn) => btn && btn.classList.remove("active"));
      updateHoveredKnobUI(knob);
      bpFeedbackLog.innerHTML = `Wheel adjustment on <strong>Knob R${knob.row + 1}·N${knob.col + 1}</strong> → weight <strong>${knob.weight > 0 ? "+" : ""}${knob.weight.toFixed(2)}</strong>.${renderTrainedMemoryStripHtml()}`;
    },
    { passive: false }
  );

  function applySelectedPrompt(newPromptText) {
    stopAutoTrain();
    promptInput.value = newPromptText;
    clearBallsAndBins(true);
    renderTokenChips();
    const promptKey = getNormalizedPromptKey(promptInput.value);
    if (state.trainedPrompts.has(promptKey)) {
      setBackpropTargetBin(state.trainedPrompts.get(promptKey).targetBinIndex, false);
    }
  }

  if (presetPromptSelect) {
    presetPromptSelect.addEventListener("change", () => {
      applySelectedPrompt(presetPromptSelect.value);
    });
  }

  // Prompt Examples Selection Dialog
  const btnOpenExamples = document.getElementById("btn-open-examples");
  const examplesDialog = document.getElementById("examples-dialog");
  const examplesList = document.getElementById("examples-list");
  const btnCloseExamples = document.getElementById("btn-close-examples");
  const btnCloseExamplesX = document.getElementById("btn-close-examples-x");

  if (btnOpenExamples && examplesDialog) {
    btnOpenExamples.addEventListener("click", () => {
      examplesDialog.showModal();
    });

    if (examplesList) {
      examplesList.addEventListener("click", (evt) => {
        const btn = evt.target.closest(".example-item-btn");
        if (!btn || !btn.dataset.prompt) return;
        applySelectedPrompt(btn.dataset.prompt);
        examplesDialog.close();
      });
    }

    if (btnCloseExamples) {
      btnCloseExamples.addEventListener("click", () => examplesDialog.close());
    }
    if (btnCloseExamplesX) {
      btnCloseExamplesX.addEventListener("click", () => examplesDialog.close());
    }

    if (!("closedBy" in HTMLDialogElement.prototype)) {
      examplesDialog.addEventListener("click", (event) => {
        if (event.target !== examplesDialog) return;
        const rect = examplesDialog.getBoundingClientRect();
        const isInside =
          rect.top <= event.clientY &&
          event.clientY <= rect.top + rect.height &&
          rect.left <= event.clientX &&
          event.clientX <= rect.left + rect.width;
        if (!isInside) examplesDialog.close();
      });
    }
  }

  promptInput.addEventListener("input", () => {
    renderTokenChips();
    const promptKey = getNormalizedPromptKey(promptInput.value);
    if (state.trainedPrompts.has(promptKey)) {
      setBackpropTargetBin(state.trainedPrompts.get(promptKey).targetBinIndex, false);
    }
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
    // Clear board automatically on every launch and reset knobs to post-Backprop base weights
    clearBallsAndBins(true);
    enqueueWordVectors(tokens);
  }

  btnDropBurst.addEventListener("click", () => {
    triggerBurstDrop();
  });

  if (btnDropSingle) {
    btnDropSingle.addEventListener("click", () => {
      const tokens = parsePromptTokens(promptInput.value, 3);
      if (!tokens.length) return;
      clearBallsAndBins(true);
      enqueueWordVectors(tokens);
    });
  }

  if (btnClearBalls) {
    btnClearBalls.addEventListener("click", () => {
      stopAutoTrain();
      clearBallsAndBins(true);
    });
  }

  btnAppendWinner.addEventListener("click", () => {
    stopAutoTrain();
    const sampled = sampleWinnerFromTop3();
    if (!sampled || !sampled.chosen || sampled.chosen.count === 0) return;

    const chosenWord = sampled.chosen.word;
    const probPct = Math.round(sampled.prob * 100);
    bpFeedbackLog.innerHTML = `🎲 Top-3 Sampling (Temp ${state.temperature.toFixed(2)}): picked <strong>#${sampled.rank} “${escapeHtml(chosenWord)}”</strong> (${probPct}% prob).`;

    promptInput.value = `${promptInput.value.trim()} ${chosenWord}`.trim();
    clearBallsAndBins(true);
    renderTokenChips();
    const promptKey = getNormalizedPromptKey(promptInput.value);
    if (state.trainedPrompts.has(promptKey)) {
      setBackpropTargetBin(state.trainedPrompts.get(promptKey).targetBinIndex, false);
    }
    renderTrainedMemoryStripHtml();
    triggerBurstDrop();
  });

  function handleMemoryChipClick(evt) {
    const chip = evt.target.closest(".bp-memory-chip");
    if (!chip) return;
    const pKey = chip.dataset.prompt;
    if (!pKey) return;
    stopAutoTrain();
    promptInput.value = pKey;
    clearBallsAndBins(true);
    renderTokenChips();
    const mem = state.trainedPrompts.get(pKey);
    if (mem && mem.targetBinIndex < VOCABULARY.length) {
      setBackpropTargetBin(mem.targetBinIndex, false);
    }
    renderTrainedMemoryStripHtml();
    triggerBurstDrop();
  }

  if (bpMemoryStrip) {
    bpMemoryStrip.addEventListener("click", handleMemoryChipClick);
  }
  bpFeedbackLog.addEventListener("click", handleMemoryChipClick);

  // Machine Configuration Dialog (top-right gear button)
  const canvasWrapper = document.getElementById("canvas-wrapper");
  const btnOpenConfig = document.getElementById("btn-open-config");
  const configDialog = document.getElementById("config-dialog");
  const btnCloseConfig = document.getElementById("btn-close-config");
  const btnCloseConfigX = document.getElementById("btn-close-config-x");
  const configStateFeedback = document.getElementById("config-state-feedback");
  const configCanvasWidthInput = document.getElementById("config-canvas-width");
  const configCanvasHeightInput = document.getElementById("config-canvas-height");
  const btnCanvasAutofit = document.getElementById("btn-canvas-autofit");
  const configRowsInput = document.getElementById("config-rows-input");
  const configRowsSummary = document.getElementById("config-rows-summary");
  const btnAutoRows = document.getElementById("btn-auto-rows");
  const btnApplyGeometry = document.getElementById("btn-apply-geometry");

  function updateConfigRowsSummaryUI(counts) {
    if (!configRowsSummary) return;
    const list = Array.isArray(counts) && counts.length > 0 ? counts : state.rowCounts;
    const totalKnobs = list.reduce((a, b) => a + b, 0);
    configRowsSummary.textContent = `${list.length} ${list.length === 1 ? "row" : "rows"} · ${totalKnobs} knobs`;
  }

  function syncConfigDialogFields() {
    if (configCanvasWidthInput) configCanvasWidthInput.value = String(state.width);
    if (configCanvasHeightInput) configCanvasHeightInput.value = String(state.height);
    if (configRowsInput) configRowsInput.value = state.rowCounts.join(", ");
    updateConfigRowsSummaryUI(state.rowCounts);
  }

  function applyCanvasDimensions(w, h, isCustom) {
    const dpr = window.devicePixelRatio || 1;
    state.dpr = dpr;
    state.width = Math.round(w);
    state.height = Math.round(h);

    if (isCustom) {
      state.customCanvasSize = { width: state.width, height: state.height };
      if (canvasWrapper) canvasWrapper.classList.add("custom-canvas-size");
      canvas.style.width = `${state.width}px`;
      canvas.style.height = `${state.height}px`;
    } else {
      state.customCanvasSize = null;
      if (canvasWrapper) canvasWrapper.classList.remove("custom-canvas-size");
      canvas.style.width = "100%";
      canvas.style.height = "100%";
    }

    canvas.width = Math.round(state.width * dpr);
    canvas.height = Math.round(state.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  if (btnOpenConfig && configDialog) {
    btnOpenConfig.addEventListener("click", () => {
      syncConfigDialogFields();
      configDialog.showModal();
    });
    if (btnCloseConfig) {
      btnCloseConfig.addEventListener("click", () => configDialog.close());
    }
    if (btnCloseConfigX) {
      btnCloseConfigX.addEventListener("click", () => configDialog.close());
    }
    if (!("closedBy" in HTMLDialogElement.prototype)) {
      configDialog.addEventListener("click", (event) => {
        if (event.target !== configDialog) return;
        const rect = configDialog.getBoundingClientRect();
        const isInside =
          rect.top <= event.clientY &&
          event.clientY <= rect.top + rect.height &&
          rect.left <= event.clientX &&
          event.clientX <= rect.left + rect.width;
        if (!isInside) configDialog.close();
      });
    }
  }

  // Automatically pre-configure recommended rows & knobs when canvas dimensions change in the dialog
  function handleCanvasDimensionInputChange() {
    if (!configCanvasWidthInput || !configCanvasHeightInput || !configRowsInput) return;
    const w = Math.max(480, Math.min(1920, Number(configCanvasWidthInput.value) || state.width));
    const h = Math.max(420, Math.min(1400, Number(configCanvasHeightInput.value) || state.height));
    const recommended = computeRecommendedRowCounts(w, h);
    configRowsInput.value = recommended.join(", ");
    updateConfigRowsSummaryUI(recommended);
  }

  if (configCanvasWidthInput) {
    configCanvasWidthInput.addEventListener("input", handleCanvasDimensionInputChange);
  }
  if (configCanvasHeightInput) {
    configCanvasHeightInput.addEventListener("input", handleCanvasDimensionInputChange);
  }

  if (configRowsInput) {
    configRowsInput.addEventListener("input", () => {
      const parsed = parseRowCountsString(configRowsInput.value);
      if (parsed) {
        updateConfigRowsSummaryUI(parsed);
      }
    });
  }

  if (btnAutoRows) {
    btnAutoRows.addEventListener("click", () => {
      const w = Math.max(480, Math.min(1920, Number(configCanvasWidthInput?.value) || state.width));
      const h = Math.max(420, Math.min(1400, Number(configCanvasHeightInput?.value) || state.height));
      const recommended = computeRecommendedRowCounts(w, h);
      if (configRowsInput) configRowsInput.value = recommended.join(", ");
      updateConfigRowsSummaryUI(recommended);
    });
  }

  if (btnCanvasAutofit) {
    btnCanvasAutofit.addEventListener("click", () => {
      stopAutoTrain();
      state.customCanvasSize = null;
      state.customRowCounts = false;
      if (canvasWrapper) canvasWrapper.classList.remove("custom-canvas-size");
      canvas.style.width = "100%";
      canvas.style.height = "100%";

      const rect = canvasWrapper ? canvasWrapper.getBoundingClientRect() : { width: 960, height: 700 };
      const w = Math.max(480, Math.round(rect.width));
      const h = Math.max(420, Math.round(rect.height));
      applyCanvasDimensions(w, h, false);
      state.rowCounts = computeRecommendedRowCounts(w, h);
      layoutBoard(false);
      clearBallsAndBins(true);
      syncConfigDialogFields();

      const totalKnobs = state.knobs.length;
      const msg = `↔ Canvas fitted to window (${w}×${h}px) with auto layout: ${state.rowCounts.length} rows (${totalKnobs} knobs).`;
      bpFeedbackLog.innerHTML = msg;
      if (configStateFeedback) configStateFeedback.innerHTML = msg;
    });
  }

  if (btnApplyGeometry) {
    btnApplyGeometry.addEventListener("click", () => {
      stopAutoTrain();
      const w = Math.max(480, Math.min(1920, Number(configCanvasWidthInput?.value) || state.width));
      const h = Math.max(420, Math.min(1400, Number(configCanvasHeightInput?.value) || state.height));
      const parsedRows =
        parseRowCountsString(configRowsInput?.value) || computeRecommendedRowCounts(w, h);

      const sizeChanged = w !== state.width || h !== state.height;
      applyCanvasDimensions(w, h, sizeChanged || Boolean(state.customCanvasSize));
      state.rowCounts = parsedRows;
      state.customRowCounts = true;
      state.trainedPrompts.clear();
      layoutBoard(false);
      clearBallsAndBins(true);
      updateMemoryBadgeUI();
      syncConfigDialogFields();

      const msg = `✓ Applied canvas ${w}×${h}px and ${state.rowCounts.length} rows [${state.rowCounts.join(", ")}] (${state.knobs.length} knobs).`;
      bpFeedbackLog.innerHTML = msg;
      if (configStateFeedback) configStateFeedback.innerHTML = msg;
    });
  }

  Object.entries(presetBtns).forEach(([modeKey, btn]) => {
    if (!btn) return;
    btn.addEventListener("click", () => {
      stopAutoTrain();
      applyKnobPreset(modeKey, true);
      clearBallsAndBins(true);
      const msg =
        modeKey === "random"
          ? "🎲 Knobs set to random weights. Click “▶ Launch Prompt” or train with “⚡ Backprop”."
          : modeKey === "neutral"
            ? "⚖️ Knobs set to neutral (0°). Click “▶ Launch Prompt” or train with “⚡ Backprop”."
            : "🎯 Pre-trained knobs ready. Click “▶ Launch Prompt”.";
      bpFeedbackLog.innerHTML = msg;
      if (configStateFeedback) {
        configStateFeedback.innerHTML = msg;
      }
    });
  });

  // Save and Load Full Machine State
  const btnSaveState = document.getElementById("btn-save-state");
  const btnLoadState = document.getElementById("btn-load-state");
  const SAVED_STATE_KEY = "galton_llm_saved_state_v1";

  if (btnSaveState) {
    btnSaveState.addEventListener("click", () => {
      stopAutoTrain();
      const serializedRules = [];
      for (const [pKey, entry] of state.trainedPrompts.entries()) {
        serializedRules.push({
          promptKey: pKey,
          targetBinIndex: entry.targetBinIndex,
          epochs: entry.epochs || 0,
          converged: Boolean(entry.converged)
        });
      }
      const payload = {
        mode: state.mode,
        bpEpochs: state.bpEpochs,
        backpropTargetBinIndex: state.backpropTargetBinIndex,
        width: state.width,
        height: state.height,
        customCanvasSize: state.customCanvasSize,
        rowCounts: [...state.rowCounts],
        customRowCounts: state.customRowCounts,
        ballsPerWord: state.ballsPerWord,
        learningRate: state.learningRate,
        knobStrength: state.knobStrength,
        knobResistance: state.knobResistance,
        temperature: state.temperature,
        vocabWords: VOCABULARY.map((v) => v.word),
        knobBaseWeights: state.knobs.map((k) => k.baseWeight),
        rules: serializedRules
      };
      state.savedSnapshot = payload;
      try {
        localStorage.setItem(SAVED_STATE_KEY, JSON.stringify(payload));
      } catch (_) {
        // Ignore if localStorage is unavailable
      }
      const summaryMsg = `💾 State saved: ${state.knobs.length} knobs (${state.rowCounts.length} rows), ${VOCABULARY.length} vocab words, ${state.ballsPerWord} balls/word, η=${state.learningRate.toFixed(2)}, Strength=${state.knobStrength.toFixed(1)}×, Resist.=${Math.round(state.knobResistance * 100)}%, T=${state.temperature.toFixed(2)}, and ${state.trainedPrompts.size} ${state.trainedPrompts.size === 1 ? "rule" : "rules"}.`;
      bpFeedbackLog.innerHTML = summaryMsg;
      if (configStateFeedback) {
        configStateFeedback.innerHTML = summaryMsg;
      }
    });
  }

  if (btnLoadState) {
    btnLoadState.addEventListener("click", () => {
      stopAutoTrain();
      let payload = state.savedSnapshot || null;
      if (!payload) {
        try {
          const raw = localStorage.getItem(SAVED_STATE_KEY);
          if (raw) payload = JSON.parse(raw);
        } catch (_) {
          payload = null;
        }
      }
      if (!payload || !Array.isArray(payload.knobBaseWeights)) {
        const warnMsg = `⚠️ No saved state found yet. Click <strong>💾 Save State</strong> first.`;
        bpFeedbackLog.innerHTML = warnMsg;
        if (configStateFeedback) {
          configStateFeedback.innerHTML = warnMsg;
        }
        return;
      }

      // 1. Restore Canvas Dimensions & Row Counts
      if (payload.customCanvasSize && payload.width && payload.height) {
        applyCanvasDimensions(payload.width, payload.height, true);
      }
      if (Array.isArray(payload.rowCounts) && payload.rowCounts.length > 0) {
        state.rowCounts = payload.rowCounts.map((n) => Math.max(2, Math.min(45, Number(n) || 10)));
        state.customRowCounts = Boolean(payload.customRowCounts);
      }

      // 2. Restore Vocabulary
      if (Array.isArray(payload.vocabWords) && payload.vocabWords.length >= 4) {
        VOCABULARY = payload.vocabWords.map((w) => ({
          word: w,
          role: inferWordRole(w)
        }));
        state.bins = VOCABULARY.map((item, index) => ({
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
        }));
        if (vocabCountBadge) {
          vocabCountBadge.textContent = String(VOCABULARY.length);
        }
        initTargetWordSelect();
      }

      layoutBoard(false);

      // 3. Restore Hyperparameters (balls/word, learning rate, strength, resistance, temperature)
      if (typeof payload.ballsPerWord === "number") {
        state.ballsPerWord = payload.ballsPerWord;
        if (ballsPerWordSlider) ballsPerWordSlider.value = String(state.ballsPerWord);
        if (ballsPerWordVal) ballsPerWordVal.textContent = String(state.ballsPerWord);
      }
      if (typeof payload.learningRate === "number") {
        state.learningRate = payload.learningRate;
        if (learningRateSlider) learningRateSlider.value = String(state.learningRate);
        if (learningRateVal) learningRateVal.textContent = state.learningRate.toFixed(2);
      }
      if (typeof payload.knobStrength === "number") {
        state.knobStrength = payload.knobStrength;
        if (knobStrengthSlider) knobStrengthSlider.value = String(state.knobStrength);
        if (knobStrengthVal) knobStrengthVal.textContent = `${state.knobStrength.toFixed(1)}×`;
      }
      if (typeof payload.knobResistance === "number") {
        state.knobResistance = payload.knobResistance;
        if (knobResistanceSlider) knobResistanceSlider.value = String(state.knobResistance);
        if (knobResistanceVal) knobResistanceVal.textContent = `${Math.round(state.knobResistance * 100)}%`;
      }
      if (typeof payload.temperature === "number") {
        state.temperature = payload.temperature;
        if (temperatureSlider) temperatureSlider.value = String(state.temperature);
        if (temperatureVal) temperatureVal.textContent = state.temperature.toFixed(2);
      }

      // 4. Restore Mode & Knob Weights
      state.mode = payload.mode === "random" || payload.mode === "neutral" ? payload.mode : "custom";
      Object.entries(presetBtns).forEach(([key, btn]) => {
        if (btn) btn.classList.toggle("active", key === state.mode);
      });

      state.knobs.forEach((knob, idx) => {
        if (typeof payload.knobBaseWeights[idx] === "number") {
          knob.baseWeight = payload.knobBaseWeights[idx];
          knob.weight = payload.knobBaseWeights[idx];
          knob.lastDelta = 0;
        }
      });

      // 5. Restore Trained Rules
      const restoredPrompts = new Map();
      if (Array.isArray(payload.rules)) {
        for (const r of payload.rules) {
          restoredPrompts.set(r.promptKey, {
            promptKey: r.promptKey,
            targetBinIndex: r.targetBinIndex,
            epochs: r.epochs || 0,
            converged: Boolean(r.converged)
          });
        }
      }
      state.trainedPrompts = restoredPrompts;
      state.bpEpochs = payload.bpEpochs || 0;
      bpEpochCountEl.textContent = String(state.bpEpochs);

      if (
        typeof payload.backpropTargetBinIndex === "number" &&
        payload.backpropTargetBinIndex < VOCABULARY.length
      ) {
        setBackpropTargetBin(payload.backpropTargetBinIndex, false);
      }

      clearBallsAndBins(true);
      renderTokenChips();
      updateMemoryBadgeUI();
      updatePredictionUI();
      syncConfigDialogFields();

      const loadMsg = `📂 State restored: ${state.knobs.length} knobs (${state.rowCounts.length} rows), ${VOCABULARY.length} vocab words, ${state.ballsPerWord} balls/word, η=${state.learningRate.toFixed(2)}, Strength=${state.knobStrength.toFixed(1)}×, Resist.=${Math.round(state.knobResistance * 100)}%, T=${state.temperature.toFixed(2)}, and ${state.trainedPrompts.size} ${state.trainedPrompts.size === 1 ? "rule" : "rules"}.`;
      bpFeedbackLog.innerHTML = loadMsg;
      if (configStateFeedback) {
        configStateFeedback.innerHTML = loadMsg;
      }
    });
  }

  ballsPerWordSlider.addEventListener("input", () => {
    state.ballsPerWord = Number(ballsPerWordSlider.value);
    ballsPerWordVal.textContent = String(state.ballsPerWord);
    renderTokenChips();
  });

  temperatureSlider.addEventListener("input", () => {
    state.temperature = Number(temperatureSlider.value);
    temperatureVal.textContent = state.temperature.toFixed(2);
    updatePredictionUI();
  });

  knobStrengthSlider.addEventListener("input", () => {
    state.knobStrength = Number(knobStrengthSlider.value);
    if (knobStrengthVal) {
      knobStrengthVal.textContent = `${state.knobStrength.toFixed(1)}×`;
    }
  });

  if (knobResistanceSlider) {
    knobResistanceSlider.addEventListener("input", () => {
      state.knobResistance = Number(knobResistanceSlider.value);
      if (knobResistanceVal) {
        knobResistanceVal.textContent = `${Math.round(state.knobResistance * 100)}%`;
      }
    });
  }

  if (simSpeedSlider) {
    simSpeedSlider.addEventListener("input", () => {
      state.simSpeed = Number(simSpeedSlider.value);
      if (simSpeedVal) {
        simSpeedVal.textContent = `${state.simSpeed.toFixed(1)}×`;
      }
    });
  }

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

  if (btnBackpropNudgeOnly) {
    btnBackpropNudgeOnly.addEventListener("click", () => {
      stopAutoTrain();
      performBackpropStep(false);
    });
  }

  function stopAutoTrain(customMessage) {
    if (!state.autoTraining && !customMessage) return;
    state.autoTraining = false;
    state.autoTrainPendingStep = false;
    btnAutoTrain.classList.remove("active-loop");
    btnAutoTrain.textContent = "🔁 Auto-Train";
    if (customMessage) {
      bpFeedbackLog.innerHTML = customMessage;
    }
  }

  btnAutoTrain.addEventListener("click", () => {
    if (state.autoTraining) {
      stopAutoTrain("⏹ Auto-training paused by user.");
      return;
    }

    const currentKey = getNormalizedPromptKey(promptInput.value) || "default";
    if (!state.trainedPrompts.has(currentKey)) {
      performBackpropStep(false);
    }

    for (const entry of state.trainedPrompts.values()) {
      entry.converged = false;
    }
    state.autoTrainPassSuccessCount = 0;

    const keys = Array.from(state.trainedPrompts.keys());
    state.autoTrainIndex = 0;
    if (keys.length > 0) {
      const firstKey = keys[0];
      const firstEntry = state.trainedPrompts.get(firstKey);
      promptInput.value = firstKey;
      if (firstEntry && firstEntry.targetBinIndex < VOCABULARY.length) {
        setBackpropTargetBin(firstEntry.targetBinIndex, false);
      }
      renderTokenChips();
      renderTrainedMemoryStripHtml();
    }

    state.autoTraining = true;
    btnAutoTrain.classList.add("active-loop");
    btnAutoTrain.textContent = "⏹ Stop";
    clearBallsAndBins(true);
    triggerBurstDrop();
  });

  // Output Vocabulary Dialog Management
  const VERB_HINTS = new Set([
    "is", "has", "sat", "ran", "sees", "eats", "goes", "gets",
    "runs", "sleeps", "jumps", "flies", "plays", "reads"
  ]);

  function inferWordRole(w) {
    if (CONNECTORS.has(w)) return "conn";
    if (VERB_HINTS.has(w)) return "verb";
    return "noun";
  }

  function parseVocabInput(rawText) {
    const parts = rawText
      .split(/[,;\n\r\t]+|\s+/)
      .map((s) => s.replace(/[."'!¡?¿()[\]{}]/g, "").trim().toLowerCase())
      .filter(Boolean);

    const unique = [];
    const seen = new Set();
    for (const w of parts) {
      if (!seen.has(w)) {
        seen.add(w);
        unique.push(w);
      }
      if (unique.length >= 40) break;
    }
    return unique;
  }

  function updateVocabDialogPreview() {
    if (!vocabTextarea || !vocabDialogCount || !vocabDialogChips) return;
    const words = parseVocabInput(vocabTextarea.value);
    vocabDialogCount.textContent = `${words.length} ${words.length === 1 ? "word detected" : "words detected"} (min 4 · max 40)`;
    vocabDialogChips.innerHTML = words
      .map((w, idx) => `<span class="vocab-chip-item">#${idx + 1} ${escapeHtml(w)}</span>`)
      .join("");
  }

  function applyVocabularyWords(wordList) {
    if (wordList.length < 4) return false;
    stopAutoTrain();

    VOCABULARY = wordList.map((w) => ({
      word: w,
      role: inferWordRole(w)
    }));

    state.bins = VOCABULARY.map((item, index) => ({
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
    }));

    if (state.backpropTargetBinIndex >= VOCABULARY.length) {
      state.backpropTargetBinIndex = 0;
    }

    if (vocabCountBadge) {
      vocabCountBadge.textContent = String(VOCABULARY.length);
    }

    initTargetWordSelect();
    layoutBoard(true);
    clearBallsAndBins(true);
    renderTokenChips();
    return true;
  }

  if (btnOpenVocab && vocabDialog) {
    btnOpenVocab.addEventListener("click", () => {
      vocabTextarea.value = VOCABULARY.map((v) => v.word).join(", ");
      updateVocabDialogPreview();
      vocabDialog.showModal();
    });

    vocabTextarea.addEventListener("input", updateVocabDialogPreview);

    if (btnResetVocab) {
      btnResetVocab.addEventListener("click", () => {
        vocabTextarea.value = DEFAULT_VOCABULARY.map((v) => v.word).join(", ");
        updateVocabDialogPreview();
      });
    }

    if (btnCancelVocab) {
      btnCancelVocab.addEventListener("click", () => {
        vocabDialog.close();
      });
    }

    if (btnCloseVocabX) {
      btnCloseVocabX.addEventListener("click", () => {
        vocabDialog.close();
      });
    }

    if (vocabForm) {
      vocabForm.addEventListener("submit", (evt) => {
        const words = parseVocabInput(vocabTextarea.value);
        if (words.length < 4) {
          evt.preventDefault();
          vocabDialogCount.textContent = "⚠️ Please enter at least 4 distinct words";
          return;
        }
        applyVocabularyWords(words);
      });
    }

    if (!("closedBy" in HTMLDialogElement.prototype)) {
      vocabDialog.addEventListener("click", (event) => {
        if (event.target !== vocabDialog) return;
        const rect = vocabDialog.getBoundingClientRect();
        const isDialogContent =
          rect.top <= event.clientY &&
          event.clientY <= rect.top + rect.height &&
          rect.left <= event.clientX &&
          event.clientX <= rect.left + rect.width;
        if (isDialogContent) return;
        vocabDialog.close();
      });
    }
  }

  // Training Rules Dialog Management (prompt: target_word)
  const ALL_EXAMPLE_RULES = [
    { prompt: "the cat sat on", target: "the" },
    { prompt: "the cat sat on the", target: "mat" },
    { prompt: "the dog ran to the", target: "man" },
    { prompt: "the sun is", target: "hot" },
    { prompt: "the boy has a red", target: "hat" },
    { prompt: "the girl has a blue", target: "bag" },
    { prompt: "a big fish is in the", target: "bag" }
  ];

  function parseRulesInput(rawText) {
    const lines = rawText.split(/\r?\n/);
    const parsedMap = new Map();

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) continue;

      const match = line.match(/^(.+?)\s*(?::|->|→)\s*(.+)$/);
      if (!match) continue;

      const promptKey = getNormalizedPromptKey(match[1]);
      const targetWord = match[2]
        .replace(/[.,;:!?¿¡"'()[\]{}]/g, "")
        .trim()
        .toLowerCase()
        .split(/\s+/)[0];

      if (!promptKey || !targetWord) continue;
      parsedMap.set(promptKey, { promptKey, targetWord });
    }

    return Array.from(parsedMap.values());
  }

  function formatTrainedPromptsForTextarea() {
    const lines = [];
    for (const [pKey, entry] of state.trainedPrompts.entries()) {
      const targetW = VOCABULARY[entry.targetBinIndex]?.word || "";
      if (pKey && targetW) {
        lines.push(`${pKey}: ${targetW}`);
      }
    }
    return lines.join("\n");
  }

  function updateRulesDialogPreview() {
    if (!rulesTextarea || !rulesDialogCount || !rulesDialogChips) return;
    const rules = parseRulesInput(rulesTextarea.value);
    rulesDialogCount.textContent = `${rules.length} ${rules.length === 1 ? "rule detected" : "rules detected"}`;
    rulesDialogChips.innerHTML = rules
      .map((r) => {
        const inVocab = VOCABULARY.some((v) => v.word === r.targetWord);
        return `<span class="vocab-chip-item">“${escapeHtml(r.promptKey)}” → <strong>${escapeHtml(r.targetWord)}</strong>${inVocab ? "" : " <em>(+vocab)</em>"}</span>`;
      })
      .join("");
  }

  if (bpMemoryBadge && rulesDialog) {
    bpMemoryBadge.addEventListener("click", () => {
      stopAutoTrain();
      rulesTextarea.value = formatTrainedPromptsForTextarea();
      updateRulesDialogPreview();
      rulesDialog.showModal();
    });

    rulesTextarea.addEventListener("input", updateRulesDialogPreview);

    if (btnAddAllExamples) {
      btnAddAllExamples.addEventListener("click", () => {
        const currentRules = parseRulesInput(rulesTextarea.value);
        const mergedMap = new Map();
        for (const r of currentRules) {
          mergedMap.set(r.promptKey, r.targetWord);
        }
        for (const ex of ALL_EXAMPLE_RULES) {
          const pk = getNormalizedPromptKey(ex.prompt);
          if (!mergedMap.has(pk)) {
            mergedMap.set(pk, ex.target);
          }
        }
        const mergedLines = [];
        for (const [pk, tw] of mergedMap.entries()) {
          mergedLines.push(`${pk}: ${tw}`);
        }
        rulesTextarea.value = mergedLines.join("\n");
        updateRulesDialogPreview();
      });
    }

    if (btnClearRules) {
      btnClearRules.addEventListener("click", () => {
        rulesTextarea.value = "";
        updateRulesDialogPreview();
      });
    }

    if (btnCancelRules) {
      btnCancelRules.addEventListener("click", () => {
        rulesDialog.close();
      });
    }

    if (btnCloseRulesX) {
      btnCloseRulesX.addEventListener("click", () => {
        rulesDialog.close();
      });
    }

    if (rulesForm) {
      rulesForm.addEventListener("submit", () => {
        stopAutoTrain();
        const rules = parseRulesInput(rulesTextarea.value);

        let vocabModified = false;
        for (const r of rules) {
          const exists = VOCABULARY.some((v) => v.word === r.targetWord);
          if (!exists && VOCABULARY.length < 40) {
            VOCABULARY.push({
              word: r.targetWord,
              role: inferWordRole(r.targetWord)
            });
            vocabModified = true;
          }
        }

        if (vocabModified) {
          state.bins = VOCABULARY.map((item, index) => ({
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
          }));
          if (vocabCountBadge) {
            vocabCountBadge.textContent = String(VOCABULARY.length);
          }
          initTargetWordSelect();
          layoutBoard(true);
        }

        const nextTrainedPrompts = new Map();
        for (const r of rules) {
          const targetBinIndex = VOCABULARY.findIndex((v) => v.word === r.targetWord);
          if (targetBinIndex < 0) continue;

          const existing = state.trainedPrompts.get(r.promptKey);
          if (existing && existing.targetBinIndex === targetBinIndex) {
            nextTrainedPrompts.set(r.promptKey, existing);
          } else {
            nextTrainedPrompts.set(r.promptKey, {
              promptKey: r.promptKey,
              targetBinIndex,
              epochs: 0,
              converged: false
            });
          }
        }

        state.trainedPrompts = nextTrainedPrompts;
        updateMemoryBadgeUI();

        if (state.trainedPrompts.size > 0) {
          const currentKey = getNormalizedPromptKey(promptInput.value);
          const activeKey = state.trainedPrompts.has(currentKey)
            ? currentKey
            : Array.from(state.trainedPrompts.keys())[0];
          const entry = state.trainedPrompts.get(activeKey);
          promptInput.value = activeKey;
          if (entry) {
            setBackpropTargetBin(entry.targetBinIndex, false);
          }
          renderTokenChips();
          renderTrainedMemoryStripHtml();
          bpFeedbackLog.innerHTML = `✅ ${state.trainedPrompts.size} ${state.trainedPrompts.size === 1 ? "rule loaded" : "rules loaded"} into memory. Click <strong>🔁 Auto-Train</strong> to train them one by one.`;
        } else {
          bpFeedbackLog.innerHTML = "Rule memory cleared. Choose a target word and click “⚡ Backprop”.";
        }
      });
    }

    if (!("closedBy" in HTMLDialogElement.prototype)) {
      rulesDialog.addEventListener("click", (event) => {
        if (event.target !== rulesDialog) return;
        const rect = rulesDialog.getBoundingClientRect();
        const isDialogContent =
          rect.top <= event.clientY &&
          event.clientY <= rect.top + rect.height &&
          rect.left <= event.clientX &&
          event.clientX <= rect.left + rect.width;
        if (isDialogContent) return;
        rulesDialog.close();
      });
    }
  }

  const resizeObserver = new ResizeObserver((entries) => {
    if (state.customCanvasSize) return;
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

      if (!state.customRowCounts) {
        const nextRowCounts = computeRecommendedRowCounts(state.width, state.height);
        const changed = nextRowCounts.join(",") !== state.rowCounts.join(",");
        state.rowCounts = nextRowCounts;
        layoutBoard(state.knobs.length > 0 && !changed);
      } else {
        layoutBoard(state.knobs.length > 0);
      }
      syncConfigDialogFields();
    }
  });
  if (canvasWrapper) {
    resizeObserver.observe(canvasWrapper);
  }

  // Paso de física fijo: la simulación es determinista e independiente de los FPS del navegador.
  const FIXED_PHYSICS_DT = 1 / 60;
  const MAX_PHYSICS_STEPS_PER_FRAME = 24;
  let physicsAccumulator = 0;
  let lastTime = performance.now();
  function animate(now) {
    const rawDt = Math.min(0.035, (now - lastTime) / 1000);
    lastTime = now;

    physicsAccumulator += rawDt * state.simSpeed;
    let steps = 0;
    while (physicsAccumulator >= FIXED_PHYSICS_DT && steps < MAX_PHYSICS_STEPS_PER_FRAME) {
      stepPhysics(FIXED_PHYSICS_DT);
      physicsAccumulator -= FIXED_PHYSICS_DT;
      steps++;
    }
    if (steps >= MAX_PHYSICS_STEPS_PER_FRAME) {
      physicsAccumulator = 0; // descartar tiempo atrasado (ralentiza, pero no altera el resultado)
    }
    drawBoard();

    requestAnimationFrame(animate);
  }

  initTargetWordSelect();
  renderTokenChips();
  layoutBoard(false);
  syncConfigDialogFields();
  requestAnimationFrame((t) => {
    lastTime = t;
    requestAnimationFrame(animate);
  });
})();
