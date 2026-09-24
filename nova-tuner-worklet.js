/*
=========================================================
NOVA MUSICAL
Precision Guitar Tuner AudioWorklet Engine

File:
nova-tuner-worklet.js

Purpose:
- continuous microphone processing
- fast response to a new pluck
- fundamental pitch detection
- harmonic-resistant detection
- smooth, accurate frequency estimates
- sends clean pitch data back to guitar.html
=========================================================
*/

class NovaTunerProcessor extends AudioWorkletProcessor {

  constructor() {
    super();

    /*
    -----------------------------------------------------
    RING BUFFER

    We continuously collect microphone samples here.

    16384 samples gives enough information for Low E
    while still allowing the tuner to respond quickly.
    -----------------------------------------------------
    */

    this.BUFFER_SIZE = 16384;

    this.buffer = new Float32Array(this.BUFFER_SIZE);

    this.writePosition = 0;

    this.totalWritten = 0;


    /*
    -----------------------------------------------------
    ANALYSIS TIMING

    AudioWorklet normally receives 128 samples at a time.

    We do NOT wait several seconds before analysing.

    Once enough audio exists, analyse frequently.
    -----------------------------------------------------
    */

    this.samplesSinceAnalysis = 0;

    /*
    Around 2048 new samples between analyses.

    At 48 kHz this is ~43 ms.
    */

    this.analysisHop = 2048;


    /*
    -----------------------------------------------------
    SIGNAL STATE
    -----------------------------------------------------
    */

    this.previousRMS = 0;

    this.noiseFloor = 0.0005;

    this.onsetCooldown = 0;

    this.lastPitch = 0;

    this.lastClarity = 0;

    this.pitchHistory = [];


    /*
    -----------------------------------------------------
    GUITAR RANGE

    Standard guitar:
    Low E = 82.4069 Hz
    High E = 329.6276 Hz

    Wider range lets us identify slightly unusual
    readings without octave-folding them.
    -----------------------------------------------------
    */

    this.MIN_FREQUENCY = 65;
    this.MAX_FREQUENCY = 420;


    /*
    Tell the page that the engine loaded correctly.
    */

    this.port.postMessage({
      type: "ready"
    });
  }


  /*
  =======================================================
  MAIN AUDIO CALLBACK
  =======================================================
  */

  process(inputs) {

    const input = inputs[0];

    if (!input || !input[0]) {
      return true;
    }

    const channel = input[0];


    /*
    -----------------------------------------------------
    WRITE AUDIO INTO CIRCULAR BUFFER
    -----------------------------------------------------
    */

    for (let i = 0; i < channel.length; i++) {

      this.buffer[this.writePosition] = channel[i];

      this.writePosition++;

      if (this.writePosition >= this.BUFFER_SIZE) {
        this.writePosition = 0;
      }

      this.totalWritten++;
    }


    this.samplesSinceAnalysis += channel.length;


    /*
    -----------------------------------------------------
    QUICK LEVEL / ONSET ANALYSIS

    This lets Nova notice that the user has plucked a
    DIFFERENT string instead of remaining stuck on the
    previous Low E.
    -----------------------------------------------------
    */

    const rms = this.calculateRMS(channel);


    /*
    Slowly learn background microphone noise.
    */

    if (rms < 0.02) {

      this.noiseFloor =
        this.noiseFloor * 0.995 +
        rms * 0.005;
    }


    const safeNoise =
      Math.max(
        0.0004,
        this.noiseFloor
      );


    /*
    A rapid energy increase usually means a fresh pluck.
    */

    const onset =
      rms > safeNoise * 3.0 &&
      rms > this.previousRMS * 1.65 &&
      rms > 0.004;


    this.previousRMS =
      this.previousRMS * 0.72 +
      rms * 0.28;


    if (this.onsetCooldown > 0) {
      this.onsetCooldown -= channel.length;
    }


    if (onset && this.onsetCooldown <= 0) {

      /*
      NEW PLUCK.

      Throw away old pitch history immediately.

      This is specifically intended to prevent:

      Low E
      ↓
      user plays G
      ↓
      tuner remains stuck on Low E.
      */

      this.pitchHistory = [];

      this.lastPitch = 0;

      this.lastClarity = 0;

      this.onsetCooldown =
        sampleRate * 0.08;


      this.port.postMessage({
        type: "onset"
      });
    }


    /*
    -----------------------------------------------------
    RUN FULL PITCH ANALYSIS
    -----------------------------------------------------
    */

    if (
      this.totalWritten >= this.BUFFER_SIZE &&
      this.samplesSinceAnalysis >= this.analysisHop
    ) {

      this.samplesSinceAnalysis = 0;

      this.analyse();
    }


    return true;
  }


  /*
  =======================================================
  FULL ANALYSIS
  =======================================================
  */

  analyse() {

    const signal =
      this.getOrderedBuffer();


    /*
    -----------------------------------------------------
    SIGNAL LEVEL
    -----------------------------------------------------
    */

    const rms =
      this.calculateRMS(signal);


    /*
    Ignore genuine silence.
    */

    const threshold =
      Math.max(
        0.0018,
        this.noiseFloor * 2.2
      );


    if (rms < threshold) {

      this.port.postMessage({
        type: "silence",
        level: rms
      });

      return;
    }


    /*
    -----------------------------------------------------
    REMOVE DC OFFSET
    -----------------------------------------------------
    */

    this.removeDC(signal);


    /*
    -----------------------------------------------------
    APPLY GENTLE WINDOW

    Reduces edge discontinuities without destroying
    the periodic information needed by the detector.
    -----------------------------------------------------
    */

    this.applyHann(signal);


    /*
    -----------------------------------------------------
    MULTI-DETECTOR STRATEGY

    1. McLeod/NSDF-style detector
    2. YIN detector

    If both agree closely, confidence becomes very high.

    If only one is reliable, Nova can still use it.
    -----------------------------------------------------
    */

    const mpm =
      this.detectMPM(signal);


    const yin =
      this.detectYIN(signal);


    const combined =
      this.combineDetectors(
        mpm,
        yin
      );


    if (!combined) {
      return;
    }


    /*
    -----------------------------------------------------
    TEMPORAL VALIDATION

    This rejects one-frame crazy readings such as:

    -9
    -9
    +26
    -100
    -9

    without making the tuner take several seconds to
    respond to a genuinely different string.
    -----------------------------------------------------
    */

    const stable =
      this.stabilisePitch(
        combined.frequency,
        combined.confidence
      );


    if (!stable) {
      return;
    }


    this.lastPitch =
      stable.frequency;

    this.lastClarity =
      stable.confidence;


    /*
    Send the ACTUAL detected frequency.

    guitar.html will decide whether it is E/A/D/G/B/E.

    The worklet never forces the sound toward a
    selected string.
    */

    this.port.postMessage({

      type: "pitch",

      frequency:
        stable.frequency,

      confidence:
        stable.confidence,

      level:
        rms

    });
  }


  /*
  =======================================================
  GET CHRONOLOGICAL AUDIO FROM CIRCULAR BUFFER
  =======================================================
  */

  getOrderedBuffer() {

    const result =
      new Float32Array(
        this.BUFFER_SIZE
      );


    const firstLength =
      this.BUFFER_SIZE -
      this.writePosition;


    result.set(
      this.buffer.subarray(
        this.writePosition
      ),
      0
    );


    result.set(
      this.buffer.subarray(
        0,
        this.writePosition
      ),
      firstLength
    );


    return result;
  }


  /*
  =======================================================
  RMS
  =======================================================
  */

  calculateRMS(data) {

    let sum = 0;


    for (
      let i = 0;
      i < data.length;
      i++
    ) {

      const value =
        data[i];

      sum +=
        value * value;
    }


    return Math.sqrt(
      sum / data.length
    );
  }


  /*
  =======================================================
  REMOVE DC OFFSET
  =======================================================
  */

  removeDC(data) {

    let mean = 0;


    for (
      let i = 0;
      i < data.length;
      i++
    ) {

      mean += data[i];
    }


    mean /=
      data.length;


    for (
      let i = 0;
      i < data.length;
      i++
    ) {

      data[i] -= mean;
    }
  }


  /*
  =======================================================
  HANN WINDOW
  =======================================================
  */

  applyHann(data) {

    const length =
      data.length;


    for (
      let i = 0;
      i < length;
      i++
    ) {

      const window =
        0.5 *
        (
          1 -
          Math.cos(
            (2 * Math.PI * i) /
            (length - 1)
          )
        );


      data[i] *= window;
    }
  }


  /*
  =======================================================
  McLEOD / NSDF STYLE DETECTOR
  =======================================================
  */

  detectMPM(data) {

    const minLag =
      Math.floor(
        sampleRate /
        this.MAX_FREQUENCY
      );


    const maxLag =
      Math.ceil(
        sampleRate /
        this.MIN_FREQUENCY
      );


    /*
    Use enough samples for Low E,
    but avoid doing unnecessary work.
    */

    const analysisLength =
      Math.min(
        8192,
        data.length -
        maxLag -
        1
      );


    if (analysisLength < 4096) {
      return null;
    }


    const nsdf =
      new Float64Array(
        maxLag + 2
      );


    /*
    -----------------------------------------------------
    NORMALIZED SQUARE DIFFERENCE FUNCTION
    -----------------------------------------------------
    */

    for (
      let lag = minLag;
      lag <= maxLag;
      lag++
    ) {

      let correlation = 0;

      let energy = 0;


      /*
      Step by 2 for mobile performance.

      Fractional-lag interpolation below restores
      precision beyond whole-sample resolution.
      */

      for (
        let i = 0;
        i < analysisLength;
        i += 2
      ) {

        const a =
          data[i];

        const b =
          data[i + lag];


        correlation +=
          a * b;


        energy +=
          a * a +
          b * b;
      }


      nsdf[lag] =
        energy > 0
        ? (2 * correlation) / energy
        : 0;
    }


    /*
    -----------------------------------------------------
    FIND LOCAL MAXIMA
    -----------------------------------------------------
    */

    const peaks = [];


    for (
      let lag = minLag + 1;
      lag < maxLag;
      lag++
    ) {

      if (
        nsdf[lag] > 0 &&
        nsdf[lag] >= nsdf[lag - 1] &&
        nsdf[lag] > nsdf[lag + 1]
      ) {

        peaks.push({

          lag: lag,

          clarity:
            nsdf[lag]

        });
      }
    }


    if (!peaks.length) {
      return null;
    }


    /*
    -----------------------------------------------------
    FIND STRONGEST PEAK
    -----------------------------------------------------
    */

    let strongest =
      peaks[0];


    for (const peak of peaks) {

      if (
        peak.clarity >
        strongest.clarity
      ) {

        strongest = peak;
      }
    }


    if (
      strongest.clarity < 0.72
    ) {

      return null;
    }


    /*
    -----------------------------------------------------
    FUNDAMENTAL PEAK SELECTION

    Pick the earliest peak that is almost as convincing
    as the strongest periodicity.

    This helps avoid later period multiples while still
    rejecting weak high-frequency harmonics.
    -----------------------------------------------------
    */

    const cutoff =
      strongest.clarity * 0.92;


    let chosen = null;


    for (const peak of peaks) {

      if (
        peak.clarity >= cutoff
      ) {

        chosen = peak;

        break;
      }
    }


    if (!chosen) {

      chosen = strongest;
    }


    /*
    -----------------------------------------------------
    PARABOLIC INTERPOLATION
    -----------------------------------------------------
    */

    const lag =
      chosen.lag;


    let refinedLag =
      lag;


    if (
      lag > minLag &&
      lag < maxLag
    ) {

      const left =
        nsdf[lag - 1];

      const center =
        nsdf[lag];

      const right =
        nsdf[lag + 1];


      const denominator =
        left -
        2 * center +
        right;


      if (
        Math.abs(denominator) >
        0.0000001
      ) {

        const shift =
          0.5 *
          (left - right) /
          denominator;


        if (
          Number.isFinite(shift) &&
          Math.abs(shift) <= 1
        ) {

          refinedLag +=
            shift;
        }
      }
    }


    const frequency =
      sampleRate /
      refinedLag;


    if (
      !Number.isFinite(frequency) ||
      frequency <
        this.MIN_FREQUENCY ||
      frequency >
        this.MAX_FREQUENCY
    ) {

      return null;
    }


    return {

      frequency:
        frequency,

      confidence:
        Math.max(
          0,
          Math.min(
            1,
            chosen.clarity
          )
        )

    };
  }


  /*
  =======================================================
  YIN DETECTOR
  =======================================================
  */

  detectYIN(data) {

    const minLag =
      Math.floor(
        sampleRate /
        this.MAX_FREQUENCY
      );


    const maxLag =
      Math.ceil(
        sampleRate /
        this.MIN_FREQUENCY
      );


    const length =
      Math.min(
        8192,
        data.length -
        maxLag -
        1
      );


    if (length < 4096) {
      return null;
    }


    const difference =
      new Float64Array(
        maxLag + 2
      );


    /*
    -----------------------------------------------------
    DIFFERENCE FUNCTION
    -----------------------------------------------------
    */

    for (
      let lag = 1;
      lag <= maxLag;
      lag++
    ) {

      let sum = 0;


      for (
        let i = 0;
        i < length;
        i += 2
      ) {

        const delta =
          data[i] -
          data[i + lag];


        sum +=
          delta * delta;
      }


      difference[lag] =
        sum;
    }


    /*
    -----------------------------------------------------
    CUMULATIVE MEAN NORMALIZED DIFFERENCE
    -----------------------------------------------------
    */

    const cmnd =
      new Float64Array(
        maxLag + 2
      );


    cmnd[0] = 1;


    let runningSum = 0;


    for (
      let lag = 1;
      lag <= maxLag;
      lag++
    ) {

      runningSum +=
        difference[lag];


      cmnd[lag] =
        runningSum > 0
        ? (
            difference[lag] *
            lag
          ) /
          runningSum
        : 1;
    }


    /*
    -----------------------------------------------------
    FIRST RELIABLE VALLEY
    -----------------------------------------------------
    */

    let chosenLag = -1;


    const threshold =
      0.12;


    for (
      let lag = minLag;
      lag <= maxLag;
      lag++
    ) {

      if (
        cmnd[lag] <
        threshold
      ) {

        while (
          lag + 1 <= maxLag &&
          cmnd[lag + 1] <
          cmnd[lag]
        ) {

          lag++;
        }


        chosenLag =
          lag;

        break;
      }
    }


    /*
    -----------------------------------------------------
    FALLBACK DEEPEST VALLEY
    -----------------------------------------------------
    */

    if (chosenLag < 0) {

      let bestValue = 1;


      for (
        let lag = minLag;
        lag <= maxLag;
        lag++
      ) {

        if (
          cmnd[lag] <
          bestValue
        ) {

          bestValue =
            cmnd[lag];

          chosenLag =
            lag;
        }
      }


      if (
        chosenLag < 0 ||
        bestValue > 0.24
      ) {

        return null;
      }
    }


    /*
    -----------------------------------------------------
    FRACTIONAL LAG
    -----------------------------------------------------
    */

    let refinedLag =
      chosenLag;


    if (
      chosenLag > minLag &&
      chosenLag < maxLag
    ) {

      const left =
        cmnd[
          chosenLag - 1
        ];

      const center =
        cmnd[
          chosenLag
        ];

      const right =
        cmnd[
          chosenLag + 1
        ];


      const denominator =
        left -
        2 * center +
        right;


      if (
        Math.abs(denominator) >
        0.0000001
      ) {

        const shift =
          0.5 *
          (left - right) /
          denominator;


        if (
          Number.isFinite(shift) &&
          Math.abs(shift) <= 1
        ) {

          refinedLag +=
            shift;
        }
      }
    }


    const frequency =
      sampleRate /
      refinedLag;


    const confidence =
      1 -
      cmnd[chosenLag];


    if (
      !Number.isFinite(frequency) ||
      frequency <
        this.MIN_FREQUENCY ||
      frequency >
        this.MAX_FREQUENCY
    ) {

      return null;
    }


    return {

      frequency:
        frequency,

      confidence:
        Math.max(
          0,
          Math.min(
            1,
            confidence
          )
        )

    };
  }


  /*
  =======================================================
  COMBINE DETECTORS
  =======================================================
  */

  combineDetectors(
    mpm,
    yin
  ) {

    if (!mpm && !yin) {
      return null;
    }


    if (mpm && !yin) {

      if (
        mpm.confidence < 0.86
      ) {

        return null;
      }


      return mpm;
    }


    if (yin && !mpm) {

      if (
        yin.confidence < 0.86
      ) {

        return null;
      }


      return yin;
    }


    /*
    Both detectors produced a pitch.

    Compare them musically in cents.
    */

    const difference =
      Math.abs(
        1200 *
        Math.log2(
          mpm.frequency /
          yin.frequency
        )
      );


    /*
    Excellent agreement.
    */

    if (difference <= 5) {

      const mWeight =
        mpm.confidence *
        mpm.confidence;


      const yWeight =
        yin.confidence *
        yin.confidence;


      const frequency =
        (
          mpm.frequency *
          mWeight +
          yin.frequency *
          yWeight
        ) /
        (
          mWeight +
          yWeight
        );


      return {

        frequency:
          frequency,

        confidence:
          Math.min(
            1,
            (
              mpm.confidence +
              yin.confidence
            ) / 2 +
            0.04
          )

      };
    }


    /*
    Mild disagreement.

    Trust the clearly stronger detector.
    */

    if (difference <= 18) {

      if (
        mpm.confidence >
        yin.confidence + 0.08
      ) {

        return mpm;
      }


      if (
        yin.confidence >
        mpm.confidence + 0.08
      ) {

        return yin;
      }


      /*
      Neither one clearly wins.

      Don't show the player a fake answer.
      */

      return null;
    }


    /*
    Large disagreement usually means a harmonic,
    attack transient or noisy measurement.

    Reject it.
    */

    return null;
  }


  /*
  =======================================================
  SHORT STABILITY FILTER
  =======================================================
  */

  stabilisePitch(
    frequency,
    confidence
  ) {

    const nowPitch = {

      frequency:
        frequency,

      confidence:
        confidence

    };


    /*
    -----------------------------------------------------
    FAST STRING-CHANGE DETECTION

    If the new pitch is dramatically different from
    the recent pitch, assume the player may have
    plucked another string.

    Clear the old history immediately.
    -----------------------------------------------------
    */

    if (
      this.pitchHistory.length > 0
    ) {

      const previous =
        this.pitchHistory[
          this.pitchHistory.length - 1
        ].frequency;


      const jump =
        Math.abs(
          1200 *
          Math.log2(
            frequency /
            previous
          )
        );


      /*
      170 cents is far beyond normal tuning movement.

      This means it is almost certainly a new note/string.
      */

      if (jump > 170) {

        this.pitchHistory = [
          nowPitch
        ];


        /*
        Send quickly after a genuine string change
        rather than waiting several seconds.
        */

        return nowPitch;
      }
    }


    this.pitchHistory.push(
      nowPitch
    );


    /*
    Very short history.

    This is deliberate.

    Long histories made Nova get stuck on Low E.
    */

    if (
      this.pitchHistory.length > 4
    ) {

      this.pitchHistory.shift();
    }


    if (
      this.pitchHistory.length === 1
    ) {

      return nowPitch;
    }


    /*
    -----------------------------------------------------
    MEDIAN REFERENCE
    -----------------------------------------------------
    */

    const frequencies =
      this.pitchHistory
      .map(
        item =>
          item.frequency
      )
      .sort(
        (a,b)=>a-b
      );


    const median =
      frequencies[
        Math.floor(
          frequencies.length / 2
        )
      ];


    /*
    -----------------------------------------------------
    REMOVE OUTLIERS

    A reading more than 8 cents from the local median
    is not allowed to pull the displayed pitch around.
    -----------------------------------------------------
    */

    const accepted =
      this.pitchHistory
      .filter(item => {

        const cents =
          Math.abs(
            1200 *
            Math.log2(
              item.frequency /
              median
            )
          );


        return cents <= 8;
      });


    if (
      accepted.length < 2
    ) {

      return null;
    }


    /*
    -----------------------------------------------------
    CONFIDENCE-WEIGHTED GEOMETRIC MEAN

    Frequency is logarithmic musically.

    Averaging log-frequency therefore behaves better
    for pitch than an ordinary arithmetic average.
    -----------------------------------------------------
    */

    let weightedLog = 0;

    let totalWeight = 0;

    let confidenceSum = 0;


    for (
      const item of accepted
    ) {

      const weight =
        item.confidence *
        item.confidence;


      weightedLog +=
        Math.log(
          item.frequency
        ) *
        weight;


      totalWeight +=
        weight;


      confidenceSum +=
        item.confidence;
    }


    if (
      totalWeight <= 0
    ) {

      return null;
    }


    const stableFrequency =
      Math.exp(
        weightedLog /
        totalWeight
      );


    return {

      frequency:
        stableFrequency,

      confidence:
        confidenceSum /
        accepted.length

    };
  }

}


/*
=========================================================
REGISTER WORKLET
=========================================================
*/

registerProcessor(
  "nova-tuner-processor",
  NovaTunerProcessor
);
