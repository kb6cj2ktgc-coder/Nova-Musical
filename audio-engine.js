// ============================================================
// NOVA MUSICAL — GUITAR AUDIO ENGINE v0.2
// Local browser pitch detection.
// No external AI/audio API.
// ============================================================

class NovaAudioEngine {

  constructor() {
    this.audioContext = null;
    this.analyser = null;
    this.microphone = null;
    this.stream = null;
    this.buffer = null;
    this.running = false;

    this.pitchHistory = [];
  }


  async start() {

    this.stream =
      await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });

    this.audioContext =
      new (
        window.AudioContext ||
        window.webkitAudioContext
      )();

    await this.audioContext.resume();

    this.microphone =
      this.audioContext.createMediaStreamSource(
        this.stream
      );

    this.analyser =
      this.audioContext.createAnalyser();

    this.analyser.fftSize = 4096;

    this.buffer =
      new Float32Array(
        this.analyser.fftSize
      );

    this.microphone.connect(
      this.analyser
    );

    this.running = true;

    return true;
  }


  stop() {

    this.running = false;

    if (this.stream) {

      this.stream
        .getTracks()
        .forEach(
          track => track.stop()
        );
    }

    if (this.audioContext) {
      this.audioContext.close();
    }

    this.pitchHistory = [];
  }


  detectPitch() {

    if (
      !this.running ||
      !this.analyser
    ) {
      return null;
    }

    this.analyser.getFloatTimeDomainData(
      this.buffer
    );

    const frequency =
      this.autoCorrelate(
        this.buffer,
        this.audioContext.sampleRate
      );

    if (
      !frequency ||
      frequency < 65 ||
      frequency > 380
    ) {
      return null;
    }

    const stableFrequency =
      this.stabilize(
        frequency
      );

    return {
      frequency:
        Math.round(
          stableFrequency * 100
        ) / 100
    };
  }


  stabilize(frequency) {

    this.pitchHistory.push(
      frequency
    );

    if (
      this.pitchHistory.length > 5
    ) {
      this.pitchHistory.shift();
    }

    const sorted =
      [...this.pitchHistory]
        .sort(
          (a, b) => a - b
        );

    return sorted[
      Math.floor(
        sorted.length / 2
      )
    ];
  }


  autoCorrelate(
    buffer,
    sampleRate
  ) {

    const size =
      buffer.length;

    let rms = 0;

    for (
      let i = 0;
      i < size;
      i++
    ) {

      rms +=
        buffer[i] *
        buffer[i];
    }

    rms =
      Math.sqrt(
        rms / size
      );

    if (rms < 0.012) {
      return -1;
    }


    // Remove DC offset.

    let mean = 0;

    for (
      let i = 0;
      i < size;
      i++
    ) {
      mean += buffer[i];
    }

    mean /= size;


    const clean =
      new Float32Array(
        size
      );

    for (
      let i = 0;
      i < size;
      i++
    ) {
      clean[i] =
        buffer[i] - mean;
    }


    // Guitar range.
    // Search periods corresponding
    // approximately to 65–380 Hz.

    const minLag =
      Math.floor(
        sampleRate / 380
      );

    const maxLag =
      Math.min(
        Math.floor(
          sampleRate / 65
        ),
        size - 2
      );


    let bestLag = -1;
    let bestScore = -Infinity;


    for (
      let lag = minLag;
      lag <= maxLag;
      lag++
    ) {

      let numerator = 0;
      let energyA = 0;
      let energyB = 0;


      for (
        let i = 0;
        i < size - lag;
        i++
      ) {

        const a =
          clean[i];

        const b =
          clean[i + lag];

        numerator +=
          a * b;

        energyA +=
          a * a;

        energyB +=
          b * b;
      }


      const denominator =
        Math.sqrt(
          energyA *
          energyB
        );


      if (
        denominator === 0
      ) {
        continue;
      }


      const score =
        numerator /
        denominator;


      if (
        score >
        bestScore
      ) {

        bestScore =
          score;

        bestLag =
          lag;
      }
    }


    if (
      bestLag < 0 ||
      bestScore < 0.65
    ) {
      return -1;
    }


    /*
    Small interpolation around
    the winning lag.
    */

    const correlationAt =
      lag => {

        if (
          lag < minLag ||
          lag > maxLag
        ) {
          return bestScore;
        }

        let numerator = 0;
        let energyA = 0;
        let energyB = 0;

        for (
          let i = 0;
          i < size - lag;
          i++
        ) {

          const a =
            clean[i];

          const b =
            clean[i + lag];

          numerator +=
            a * b;

          energyA +=
            a * a;

          energyB +=
            b * b;
        }

        const denominator =
          Math.sqrt(
            energyA *
            energyB
          );

        return denominator
          ? numerator / denominator
          : 0;
      };


    const left =
      correlationAt(
        bestLag - 1
      );

    const centre =
      bestScore;

    const right =
      correlationAt(
        bestLag + 1
      );


    const divisor =
      (
        left -
        2 * centre +
        right
      );


    let refinedLag =
      bestLag;


    if (
      Math.abs(divisor) >
      0.000001
    ) {

      refinedLag =
        bestLag +
        0.5 *
        (
          left - right
        ) /
        divisor;
    }


    return (
      sampleRate /
      refinedLag
    );
  }

}


window.NovaAudioEngine =
  NovaAudioEngine;
