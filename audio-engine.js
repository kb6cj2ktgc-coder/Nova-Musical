// NOVA MUSICAL
// Audio Engine v0.1
// Listens through the microphone and detects musical pitch locally.
// No external AI API is used.

class NovaAudioEngine {

  constructor() {
    this.audioContext = null;
    this.analyser = null;
    this.microphone = null;
    this.buffer = null;
    this.running = false;
  }

  async start() {

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    this.audioContext =
      new (window.AudioContext || window.webkitAudioContext)();

    await this.audioContext.resume();

    this.microphone =
      this.audioContext.createMediaStreamSource(stream);

    this.analyser =
      this.audioContext.createAnalyser();

    this.analyser.fftSize = 4096;

    this.buffer =
      new Float32Array(this.analyser.fftSize);

    this.microphone.connect(this.analyser);

    this.running = true;

    return true;
  }


  detectPitch() {

    if (!this.running) {
      return null;
    }

    this.analyser.getFloatTimeDomainData(this.buffer);

    const frequency = this.autoCorrelate(
      this.buffer,
      this.audioContext.sampleRate
    );

    if (frequency === -1) {
      return null;
    }

    return this.analyseFrequency(frequency);
  }


  analyseFrequency(frequency) {

    const noteNumber =
      Math.round(
        12 * Math.log2(frequency / 440)
      ) + 69;

    const noteNames = [
      "C", "C♯", "D", "D♯",
      "E", "F", "F♯", "G",
      "G♯", "A", "A♯", "B"
    ];

    const note =
      noteNames[
        ((noteNumber % 12) + 12) % 12
      ];

    const octave =
      Math.floor(noteNumber / 12) - 1;

    const perfectFrequency =
      440 *
      Math.pow(
        2,
        (noteNumber - 69) / 12
      );

    const cents =
      1200 *
      Math.log2(
        frequency / perfectFrequency
      );

    let tuning;

    if (Math.abs(cents) <= 5) {
      tuning = "IN TUNE";
    }

    else if (cents < 0) {
      tuning = "FLAT";
    }

    else {
      tuning = "SHARP";
    }


    return {

      frequency:
        Math.round(frequency * 10) / 10,

      note,

      octave,

      cents:
        Math.round(cents),

      tuning

    };

  }


  autoCorrelate(buffer, sampleRate) {

    let rms = 0;

    for (let i = 0; i < buffer.length; i++) {
      rms += buffer[i] * buffer[i];
    }

    rms =
      Math.sqrt(rms / buffer.length);

    // Ignore very quiet background noise
    if (rms < 0.01) {
      return -1;
    }


    let bestOffset = -1;
    let bestCorrelation = 0;

    const minFrequency = 55;
    const maxFrequency = 1200;

    const minOffset =
      Math.floor(sampleRate / maxFrequency);

    const maxOffset =
      Math.floor(sampleRate / minFrequency);


    for (
      let offset = minOffset;
      offset <= maxOffset;
      offset++
    ) {

      let correlation = 0;

      for (
        let i = 0;
        i < buffer.length - offset;
        i++
      ) {

        correlation +=
          buffer[i] *
          buffer[i + offset];

      }


      if (correlation > bestCorrelation) {

        bestCorrelation =
          correlation;

        bestOffset =
          offset;

      }

    }


    if (bestOffset === -1) {
      return -1;
    }


    return sampleRate / bestOffset;

  }


  stop() {

    if (this.audioContext) {
      this.audioContext.close();
    }

    this.running = false;

  }

}


// Make Nova's audio engine available to the website.

window.NovaAudioEngine =
  NovaAudioEngine;
