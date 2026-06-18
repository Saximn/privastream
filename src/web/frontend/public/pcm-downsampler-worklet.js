// AudioWorklet processor: downmix to mono, anti-aliased downsample to 16 kHz,
// and emit 16-bit PCM to the main thread.
//
// Replaces the deprecated ScriptProcessorNode path. Runs on the audio render
// thread (no main-thread jank). The old code took "every 3rd sample" with no
// anti-alias filter, which folds everything above 8 kHz back into the band and
// hurts Whisper transcription (and therefore PII recall). Here we low-pass with
// a windowed-sinc FIR before decimating.
//
// processorOptions: { targetSampleRate?: number }  (default 16000)

class PCMDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    const target = opts.targetSampleRate || 16000;

    // Integer decimation factor (e.g. 48000 -> 16000 = 3). `sampleRate` is a
    // global available inside AudioWorkletGlobalScope.
    this.ratio = Math.max(1, Math.round(sampleRate / target));

    // Windowed-sinc low-pass FIR, cutoff at the post-decimation Nyquist.
    this.taps = PCMDownsampler.buildLowpass(31, 0.5 / this.ratio);

    // Ring buffer of recent mono input for the FIR convolution.
    this.history = new Float32Array(this.taps.length);
    this.histPos = 0;
    this.phase = 0; // counts input samples toward the next decimated output

    // Batch output to ~100 ms before posting, to keep message rate low.
    this.flushSize = Math.max(1, Math.round(target * 0.1));
    this.outBuf = new Float32Array(this.flushSize);
    this.outLen = 0;
  }

  // Symmetric (linear-phase) low-pass via sinc * Hamming window, DC-normalized.
  static buildLowpass(numTaps, fc) {
    const taps = new Float32Array(numTaps);
    const mid = (numTaps - 1) / 2;
    let sum = 0;
    for (let i = 0; i < numTaps; i++) {
      const n = i - mid;
      const sinc = n === 0 ? 2 * Math.PI * fc : Math.sin(2 * Math.PI * fc * n) / n;
      const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (numTaps - 1));
      taps[i] = sinc * window;
      sum += taps[i];
    }
    for (let i = 0; i < numTaps; i++) taps[i] /= sum;
    return taps;
  }

  flush() {
    if (this.outLen === 0) return;
    const pcm = new Int16Array(this.outLen);
    for (let i = 0; i < this.outLen; i++) {
      const s = Math.max(-1, Math.min(1, this.outBuf[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    // Transfer the buffer to avoid a copy.
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
    this.outLen = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true; // keep the node alive

    const channels = input.length;
    const frames = input[0].length;

    for (let i = 0; i < frames; i++) {
      // Downmix to mono.
      let mono = 0;
      for (let c = 0; c < channels; c++) mono += input[c][i];
      mono /= channels;

      // Feed the FIR history ring.
      this.history[this.histPos] = mono;
      this.histPos = (this.histPos + 1) % this.history.length;

      // Emit one filtered sample every `ratio` input samples.
      if (++this.phase >= this.ratio) {
        this.phase = 0;
        let acc = 0;
        let idx = this.histPos - 1;
        for (let k = 0; k < this.taps.length; k++) {
          if (idx < 0) idx += this.history.length;
          acc += this.taps[k] * this.history[idx];
          idx--;
        }
        this.outBuf[this.outLen++] = acc;
        if (this.outLen >= this.flushSize) this.flush();
      }
    }

    return true; // outputs left silent — no feedback
  }
}

registerProcessor("pcm-downsampler", PCMDownsampler);
