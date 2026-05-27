/**
 * AudioWorkletProcessor — runs on the audio thread.
 * Receives Float32 PCM from the mic, converts to Int16 LE,
 * and posts 100ms chunks to the main thread via MessagePort.
 *
 * Registered as "mic-processor" in app.js.
 */

const SAMPLE_RATE   = 16000;
const CHUNK_MS      = 100;
const CHUNK_SAMPLES = (SAMPLE_RATE * CHUNK_MS) / 1000;  // 1600

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(CHUNK_SAMPLES);
    this._filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0];  // Float32, 128 samples per call at 48kHz (resampled)

    for (let i = 0; i < samples.length; i++) {
      this._buffer[this._filled++] = samples[i];

      if (this._filled >= CHUNK_SAMPLES) {
        // Convert Float32 [-1, 1] → Int16 [-32768, 32767]
        const int16 = new Int16Array(CHUNK_SAMPLES);
        for (let j = 0; j < CHUNK_SAMPLES; j++) {
          const clamped = Math.max(-1, Math.min(1, this._buffer[j]));
          int16[j] = clamped < 0 ? clamped * 32768 : clamped * 32767;
        }

        // Transfer ownership to avoid copy
        this.port.postMessage({ type: "chunk", buffer: int16.buffer }, [int16.buffer]);
        this._buffer = new Float32Array(CHUNK_SAMPLES);
        this._filled = 0;
      }
    }

    return true;  // keep processor alive
  }
}

registerProcessor("mic-processor", MicProcessor);
