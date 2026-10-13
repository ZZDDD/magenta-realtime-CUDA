const HEADER_WORDS = 4;
const FRAME_SAMPLES = 1920;
const CHANNELS = 2;

export class PcmPlayer {
  private context: AudioContext | null = null;
  private header: Int32Array;
  private samples: Float32Array;
  private capacity: number;
  level = 0;
  private revision = -1;

  constructor(private targetFrames = 2) {
    this.capacity = FRAME_SAMPLES * CHANNELS * 8 + 1;
    const shared = new SharedArrayBuffer(HEADER_WORDS * 4 + this.capacity * 4);
    this.header = new Int32Array(shared, 0, HEADER_WORDS);
    this.samples = new Float32Array(shared, HEADER_WORDS * 4);
    (this as unknown as {shared: SharedArrayBuffer}).shared = shared;
  }

  async start() {
    if (!crossOriginIsolated) throw new Error('Cross-origin isolation is required for low-latency audio');
    if (!this.context) {
      this.context = new AudioContext({sampleRate: 48000, latencyHint: 'interactive'});
      await this.context.audioWorklet.addModule('/pcm-worklet.js');
      const node = new AudioWorkletNode(this.context, 'mrt-pcm-processor', {
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });
      node.port.postMessage({
        buffer: (this as unknown as {shared: SharedArrayBuffer}).shared,
        headerWords: HEADER_WORDS,
      });
      node.connect(this.context.destination);
    }
    await this.context.resume();
  }

  suspend() {
    Atomics.store(this.header, 3, 0);
    void this.context?.suspend();
  }

  clear() {
    Atomics.store(this.header, 0, 0);
    Atomics.store(this.header, 1, 0);
    Atomics.store(this.header, 2, 0);
    Atomics.store(this.header, 3, 0);
  }

  setTargetFrames(targetFrames: number) {
    this.targetFrames = Math.max(1, targetFrames);
  }

  write(pcm: Int16Array, revision: number) {
    if (revision < this.revision) return;
    if (revision > this.revision) {
      this.clear();
      this.revision = revision;
    }
    let write = Atomics.load(this.header, 0);
    let read = Atomics.load(this.header, 1);
    let squareSum = 0;
    for (let index = 0; index < pcm.length; index++) {
      const next = (write + 1) % this.capacity;
      if (next === read) {
        read = (read + 2) % this.capacity;
        Atomics.store(this.header, 1, read);
      }
      const value = pcm[index] / 32768;
      this.samples[write] = value;
      squareSum += value * value;
      write = next;
    }
    Atomics.store(this.header, 0, write);
    this.level = Math.sqrt(squareSum / Math.max(1, pcm.length));
    if (this.queuedFrames >= this.targetFrames) Atomics.store(this.header, 3, 1);
  }

  get queuedFrames() {
    const write = Atomics.load(this.header, 0);
    const read = Atomics.load(this.header, 1);
    const values = (write - read + this.capacity) % this.capacity;
    return values / (FRAME_SAMPLES * CHANNELS);
  }

  get underruns() {
    return Atomics.load(this.header, 2);
  }
}
