class MrtPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.header = null;
    this.samples = null;
    this.port.onmessage = (event) => {
      const {buffer, headerWords} = event.data;
      this.header = new Int32Array(buffer, 0, headerWords);
      this.samples = new Float32Array(buffer, headerWords * 4);
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!this.header || !this.samples || output.length < 2) return true;
    const frames = output[0].length;
    const capacity = this.samples.length;
    let read = Atomics.load(this.header, 1);
    const write = Atomics.load(this.header, 0);
    const available = (write - read + capacity) % capacity;
    if (available < frames * 2 || Atomics.load(this.header, 3) === 0) {
      output[0].fill(0);
      output[1].fill(0);
      if (Atomics.load(this.header, 3) !== 0) Atomics.add(this.header, 2, 1);
      return true;
    }
    for (let frame = 0; frame < frames; frame++) {
      output[0][frame] = this.samples[read];
      read = (read + 1) % capacity;
      output[1][frame] = this.samples[read];
      read = (read + 1) % capacity;
    }
    Atomics.store(this.header, 1, read);
    return true;
  }
}

registerProcessor('mrt-pcm-processor', MrtPcmProcessor);
