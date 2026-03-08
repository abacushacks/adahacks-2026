class PCMStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.bufferSize = 2048;
  }

  process(inputs) {
    const channelData = inputs[0]?.[0];
    if (channelData) {
      for (let index = 0; index < channelData.length; index += 1) {
        this.buffer.push(channelData[index]);
      }

      while (this.buffer.length >= this.bufferSize) {
        const chunk = this.buffer.slice(0, this.bufferSize);
        this.buffer = this.buffer.slice(this.bufferSize);
        this.port.postMessage(new Float32Array(chunk));
      }
    }

    return true;
  }
}

registerProcessor("pcm-stream-processor", PCMStreamProcessor);
