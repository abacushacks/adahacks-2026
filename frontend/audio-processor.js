class AudioStreamProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.chunkCount = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];

        if (input && input.length > 0) {
            const channelData = input[0]; // First channel (mono)

            if (channelData && channelData.length > 0) {
                // Copy the data to ensure it's transferred properly
                const audioCopy = new Float32Array(channelData.length);
                audioCopy.set(channelData);

                // Send the audio data to the main thread
                this.port.postMessage({
                    type: 'audio',
                    data: audioCopy
                }, [audioCopy.buffer]); // Transfer ownership for efficiency

                this.chunkCount++;
            }
        }

        // Return true to keep the processor alive
        return true;
    }
}

registerProcessor('audio-stream-processor', AudioStreamProcessor);
