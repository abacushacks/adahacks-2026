class AudioManager {
    constructor(socketUrl) {
        this.socketUrl = socketUrl;
        this.socket = null;
        this.audioContext = null;
        this.lastSpeechTime = 0;
        this.isGateOpen = false;
        this.holdTime = 1500;
        this.onTranscription = null;
        this.onDebug = null;
        this.onFaceRecognized = null;
    }

    connect() {
        this.socket = new WebSocket(this.socketUrl);

        this.socket.onopen = () => {
            console.log("[WebSocket] Connected to backend");
            if (this.onConnect) this.onConnect();
        };

        this.socket.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.type === 'transcription' && this.onTranscription) {
                    this.onTranscription(data.text);
                } else if (data.type === 'debug' && this.onDebug) {
                    this.onDebug(data.message);
                } else if (data.type === 'face_recognized' && this.onFaceRecognized) {
                    this.onFaceRecognized(data);
                }
            } catch (err) {
                // Not a JSON message
            }
        };

        this.socket.onclose = () => {
            console.log("[WebSocket] Disconnected from backend");
        };

        this.socket.onerror = (err) => {
            console.error("[WebSocket] Error:", err);
        };
    }

    setupAudioInteraction(stream) {
        const startAudio = async () => {
            console.log("[Audio] Interaction detected - initializing audio context");
            if (this.audioContext && this.audioContext.state === 'running') return;

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            this.socket.send(JSON.stringify({
                type: 'config',
                sampleRate: this.audioContext.sampleRate
            }));

            let audioStream = stream;
            let audioTracks = audioStream ? audioStream.getAudioTracks() : [];
            
            if (audioTracks.length === 0 || audioTracks[0].readyState === 'ended') {
                try {
                    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                } catch (err) {
                    console.error("[Audio] Could not acquire audio stream:", err);
                    return;
                }
            }

            this.startProcessor(audioStream);
            window.removeEventListener('mousedown', startAudio);
        };

        window.addEventListener('mousedown', startAudio);
    }

    startProcessor(stream) {
        const source = this.audioContext.createMediaStreamSource(stream);
        const processor = this.audioContext.createScriptProcessor(2048, 1, 1);

        source.connect(processor);
        processor.connect(this.audioContext.destination);

        let lastGateStatus = false;

        processor.onaudioprocess = (e) => {
            if (this.socket.readyState === WebSocket.OPEN) {
                this.holdTime = parseInt(document.getElementById('holdTime')?.value) || 1500;
                this.isGateOpen = Date.now() - this.lastSpeechTime < this.holdTime;

                if (this.isGateOpen) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    this.socket.send(inputData);
                } else if (lastGateStatus) {
                    console.log("[Audio] Gate closed - flushing buffer...");
                    this.socket.send(JSON.stringify({ type: 'flush' }));
                }
                lastGateStatus = this.isGateOpen;
            }
        };
    }

    updateSpeechTime() {
        this.lastSpeechTime = Date.now();
    }

    sendFaceDescriptor(label, descriptor) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                type: 'face_descriptor',
                label: label,
                descriptor: Array.from(descriptor)
            }));
        }
    }
}
