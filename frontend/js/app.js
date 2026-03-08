document.addEventListener('DOMContentLoaded', async () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const gateIndicator = document.getElementById('gateIndicator');
    const holdTimeInput = document.getElementById('holdTime');

    const audioManager = new AudioManager('ws://localhost:8000/ws/audio/');
    audioManager.onTranscription = (text) => console.log("[Whisper] " + text);
    audioManager.onDebug = (message) => console.log("[Debug] " + message);
    audioManager.connect();

    const faceTracker = new FaceTracker(video, canvas, audioManager);
    await faceTracker.loadModels();

    audioManager.onFaceRecognized = (data) => {
        console.log("[Face] Recognized:", data.label, data.name);
        faceTracker.updatePersonData(data.label, data.name, data.metadata, data.relationship);
    };

    // Setup Webcam & Audio
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 },
            audio: true
        });
        video.srcObject = stream;
        audioManager.setupAudioInteraction(stream);
    } catch (err) {
        console.error("Camera/Microphone access error:", err);
        alert("Camera/Microphone access is required");
        return;
    }

    video.addEventListener('play', () => {
        faceTracker.detect();
        renderLoop();
    });

    function renderLoop() {
        if (video.paused || video.ended) {
            requestAnimationFrame(renderLoop);
            return;
        }

        const ctx = canvas.getContext('2d');

        // Match canvas size to window
        if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }

        if (video.videoWidth === 0 || video.videoHeight === 0) {
            requestAnimationFrame(renderLoop);
            return;
        }

        // Calculate video scaling to cover the canvas
        const videoAspect = video.videoWidth / video.videoHeight;
        const canvasAspect = canvas.width / canvas.height;
        let drawWidth, drawHeight, offsetX = 0, offsetY = 0;

        if (canvasAspect > videoAspect) {
            drawWidth = canvas.width;
            drawHeight = canvas.width / videoAspect;
            offsetY = (canvas.height - drawHeight) / 2;
        } else {
            drawHeight = canvas.height;
            drawWidth = canvas.height * videoAspect;
            offsetX = (canvas.width - drawWidth) / 2;
        }

        ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);

        const scaleX = drawWidth / video.videoWidth;
        const scaleY = drawHeight / video.videoHeight;

        faceTracker.update();

        // Hide popups that are not associated with any current detection
        const detectedLabels = new Set(faceTracker.currentDetections.map(d => d.label));
        faceTracker.activePopups.forEach((popup, label) => {
            if (!detectedLabels.has(label)) {
                popup.style.display = 'none';
            }
        });

        // Send active face labels to backend so it knows who's on screen
        const currentLabels = Array.from(detectedLabels).filter(l => l !== 'unknown').sort();
        const labelsKey = currentLabels.join(',');
        if (labelsKey !== renderLoop._lastLabelsKey) {
            renderLoop._lastLabelsKey = labelsKey;
            if (audioManager.socket && audioManager.socket.readyState === WebSocket.OPEN) {
                audioManager.socket.send(JSON.stringify({
                    type: 'active_faces',
                    labels: currentLabels
                }));
            }
        }

        faceTracker.currentDetections.forEach(detection => {
            const { x, y, width, height } = detection.box;
            const scaledX = x * scaleX + offsetX;
            const scaledY = y * scaleY + offsetY;
            const scaledWidth = width * scaleX;
            const scaledHeight = height * scaleY;

            ctx.strokeStyle = '#dfb15b'; // Match the golden/orange color from the image
            ctx.lineWidth = 2;

            // Draw rounded rectangle
            const radius = 20;
            ctx.beginPath();
            ctx.moveTo(scaledX + radius, scaledY);
            ctx.lineTo(scaledX + scaledWidth - radius, scaledY);
            ctx.quadraticCurveTo(scaledX + scaledWidth, scaledY, scaledX + scaledWidth, scaledY + radius);
            ctx.lineTo(scaledX + scaledWidth, scaledY + scaledHeight - radius);
            ctx.quadraticCurveTo(scaledX + scaledWidth, scaledY + scaledHeight, scaledX + scaledWidth - radius, scaledY + scaledHeight);
            ctx.lineTo(scaledX + radius, scaledY + scaledHeight);
            ctx.quadraticCurveTo(scaledX, scaledY + scaledHeight, scaledX, scaledY + scaledHeight - radius);
            ctx.lineTo(scaledX, scaledY + radius);
            ctx.quadraticCurveTo(scaledX, scaledY, scaledX + radius, scaledY);
            ctx.closePath();
            ctx.stroke();

            // Update floating popup
            const popup = faceTracker.activePopups.get(detection.label);
            if (popup) {
                const popupWidth = 250; // Updated to match CSS
                const margin = 20;
                let left = scaledX + scaledWidth + margin;
                if (left + popupWidth > window.innerWidth) {
                    left = scaledX - popupWidth - margin;
                }
                if (left < 0) left = 0;

                const popupHeight = popup.offsetHeight || 120;
                let top = scaledY;
                if (top + popupHeight > window.innerHeight) {
                    top = window.innerHeight - popupHeight - 10;
                }
                if (top < 0) top = 0;

                popup.style.left = `${left}px`;
                popup.style.top = `${top}px`;
                popup.style.display = 'block';
            }

            // // Draw label background
            // ctx.fillStyle = detection.color || '#00ff00';
            // const text = detection.label === 'unknown' ? 'Identifying...' : detection.label;
            // ctx.font = 'bold 16px Arial';
            // const textWidth = ctx.measureText(text).width;
            // ctx.fillRect(scaledX, scaledY - 25, textWidth + 10, 25);
            //
            // // Draw label text
            // ctx.fillStyle = '#000000';
            // ctx.fillText(text, scaledX + 5, scaledY - 7);
        });

        // Update gate indicator UI
        if (gateIndicator) {
            const holdTime = parseInt(holdTimeInput?.value) || 1500;
            const isGateOpen = Date.now() - audioManager.lastSpeechTime < holdTime;
            if (isGateOpen) {
                gateIndicator.textContent = 'OPEN';
                gateIndicator.style.color = '#00ff00';
            } else {
                gateIndicator.textContent = 'CLOSED';
                gateIndicator.style.color = '#ff4444';
            }
        }

        requestAnimationFrame(renderLoop);
    }
});
