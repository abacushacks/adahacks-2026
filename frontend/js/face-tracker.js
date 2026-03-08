class FaceTracker {
    constructor(video, canvas, audioManager) {
        this.video = video;
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.audioManager = audioManager;
        
        this.labeledDescriptors = []; // { label, descriptors: Float32Array[], color }
        this.faceMatcher = null;
        this.temporaryTrackings = []; // { descriptor, count }
        this.activePopups = new Map(); // label -> HTMLElement
        this.activeFaceLabels = new Set();
        this.facePresenceGrace = new Map();
        this.currentDetections = [];
        this.targets = [];
        
        this.colors = ['#00ff00', '#ff0000', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ffa500', '#800080'];
        this.colorIdx = 0;
        this.lipOpeningThreshold = 0.05;

        this.personData = {
            'default': { name: 'Identifying...', metadata: ['Profile loading...'] }
        };

        this.popupsContainer = document.getElementById('popups-container');
    }

    async loadModels() {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model';
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
            faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);
        console.log("Models loaded");
    }

    onFaceEntered(detection) {
        const label = detection.label;
        if (label === 'unknown') return;

        const popup = document.createElement('div');
        popup.className = 'face-popup';
        // popup.style.borderColor = detection.color; // Removing direct border color if we want to follow the image style which has a darker border or no border
        
        const data = this.personData[label] || this.personData['default'];
        const name = data.name || 'Identifying...';
        const metadata = Array.isArray(data.metadata) ? data.metadata : (data.metadata ? [data.metadata] : []);

        popup.innerHTML = `
            <div class="popup-header">RECOGNIZED PERSON</div>
            <div class="popup-name">${name}</div>
            <div class="popup-status">Relationship not learned yet</div>
            <ul class="popup-memories">
                ${metadata.map(i => `<li>${i}</li>`).join('')}
            </ul>
        `;
        this.popupsContainer.appendChild(popup);
        this.activePopups.set(label, popup);
    }

    updatePersonData(label, name, metadata) {
        // If metadata is null or empty, use a placeholder
        const cleanMetadata = (Array.isArray(metadata) && metadata.length > 0) ? metadata : ['Profile loading...'];
        const cleanName = name || 'Identifying...';
        
        this.personData[label] = { name: cleanName, metadata: cleanMetadata };
        
        // If a popup is already active for this label, update its content
        const popup = this.activePopups.get(label);
        if (popup) {
            const nameElement = popup.querySelector('.popup-name');
            const memoriesElement = popup.querySelector('.popup-memories');
            
            if (nameElement) nameElement.textContent = cleanName;
            if (memoriesElement) {
                memoriesElement.innerHTML = cleanMetadata.map(i => `<li>${i}</li>`).join('');
            }
        }
    }

    onFaceTracked(detection) {
        const label = detection.label;
        if (label === 'unknown') return;
        if (!this.activePopups.has(label)) {
            this.onFaceEntered(detection);
        }
    }

    onFaceLeft(label) {
        const popup = this.activePopups.get(label);
        if (popup) {
            popup.remove();
            this.activePopups.delete(label);
        }
    }

    async detect() {
        if (this.video.paused || this.video.ended) return;
        
        try {
            const detections = await faceapi.detectAllFaces(this.video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
                .withFaceLandmarks()
                .withFaceDescriptors();
            
            this.updateFaceMatcher();

            const newDetections = detections.map(d => this.processDetection(d));
            
            this.handleFaceTransitions(newDetections);

            this.cleanupTemporaryTrackings();

            this.checkSpeech(newDetections);

            this.targets = newDetections;
        } catch (err) {
            console.error("Detection error:", err);
        }
        setTimeout(() => this.detect(), 100);
    }

    updateFaceMatcher() {
        if (this.labeledDescriptors.length > 0) {
            const labeledFaceDescriptors = this.labeledDescriptors.map(ld => 
                new faceapi.LabeledFaceDescriptors(ld.label, ld.descriptors)
            );
            this.faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
        }
    }

    processDetection(d) {
        let color = '#ffffff';
        let label = 'unknown';

        const mouth = d.landmarks.getMouth();
        const box = d.detection.box;
        const normalizedMouth = mouth.map(p => ({
            x: (p.x - box.x) / box.width,
            y: (p.y - box.y) / box.height
        }));

        if (this.faceMatcher && this.labeledDescriptors.length > 0) {
            const bestMatch = this.faceMatcher.findBestMatch(d.descriptor);
            label = bestMatch.label;
            
            if (label !== 'unknown') {
                const person = this.labeledDescriptors.find(ld => ld.label === label);
                color = person.color;
                if (person.descriptors.length < 15 && Math.random() > 0.8) {
                    person.descriptors.push(d.descriptor);
                }
            }
        }

        if (label === 'unknown') {
            const { updatedLabel, updatedColor } = this.handleUnknownFace(d);
            label = updatedLabel;
            color = updatedColor;
        }

        return { box: d.detection.box, color: color, label: label, normalizedMouth: normalizedMouth };
    }

    handleUnknownFace(d) {
        let label = 'unknown';
        let color = '#555555';

        let tempMatch = this.temporaryTrackings.find(t => {
            const dist = faceapi.euclideanDistance(d.descriptor, t.descriptor);
            return dist < 0.4;
        });

        if (tempMatch) {
            tempMatch.count++;
            tempMatch.descriptor = d.descriptor;
            
            if (tempMatch.count > 10) {
                color = this.colors[this.colorIdx % this.colors.length];
                label = crypto.randomUUID(); // Use a unique ID instead of "Person X"
                this.colorIdx++;
                this.labeledDescriptors.push({ 
                    label: label, 
                    descriptors: [d.descriptor], 
                    color: color 
                });
                
                if (this.audioManager) {
                    this.audioManager.sendFaceDescriptor(label, d.descriptor);
                }

                this.temporaryTrackings = this.temporaryTrackings.filter(t => t !== tempMatch);
            }
        } else {
            this.temporaryTrackings.push({ descriptor: d.descriptor, count: 1 });
        }

        return { updatedLabel: label, updatedColor: color };
    }

    handleFaceTransitions(newDetections) {
        const currentConfirmedLabels = new Set(
            newDetections.filter(d => d.label !== 'unknown').map(d => d.label)
        );

        currentConfirmedLabels.forEach(label => {
            if (!this.activeFaceLabels.has(label)) {
                this.activeFaceLabels.add(label);
                const detection = newDetections.find(d => d.label === label);
                this.onFaceEntered(detection);
            }
        });

        newDetections.filter(d => d.label !== 'unknown').forEach(detection => {
            this.onFaceTracked(detection);
            this.facePresenceGrace.set(detection.label, 0);
        });

        this.activeFaceLabels.forEach(label => {
            if (!currentConfirmedLabels.has(label)) {
                let missingCount = (this.facePresenceGrace.get(label) || 0) + 1;
                this.facePresenceGrace.set(label, missingCount);

                if (missingCount > 15) {
                    this.onFaceLeft(label);
                    this.activeFaceLabels.delete(label);
                    this.facePresenceGrace.delete(label);
                }
            }
        });
    }

    cleanupTemporaryTrackings() {
        if (this.temporaryTrackings.length > 20) {
            this.temporaryTrackings.shift();
        }
    }

    checkSpeech(detections) {
        detections.forEach(nd => {
            if (nd.normalizedMouth) {
                const upperLip = nd.normalizedMouth[14];
                const lowerLip = nd.normalizedMouth[18];
                const distance = Math.abs(upperLip.y - lowerLip.y);
                if (distance > this.lipOpeningThreshold) {
                    this.audioManager.updateSpeechTime();
                }
            }
        });
    }

    update() {
        this.interpolateDetections();
        this.cleanupDetections();
    }

    interpolateDetections() {
        const lerp = (a, b, t) => a + (b - a) * t;
        const alpha = 0.2;

        this.targets.forEach(target => {
            let current = this.currentDetections.find(cd => cd.label === target.label && (cd.label !== 'unknown' || cd.color === target.color));
            if (current) {
                current.box.x = lerp(current.box.x, target.box.x, alpha);
                current.box.y = lerp(current.box.y, target.box.y, alpha);
                current.box.width = lerp(current.box.width, target.box.width, alpha);
                current.box.height = lerp(current.box.height, target.box.height, alpha);
                current.color = target.color;
                current.normalizedMouth = target.normalizedMouth;
                current.lastSeen = Date.now();
            } else {
                this.currentDetections.push({
                    ...target,
                    box: { 
                        x: target.box.x,
                        y: target.box.y,
                        width: target.box.width,
                        height: target.box.height
                    },
                    lastSeen: Date.now()
                });
            }
        });
    }

    cleanupDetections() {
        const now = Date.now();
        this.currentDetections = this.currentDetections.filter(cd => {
            const isTargeted = this.targets.some(t => t.label === cd.label && (t.label !== 'unknown' || t.color === cd.color));
            const isGraceful = this.activeFaceLabels.has(cd.label);
            return isTargeted || (isGraceful && (now - cd.lastSeen < 1000));
        });
    }
}
