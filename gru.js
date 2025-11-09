class GRUModel {
    constructor(inputShape, outputSize) {
        this.model = null;
        this.inputShape = inputShape;   // [12, 20]
        this.outputSize = outputSize;   // 30 (10*3)
        this.history = null;
        this.thresholds = Array(outputSize).fill(0.5);
        this._minEpochs = 10; // don't stop before this
    }

    // Early stopping on val_binaryAccuracy
    _makeEarlyStop(patience = 6) {
        let best = -Infinity, wait = 0;
        let epochCount = 0;
        return {
            onEpochEnd: (_epoch, logs) => {
                epochCount++;
                const v = logs.val_binaryAccuracy ?? 0;
                if (v > best + 1e-4) { best = v; wait = 0; } else { wait++; }
                if (epochCount >= this._minEpochs && wait >= patience) {
                    this.model.stopTraining = true;
                }
            }
        };
    }

    buildModel() {
        // Spec: stacked GRU → Dense(30, sigmoid)
        const layers = [];
        layers.push(tf.layers.gru({
            units: 48,
            returnSequences: true,
            inputShape: this.inputShape,
            dropout: 0.1,
            recurrentDropout: 0.05
        }));
        layers.push(tf.layers.gru({
            units: 24,
            returnSequences: false,
            dropout: 0.1,
            recurrentDropout: 0.05
        }));
        layers.push(tf.layers.dense({
            units: this.outputSize,
            activation: 'sigmoid'
        }));

        this.model = tf.sequential({ layers });
        this.model.compile({
            optimizer: tf.train.adam(0.0015), // fixed LR (no setLearningRate)
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });
        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 28, batchSize = 64) {
        if (!this.model) this.buildModel();

        const tryBatches = [batchSize, 48, 32, 24, 16];
        let lastErr = null;
        for (const bs of tryBatches) {
            try {
                this.history = await this.model.fit(X_train, y_train, {
                    epochs,
                    batchSize: bs,
                    shuffle: true,
                    validationData: [X_test, y_test],
                    callbacks: [
                        this._makeEarlyStop(6),
                        {
                            onEpochEnd: (epoch, logs) => {
                                const p = document.getElementById('trainingProgress');
                                const s = document.getElementById('status');
                                if (p) p.value = ((epoch + 1) / epochs) * 100;
                                if (s) s.textContent =
                                    `Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, acc: ${logs.binaryAccuracy.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_binaryAccuracy.toFixed(4)}`;
                            }
                        }
                    ]
                });
                return this.history;
            } catch (e) {
                lastErr = e;
                await tf.nextFrame();
            }
        }
        throw lastErr || new Error('Training failed');
    }

    async predict(X) {
        if (!this.model) throw new Error('Model not trained');
        return this.model.predict(X);
    }

    // Wider threshold sweep (0.2–0.8) to squeeze accuracy without retraining
    setThresholdsFromValidation(yTrue, yPred) {
        const yT = yTrue.arraySync();
        const yP = yPred.arraySync();
        const grid = [];
        for (let t = 0.2; t <= 0.8; t += 0.01) grid.push(parseFloat(t.toFixed(2)));

        for (let k = 0; k < this.outputSize; k++) {
            let bestT = 0.5, bestAcc = -1;
            for (const th of grid) {
                let correct = 0, total = 0;
                for (let i = 0; i < yT.length; i++) {
                    const pred = yP[i][k] > th ? 1 : 0;
                    if (pred === yT[i][k]) correct++;
                    total++;
                }
                const acc = correct / (total || 1);
                if (acc > bestAcc) { bestAcc = acc; bestT = th; }
            }
            this.thresholds[k] = bestT;
        }
    }

    evaluatePerStock(yTrue, yPred, symbols, horizon = 3) {
        const yT = yTrue.arraySync();
        const yP = yPred.arraySync();
        const stockAccuracies = {};
        const stockPredictions = {};

        symbols.forEach((symbol, sIdx) => {
            let correct = 0, total = 0;
            const preds = [];
            for (let i = 0; i < yT.length; i++) {
                for (let o = 0; o < horizon; o++) {
                    const k = sIdx * horizon + o;
                    const th = this.thresholds[k] ?? 0.5;
                    const pred = yP[i][k] > th ? 1 : 0;
                    const truth = yT[i][k];
                    if (pred === truth) correct++;
                    total++;
                    preds.push({ true: truth, pred, correct: pred === truth });
                }
            }
            stockAccuracies[symbol] = total ? correct / total : 0;
            stockPredictions[symbol] = preds;
        });

        return { stockAccuracies, stockPredictions };
    }

    dispose() { if (this.model) this.model.dispose(); }
}

export default GRUModel;
