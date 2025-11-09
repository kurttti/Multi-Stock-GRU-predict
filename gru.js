class GRUModel {
    constructor(inputShape, outputSize) {
        this.model = null;
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.history = null;
        // Per-output thresholds (stock × horizon), default 0.5
        this.thresholds = Array(outputSize).fill(0.5);
    }

    buildModel() {
        // Browser-friendly but a bit stronger: add a small dense before output
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({ units: 48, returnSequences: true, inputShape: this.inputShape }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.gru({ units: 24, returnSequences: false }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });
        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 20, batchSize = 32) {
        if (!this.model) this.buildModel();
        this.history = await this.model.fit(X_train, y_train, {
            epochs,
            batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const p = document.getElementById('trainingProgress');
                    const s = document.getElementById('status');
                    if (p) p.value = ((epoch + 1) / epochs) * 100;
                    if (s) s.textContent = `Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, acc: ${logs.binaryAccuracy.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_binaryAccuracy.toFixed(4)}`;
                }
            }
        });
        return this.history;
    }

    async predict(X) {
        if (!this.model) throw new Error('Model not trained');
        return this.model.predict(X);
    }

    // Tune per-output thresholds on validation predictions to maximize accuracy
    setThresholdsFromValidation(yTrue, yPred, horizon = 3) {
        const yT = yTrue.arraySync();
        const yP = yPred.arraySync();
        const out = this.outputSize;

        // search grid
        const grid = [];
        for (let t = 0.3; t <= 0.7; t += 0.02) grid.push(parseFloat(t.toFixed(2)));

        for (let k = 0; k < out; k++) {
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

    // Evaluate using tuned thresholds (or defaults if not set)
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
