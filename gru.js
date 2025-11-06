class GRUModel {
    constructor(inputShape, outputSize) {
        this.model = null;
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.history = null;
    }

    buildModel() {
        // Browser-friendly depth/width to avoid WebGL OOM, but still stronger than baseline
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({ units: 96, returnSequences: true, inputShape: this.inputShape }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.gru({ units: 64, returnSequences: true }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.gru({ units: 32, returnSequences: false }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 64, activation: 'relu' }),
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

    async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
        if (!this.model) this.buildModel();

        // try-catch to downshift batch size if device struggles
        let bs = Math.max(8, Math.min(batchSize, Math.floor(Math.max(1, X_train.shape[0] / 10))));
        let lastErr = null;
        for (let attempt = 0; attempt < 3; attempt++) {
            try {
                this.history = await this.model.fit(X_train, y_train, {
                    epochs,
                    batchSize: bs,
                    validationData: [X_test, y_test],
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            const progress = ((epoch + 1) / epochs) * 100;
                            const status = `Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, acc: ${logs.binaryAccuracy.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_binaryAccuracy.toFixed(4)}`;
                            const progressElement = document.getElementById('trainingProgress');
                            const statusElement = document.getElementById('status');
                            if (progressElement) progressElement.value = progress;
                            if (statusElement) statusElement.textContent = status;
                            console.log(status);
                        }
                    }
                });
                return this.history;
            } catch (err) {
                console.warn(`Fit failed at batchSize=${bs}. Retrying with half batch...`, err);
                lastErr = err;
                bs = Math.max(4, Math.floor(bs / 2));
                await tf.nextFrame();
            }
        }
        throw lastErr || new Error('Training failed');
    }

    async predict(X) {
        if (!this.model) throw new Error('Model not trained');
        return this.model.predict(X);
    }

    evaluatePerStock(yTrue, yPred, symbols, horizon = 3) {
        const yTrueArray = yTrue.arraySync();
        const yPredArray = yPred.arraySync();
        const stockAccuracies = {};
        const stockPredictions = {};

        symbols.forEach((symbol, stockIdx) => {
            let correct = 0;
            let total = 0;
            const predictions = [];
            for (let i = 0; i < yTrueArray.length; i++) {
                for (let offset = 0; offset < horizon; offset++) {
                    const targetIdx = stockIdx * horizon + offset;
                    const trueVal = yTrueArray[i][targetIdx];
                    const predVal = yPredArray[i][targetIdx] > 0.5 ? 1 : 0;
                    if (trueVal === predVal) correct++;
                    total++;
                    predictions.push({ true: trueVal, pred: predVal, correct: trueVal === predVal });
                }
            }
            stockAccuracies[symbol] = total > 0 ? correct / total : 0;
            stockPredictions[symbol] = predictions;
        });

        return { stockAccuracies, stockPredictions };
    }

    dispose() {
        if (this.model) this.model.dispose();
    }
}

export default GRUModel;
