import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.currentPredictions = null;
        this.accuracyChart = null;
        this.isTraining = false;
        // Initialise TFJS backend and update backend label
        this.initBackend();
        this.initializeEventListeners();
    }

    async initBackend() {
        try {
            await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
            await tf.ready();
            const info = `TFJS ${tf.version_core} | backend: ${tf.getBackend()}`;
            const el = document.getElementById('backendInfo');
            if (el) el.textContent = info;
            console.log(info);
        } catch (err) {
            console.warn('Backend init error:', err);
        }
    }

    initializeEventListeners() {
        const fileInput = document.getElementById('csvFile');
        const trainBtn = document.getElementById('trainBtn');
        const predictBtn = document.getElementById('predictBtn');
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        trainBtn.addEventListener('click', () => this.trainModel());
        predictBtn.addEventListener('click', () => this.runPrediction());
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        try {
            this.updateStatus('Loading CSV...');
            await this.dataLoader.loadCSV(file);
            this.updateStatus('Preprocessing data...');
            this.dataLoader.createSequences();
            document.getElementById('trainBtn').disabled = false;
            this.updateStatus('Data loaded. Click Train Model to begin training.');
        } catch (error) {
            this.updateStatus(`Error: ${error.message}`);
            console.error(error);
        }
    }

    async trainModel() {
        if (this.isTraining) return;
        this.isTraining = true;
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('predictBtn').disabled = true;
        try {
            const { X_train, y_train, X_test, y_test, symbols } = this.dataLoader;
            const numFeatures = this.dataLoader.numFeaturesPerStock || 2;
            this.model = new GRUModel([12, symbols.length * numFeatures], symbols.length * 3);
            this.updateStatus('Training model...');
            await this.model.train(X_train, y_train, X_test, y_test, 20, 32);
            document.getElementById('predictBtn').disabled = false;
            this.updateStatus('Training completed. Click Run Prediction to evaluate.');
        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    async runPrediction() {
        if (!this.model) {
            alert('Please train the model first');
            return;
        }
        try {
            this.updateStatus('Running predictions...');
            const { X_test, y_test, symbols } = this.dataLoader;
            const preds = await this.model.predict(X_test);
            const evaln = this.model.evaluatePerStock(y_test, preds, symbols);
            this.currentPredictions = evaln;
            this.visualizeResults(evaln);
            this.updateStatus('Prediction completed. Results displayed below.');
            preds.dispose();
        } catch (error) {
            this.updateStatus(`Prediction error: ${error.message}`);
            console.error(error);
        }
    }

    visualizeResults(evaluation) {
        this.createAccuracyChart(evaluation.stockAccuracies);
        this.createTimelineChart(evaluation.stockPredictions, evaluation.stockAccuracies);
    }

    createAccuracyChart(accuracies) {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        const sorted = Object.entries(accuracies).sort(([, a], [, b]) => b - a);
        const labels = sorted.map(([s]) => s);
        const values = sorted.map(([, a]) => a * 100);
        const colours = values.map((v, i) => i === 0 ? 'rgba(0, 191, 255, 0.8)' : 'rgba(255, 165, 0, 0.8)');
        const borders = values.map((v, i) => i === 0 ? 'rgb(0, 191, 255)' : 'rgb(255, 165, 0)');
        if (this.accuracyChart) this.accuracyChart.destroy();
        this.accuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Accuracy (%)',
                    data: values,
                    backgroundColor: colours,
                    borderColor: borders,
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Accuracy (%)' }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    createTimelineChart(predictions, accuracies) {
        const container = document.getElementById('timelineContainer');
        container.innerHTML = '';
        const topStock = Object.entries(accuracies).sort(([, a], [, b]) => b - a)[0][0];
        const stockPreds = predictions[topStock] || [];
        const div = document.createElement('div');
        div.className = 'stock-chart';
        div.innerHTML = `<h4>${topStock} Prediction Timeline</h4><canvas id="timeline-chart"></canvas>`;
        container.appendChild(div);
        const ctx = document.getElementById('timeline-chart').getContext('2d');
        const sample = stockPreds.slice(0, Math.min(50, stockPreds.length));
        const data = sample.map(p => p.correct ? 1 : 0);
        const labels = sample.map((_, i) => `Pred ${i + 1}`);
        new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Correct Predictions',
                    data,
                    borderColor: 'rgb(0, 191, 255)',
                    backgroundColor: 'rgba(0, 191, 255, 0.2)',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: sample.map(p => p.correct ? 'rgb(0, 191, 255)' : 'rgb(255, 99, 132)')
                }]
            },
            options: {
                scales: {
                    y: {
                        min: 0,
                        max: 1,
                        ticks: {
                            callback: v => v === 1 ? 'Correct' : v === 0 ? 'Wrong' : ''
                        }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    updateStatus(text) {
        const el = document.getElementById('status');
        if (el) el.textContent = text;
    }

    dispose() {
        if (this.dataLoader) this.dataLoader.dispose();
        if (this.model) this.model.dispose();
        if (this.accuracyChart) this.accuracyChart.destroy();
    }
}

document.addEventListener('DOMContentLoaded', () => new StockPredictionApp());
