import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.currentPredictions = null;
        this.accuracyChart = null;
        this.isTraining = false;

        // prepare TF backend immediately
        this.tfReadyPromise = this.setupTF();

        this.initializeEventListeners();
    }

    async setupTF() {
        try {
            // prefer WebGL for speed; fallback to CPU if not available
            await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
            await tf.ready();
            const info = `TFJS ${tf.version_core} | backend: ${tf.getBackend()}`;
            const el = document.getElementById('backendInfo');
            if (el) el.textContent = info;
            console.log(info);
        } catch (e) {
            console.warn('TFJS init failed', e);
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
            await this.tfReadyPromise;

            document.getElementById('status').textContent = 'Loading CSV...';
            await this.dataLoader.loadCSV(file);
            
            document.getElementById('status').textContent = 'Preprocessing data...';
            this.dataLoader.createSequences();
            
            document.getElementById('trainBtn').disabled = false;
            document.getElementById('status').textContent = 'Data loaded. Click Train Model to begin training.';
            
        } catch (error) {
            document.getElementById('status').textContent = `Error: ${error.message}`;
            console.error(error);
        }
    }

    async trainModel() {
        if (this.isTraining) return;
        await this.tfReadyPromise;
        
        this.isTraining = true;
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('predictBtn').disabled = true;

        try {
            const { X_train, y_train, X_test, y_test, symbols } = this.dataLoader;

            const numFeatures = this.dataLoader.numFeaturesPerStock || 2; // 6 with new features
            this.model = new GRUModel([12, symbols.length * numFeatures], symbols.length * 3);
            
            document.getElementById('status').textContent = 'Training model...';
            // a few extra epochs, browser-friendly
            await this.model.train(X_train, y_train, X_test, y_test, 60, 32);
            
            document.getElementById('predictBtn').disabled = false;
            document.getElementById('status').textContent = 'Training completed. Click Run Prediction to evaluate.';
            
        } catch (error) {
            document.getElementById('status').textContent = `Training error: ${error.message}`;
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
            document.getElementById('status').textContent = 'Running predictions...';
            const { X_test, y_test, symbols } = this.dataLoader;
            
            const predictions = await this.model.predict(X_test);
            const evaluation = this.model.evaluatePerStock(y_test, predictions, symbols);
            
            this.currentPredictions = evaluation;
            this.visualizeResults(evaluation, symbols);
            
            document.getElementById('status').textContent = 'Prediction completed. Results displayed below.';
            predictions.dispose();
        } catch (error) {
            document.getElementById('status').textContent = `Prediction error: ${error.message}`;
            console.error(error);
        }
    }

    visualizeResults(evaluation, symbols) {
        this.createAccuracyChart(evaluation.stockAccuracies, symbols);
        this.createTimelineCharts(evaluation.stockPredictions, evaluation.stockAccuracies);
    }

    createAccuracyChart(accuracies, symbols) {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        const sortedEntries = Object.entries(accuracies).sort(([,a], [,b]) => b - a);
        const sortedSymbols = sortedEntries.map(([symbol]) => symbol);
        const sortedAccuracies = sortedEntries.map(([, acc]) => acc * 100);

        if (this.accuracyChart) this.accuracyChart.destroy();

        this.accuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sortedSymbols,
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: sortedAccuracies,
                    backgroundColor: 'rgba(75, 192, 192, 0.8)',
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: { beginAtZero: true, max: 100, title: { display: true, text: 'Accuracy (%)' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    createTimelineCharts(predictions, accuracies) {
        const container = document.getElementById('timelineContainer');
        container.innerHTML = '';

        const topStocks = Object.entries(accuracies)
            .sort(([,a], [,b]) => b - a)
            .map(([s]) => s)
            .slice(0, 3);

        topStocks.forEach(symbol => {
            const stockPredictions = predictions[symbol] || [];
            const chartContainer = document.createElement('div');
            chartContainer.className = 'stock-chart';
            chartContainer.innerHTML = `<h4>${symbol} Prediction Timeline</h4><canvas id="timeline-${symbol}"></canvas>`;
            container.appendChild(chartContainer);

            const ctx = document.getElementById(`timeline-${symbol}`).getContext('2d');
            const sampleSize = Math.min(50, stockPredictions.length);
            const sampleData = stockPredictions.slice(0, sampleSize);
            const correctData = sampleData.map(p => p.correct ? 1 : 0);
            const labels = sampleData.map((_, i) => `#${i + 1}`);

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'Correct(1)/Wrong(0)',
                        data: correctData,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: sampleData.map(p => p.correct ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)')
                    }]
                },
                options: {
                    scales: { y: { min: 0, max: 1, ticks: { callback: v => (v === 1 ? 'Correct' : v === 0 ? 'Wrong' : '') } } },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: (ctx) => {
                                    const pred = sampleData[ctx.dataIndex];
                                    return `Pred: ${pred.pred ? 'Up' : 'Down'} | Actual: ${pred.true ? 'Up' : 'Down'}`;
                                }
                            }
                        }
                    }
                }
            });
        });
    }

    dispose() {
        if (this.dataLoader) this.dataLoader.dispose();
        if (this.model) this.model.dispose();
        if (this.accuracyChart) this.accuracyChart.destroy();
    }
}

document.addEventListener('DOMContentLoaded', () => new StockPredictionApp());
