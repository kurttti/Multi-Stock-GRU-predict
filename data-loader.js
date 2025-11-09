class DataLoader {
    constructor() {
        this.stocksData = null;
        this.normalizedData = null;
        this.symbols = [];
        this.dates = [];
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.testDates = [];
        // EXACT spec: 2 features per stock (Open, Close) -> input (12, 20)
        this.numFeaturesPerStock = 0;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
                    resolve(this.stocksData);
                } catch (error) { reject(error); }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        const data = {};
        const symbols = new Set();
        const dates = new Set();

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            if (values.length !== headers.length) continue;

            const row = {};
            headers.forEach((header, index) => row[header.trim()] = values[index].trim());

            const symbol = row.Symbol;
            const date = row.Date;
            symbols.add(symbol);
            dates.add(date);

            if (!data[symbol]) data[symbol] = {};
            data[symbol][date] = {
                Open: parseFloat(row.Open),
                Close: parseFloat(row.Close)
            };
        }

        this.symbols = Array.from(symbols).sort();
        this.dates = Array.from(dates).sort();
        this.stocksData = data;
    }

    normalizeData() {
        if (!this.stocksData) throw new Error('No data loaded');
        this.normalizedData = {};
        const minMax = {};

        // Min/max for Open & Close only
        this.symbols.forEach(symbol => {
            minMax[symbol] = {
                Open:  { min: Infinity, max: -Infinity },
                Close: { min: Infinity, max: -Infinity }
            };
            this.dates.forEach(date => {
                const p = this.stocksData[symbol][date];
                if (!p) return;
                minMax[symbol].Open.min  = Math.min(minMax[symbol].Open.min,  p.Open);
                minMax[symbol].Open.max  = Math.max(minMax[symbol].Open.max,  p.Open);
                minMax[symbol].Close.min = Math.min(minMax[symbol].Close.min, p.Close);
                minMax[symbol].Close.max = Math.max(minMax[symbol].Close.max, p.Close);
            });
        });

        this.symbols.forEach(symbol => {
            this.normalizedData[symbol] = {};
            this.dates.forEach(date => {
                const p = this.stocksData[symbol][date];
                if (!p) return;
                const nOpen  = (minMax[symbol].Open.max  - minMax[symbol].Open.min)  || 1;
                const nClose = (minMax[symbol].Close.max - minMax[symbol].Close.min) || 1;
                this.normalizedData[symbol][date] = {
                    Open:  (p.Open  - minMax[symbol].Open.min)  / nOpen,
                    Close: (p.Close - minMax[symbol].Close.min) / nClose
                };
            });
        });

        this.numFeaturesPerStock = 2; // (Open, Close)
        return this.normalizedData;
    }

    createSequences(sequenceLength = 12, predictionHorizon = 3) {
        if (!this.normalizedData) this.normalizeData();

        const sequences = [];
        const targets = [];
        const validDates = [];

        for (let i = sequenceLength; i < this.dates.length - predictionHorizon; i++) {
            const currentDate = this.dates[i];
            const seq = [];
            let ok = true;

            // Build 12×(10*2)=12×20 inputs
            for (let j = sequenceLength - 1; j >= 0; j--) {
                const d = this.dates[i - j];
                const step = [];
                this.symbols.forEach(sym => {
                    const n = this.normalizedData[sym][d];
                    if (!n) { ok = false; return; }
                    step.push(n.Open, n.Close);
                });
                if (ok) seq.push(step);
            }
            if (!ok) continue;

            // Labels: for each stock compare Close(D+offset) vs Close(D)
            const baseClose = this.symbols.map(sym => this.stocksData[sym][currentDate].Close);
            const tgt = [];
            for (let off = 1; off <= predictionHorizon; off++) {
                const fDate = this.dates[i + off];
                this.symbols.forEach((sym, idx) => {
                    const f = this.stocksData[sym][fDate];
                    if (!f) { ok = false; return; }
                    tgt.push(f.Close > baseClose[idx] ? 1 : 0);
                });
                if (!ok) break;
            }
            if (!ok) continue;

            sequences.push(seq);
            targets.push(tgt);
            validDates.push(currentDate);
        }

        const split = Math.floor(sequences.length * 0.8);
        this.X_train = tf.tensor3d(sequences.slice(0, split));
        this.y_train = tf.tensor2d(targets.slice(0, split));
        this.X_test  = tf.tensor3d(sequences.slice(split));
        this.y_test  = tf.tensor2d(targets.slice(split));
        this.testDates = validDates.slice(split);

        return {
            X_train: this.X_train, y_train: this.y_train,
            X_test: this.X_test,   y_test: this.y_test,
            symbols: this.symbols, testDates: this.testDates
        };
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test)  this.X_test.dispose();
        if (this.y_test)  this.y_test.dispose();
    }
}

export default DataLoader;
