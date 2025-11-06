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
        this.numFeaturesPerStock = 0; // dynamically set after normalize
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
                    resolve(this.stocksData);
                } catch (error) {
                    reject(error);
                }
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
            headers.forEach((header, index) => {
                row[header.trim()] = values[index].trim();
            });

            const symbol = row.Symbol;
            const date = row.Date;
            
            symbols.add(symbol);
            dates.add(date);

            if (!data[symbol]) data[symbol] = {};
            data[symbol][date] = {
                Open: parseFloat(row.Open),
                Close: parseFloat(row.Close),
                High: parseFloat(row.High),
                Low: parseFloat(row.Low),
                Volume: parseFloat(row.Volume)
            };
        }

        this.symbols = Array.from(symbols).sort();
        this.dates = Array.from(dates).sort();
        this.stocksData = data;

        console.log(`Loaded ${this.symbols.length} stocks with ${this.dates.length} trading days`);
    }

    normalizeData() {
        if (!this.stocksData) throw new Error('No data loaded');

        this.normalizedData = {};
        const minMax = {};

        // Per-stock min/max for Open/Close/High/Low/Volume + derived Return
        this.symbols.forEach(symbol => {
            minMax[symbol] = {
                Open: { min: Infinity, max: -Infinity },
                Close: { min: Infinity, max: -Infinity },
                High: { min: Infinity, max: -Infinity },
                Low: { min: Infinity, max: -Infinity },
                Volume: { min: Infinity, max: -Infinity },
                Return: { min: Infinity, max: -Infinity }
            };

            this.dates.forEach(date => {
                const p = this.stocksData[symbol][date];
                if (!p) return;
                minMax[symbol].Open.min = Math.min(minMax[symbol].Open.min, p.Open);
                minMax[symbol].Open.max = Math.max(minMax[symbol].Open.max, p.Open);
                minMax[symbol].Close.min = Math.min(minMax[symbol].Close.min, p.Close);
                minMax[symbol].Close.max = Math.max(minMax[symbol].Close.max, p.Close);
                minMax[symbol].High.min = Math.min(minMax[symbol].High.min, p.High);
                minMax[symbol].High.max = Math.max(minMax[symbol].High.max, p.High);
                minMax[symbol].Low.min = Math.min(minMax[symbol].Low.min, p.Low);
                minMax[symbol].Low.max = Math.max(minMax[symbol].Low.max, p.Low);
                minMax[symbol].Volume.min = Math.min(minMax[symbol].Volume.min, p.Volume);
                minMax[symbol].Volume.max = Math.max(minMax[symbol].Volume.max, p.Volume);
                const ret = p.Open !== 0 ? (p.Close - p.Open) / p.Open : 0;
                minMax[symbol].Return.min = Math.min(minMax[symbol].Return.min, ret);
                minMax[symbol].Return.max = Math.max(minMax[symbol].Return.max, ret);
            });
        });

        // Normalize to [0,1] (constant-range -> 0)
        this.symbols.forEach(symbol => {
            this.normalizedData[symbol] = {};
            this.dates.forEach(date => {
                const p = this.stocksData[symbol][date];
                if (!p) return;
                const norm = {};
                ['Open','Close','High','Low','Volume'].forEach(key => {
                    const denom = (minMax[symbol][key].max - minMax[symbol][key].min);
                    norm[key] = denom === 0 ? 0 : (p[key] - minMax[symbol][key].min) / denom;
                });
                const ret = p.Open !== 0 ? (p.Close - p.Open) / p.Open : 0;
                const denomR = (minMax[symbol].Return.max - minMax[symbol].Return.min);
                norm.Return = denomR === 0 ? 0 : (ret - minMax[symbol].Return.min) / denomR;

                this.normalizedData[symbol][date] = norm;
            });
        });

        this.numFeaturesPerStock = Object.keys(this.normalizedData[this.symbols[0]][this.dates[0]]).length; // 6
        return this.normalizedData;
    }

    createSequences(sequenceLength = 12, predictionHorizon = 3) {
        if (!this.normalizedData) this.normalizeData();

        const sequences = [];
        const targets = [];
        const validDates = [];

        for (let i = sequenceLength; i < this.dates.length - predictionHorizon; i++) {
            const currentDate = this.dates[i];
            const sequenceData = [];
            let valid = true;

            // Build 12-day window (earliest -> latest)
            for (let j = sequenceLength - 1; j >= 0; j--) {
                const seqDate = this.dates[i - j];
                const step = [];
                this.symbols.forEach(symbol => {
                    const n = this.normalizedData[symbol][seqDate];
                    if (!n) { valid = false; return; }
                    // feature order must stay stable:
                    step.push(n.Open, n.Close, n.High, n.Low, n.Volume, n.Return);
                });
                if (valid) sequenceData.push(step);
            }

            if (!valid) continue;

            // Targets: for each stock, compare Close at future offsets to Close at D
            const baseClose = this.symbols.map(sym => this.stocksData[sym][currentDate].Close);
            const tgt = [];
            for (let offset = 1; offset <= predictionHorizon; offset++) {
                const fDate = this.dates[i + offset];
                this.symbols.forEach((sym, idx) => {
                    const f = this.stocksData[sym][fDate];
                    if (!f) { valid = false; return; }
                    tgt.push(f.Close > baseClose[idx] ? 1 : 0);
                });
                if (!valid) break;
            }
            if (!valid) continue;

            sequences.push(sequenceData);
            targets.push(tgt);
            validDates.push(currentDate);
        }

        // chronological 80/20 split
        const split = Math.floor(sequences.length * 0.8);
        this.X_train = tf.tensor3d(sequences.slice(0, split));
        this.y_train = tf.tensor2d(targets.slice(0, split));
        this.X_test  = tf.tensor3d(sequences.slice(split));
        this.y_test  = tf.tensor2d(targets.slice(split));
        this.testDates = validDates.slice(split);

        console.log(`Created ${sequences.length} sequences`);
        console.log(`Training: ${this.X_train.shape[0]}, Test: ${this.X_test.shape[0]}`);

        return {
            X_train: this.X_train,
            y_train: this.y_train,
            X_test: this.X_test,
            y_test: this.y_test,
            symbols: this.symbols,
            testDates: this.testDates
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
