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
        // track how many features per stock we expose to the model. The original
        // implementation only used Open and Close (2 features). We will later
        // populate this value based on the normalized feature keys. This is
        // consumed by the app when determining the input shape for the GRU.
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

        // Parse all rows
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

        // Calculate min and max per stock for each feature. In addition to
        // Open and Close we now also include High, Low, Volume and a derived
        // Return feature ((Close - Open) / Open). Performing per-stock
        // normalisation helps the model learn relative movements while
        // preserving relationships across stocks.
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
                const point = this.stocksData[symbol][date];
                if (point) {
                    // update raw min/max for direct price/volume features
                    minMax[symbol].Open.min = Math.min(minMax[symbol].Open.min, point.Open);
                    minMax[symbol].Open.max = Math.max(minMax[symbol].Open.max, point.Open);
                    minMax[symbol].Close.min = Math.min(minMax[symbol].Close.min, point.Close);
                    minMax[symbol].Close.max = Math.max(minMax[symbol].Close.max, point.Close);
                    minMax[symbol].High.min = Math.min(minMax[symbol].High.min, point.High);
                    minMax[symbol].High.max = Math.max(minMax[symbol].High.max, point.High);
                    minMax[symbol].Low.min = Math.min(minMax[symbol].Low.min, point.Low);
                    minMax[symbol].Low.max = Math.max(minMax[symbol].Low.max, point.Low);
                    minMax[symbol].Volume.min = Math.min(minMax[symbol].Volume.min, point.Volume);
                    minMax[symbol].Volume.max = Math.max(minMax[symbol].Volume.max, point.Volume);

                    // compute return value for min/max tracking. We protect
                    // against division by zero by checking Open > 0. If Open is
                    // 0 (unlikely in real stock data), we set the return to 0.
                    const ret = point.Open !== 0 ? (point.Close - point.Open) / point.Open : 0;
                    minMax[symbol].Return.min = Math.min(minMax[symbol].Return.min, ret);
                    minMax[symbol].Return.max = Math.max(minMax[symbol].Return.max, ret);
                }
            });
        });

        // Normalize data. We map each feature into [0,1]. For returns,
        // negative values are handled by the min/max range; features with
        // constant values (max == min) are set to 0 to avoid NaN.
        this.symbols.forEach(symbol => {
            this.normalizedData[symbol] = {};
            this.dates.forEach(date => {
                const point = this.stocksData[symbol][date];
                if (point) {
                    const ret = point.Open !== 0 ? (point.Close - point.Open) / point.Open : 0;
                    const norm = {};
                    ['Open', 'Close', 'High', 'Low', 'Volume'].forEach(key => {
                        const denom = minMax[symbol][key].max - minMax[symbol][key].min;
                        norm[key] = denom === 0 ? 0 : (point[key] - minMax[symbol][key].min) / denom;
                    });
                    // normalise return separately
                    const denomReturn = minMax[symbol].Return.max - minMax[symbol].Return.min;
                    norm.Return = denomReturn === 0 ? 0 : (ret - minMax[symbol].Return.min) / denomReturn;

                    this.normalizedData[symbol][date] = norm;
                }
            });
        });

        // update feature count for downstream modules
        this.numFeaturesPerStock = Object.keys(this.normalizedData[this.symbols[0]][this.dates[0]]).length;

        return this.normalizedData;
    }

    createSequences(sequenceLength = 12, predictionHorizon = 3) {
        // Ensure normalization has been executed so that this.normalizedData and
        // this.numFeaturesPerStock are populated. If normalizeData() hasn't
        // been called yet then it will be invoked here.
        if (!this.normalizedData) this.normalizeData();

        const sequences = [];
        const targets = [];
        const validDates = [];

        // Iterate over all possible sequence starting points. We stop
        // predictionHorizon days before the end of the time series to ensure
        // future target dates exist.
        for (let i = sequenceLength; i < this.dates.length - predictionHorizon; i++) {
            const currentDate = this.dates[i];
            const sequenceData = [];
            let validSequence = true;

            // Build a sequence of length `sequenceLength` for each time step. We
            // iterate backwards from the current index so that the earliest
            // data point appears first in the resulting sequence array.
            for (let j = sequenceLength - 1; j >= 0; j--) {
                const seqDate = this.dates[i - j];
                const timeStepData = [];

                this.symbols.forEach(symbol => {
                    const normPoint = this.normalizedData[symbol][seqDate];
                    if (normPoint) {
                        // Append all available features for this stock at this
                        // date. The order here must match the order used in
                        // normalization above (Open, Close, High, Low, Volume, Return).
                        timeStepData.push(
                            normPoint.Open,
                            normPoint.Close,
                            normPoint.High,
                            normPoint.Low,
                            normPoint.Volume,
                            normPoint.Return
                        );
                    } else {
                        validSequence = false;
                    }
                });

                if (validSequence) sequenceData.push(timeStepData);
            }

            // Create target labels only if the sequence was valid. We compare
            // the close price on the currentDate with the close price at
            // future offsets to generate binary labels per stock.
            if (validSequence) {
                const target = [];
                const baseClosePrices = [];
                this.symbols.forEach(symbol => {
                    baseClosePrices.push(this.stocksData[symbol][currentDate].Close);
                });

                for (let offset = 1; offset <= predictionHorizon; offset++) {
                    const futureDate = this.dates[i + offset];
                    this.symbols.forEach((symbol, idx) => {
                        const futureClose = this.stocksData[symbol][futureDate]?.Close;
                        if (futureClose !== undefined) {
                            target.push(futureClose > baseClosePrices[idx] ? 1 : 0);
                        } else {
                            validSequence = false;
                        }
                    });
                }

                if (validSequence) {
                    sequences.push(sequenceData);
                    targets.push(target);
                    validDates.push(currentDate);
                }
            }
        }

        // Split into train/test sets chronologically. A typical 80/20 split is
        // used to respect temporal ordering and prevent data leakage.
        const splitIndex = Math.floor(sequences.length * 0.8);
        this.X_train = tf.tensor3d(sequences.slice(0, splitIndex));
        this.y_train = tf.tensor2d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(sequences.slice(splitIndex));
        this.y_test = tf.tensor2d(targets.slice(splitIndex));
        this.testDates = validDates.slice(splitIndex);

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
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}

export default DataLoader;
