interface TrainingData {
    features: number[];
    label: number;
}

interface Node {
    featureIndex?: number;
    threshold?: number;
    label?: number;
    left?: Node;
    right?: Node;
}

export class DecisionTreeClassifier {
    private readonly root: Node;

    constructor(trainingData: TrainingData[]) {
        this.root = this.buildTree(trainingData);
    }

    classify(features: number[]): number {
        let node = this.root;
        while (node.featureIndex !== undefined) {
            if (features[node.featureIndex] < node.threshold!) {
                node = node.left!;
            } else {
                node = node.right!;
            }
        }
        return node.label!;
    }

    private buildTree(data: TrainingData[]): Node {
        const labels = Array.from(new Set(data.map(({ label }) => label)));
        if (labels.length === 1) {
            return { label: labels[0] };
        }
        if (data.length === 0) {
            throw new Error('No data provided');
        }

        const bestSplit = this.getBestSplit(data);
        const leftData = data.filter(({ features }) => features[bestSplit.featureIndex!] < bestSplit.threshold!);
        const rightData = data.filter(({ features }) => features[bestSplit.featureIndex!] >= bestSplit.threshold!);

        const left = this.buildTree(leftData);
        const right = this.buildTree(rightData);

        return {
            featureIndex: bestSplit.featureIndex,
            threshold: bestSplit.threshold,
            left,
            right
        };
    }

    private getBestSplit(data: TrainingData[]): { featureIndex: number, threshold: number } {
        let bestSplit: { featureIndex: number, threshold: number } = { featureIndex: -1, threshold: 0 };
        let bestScore = -Infinity;

        for (let featureIndex = 0; featureIndex < data[0].features.length; featureIndex++) {
            const featureValues = data.map(({ features }) => features[featureIndex]);
            const uniqueValues = Array.from(new Set(featureValues));
            for (let threshold of uniqueValues) {
                const leftData = data.filter(({ features }) => features[featureIndex] < threshold);
                const rightData = data.filter(({ features }) => features[featureIndex] >= threshold);
                const score = this.calculateScore(leftData, rightData);
                if (score > bestScore) {
                    bestScore = score;
                    bestSplit = { featureIndex, threshold };
                }
            }
        }

        return bestSplit;
    }

    private calculateScore(leftData: TrainingData[], rightData: TrainingData[]): number {
        const leftCounts = this.countLabels(leftData);
        const rightCounts = this.countLabels(rightData);
        const totalCount = leftData.length + rightData.length;

        let score = 0;
        for (let counts of [leftCounts, rightCounts]) {
            let entropy = 0;
            for (let labelCount of Object.values(counts)) {
                const p = labelCount / Object.values(counts).reduce((a, b) => a + b);
                entropy -= p * Math.log2(p);
            }
            score += Object.values(counts).reduce((a, b) => a + b, 0) / totalCount * entropy;
        }

        return -score;
    }

    private countLabels(data: TrainingData[]): Record<string, number> {
        return data.reduce((labelCounts, { label }) => {
            labelCounts[label] = (labelCounts[label] || 0) + 1;
            return labelCounts;
        }, {});
    }
}
