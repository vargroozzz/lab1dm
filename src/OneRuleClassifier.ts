type TrainingData = {
    features: Record<number, number>;
    label: number;
}

export class OneRuleClassifier {
    private bestRule: { feature: number; value: number } | null = null;
    private bestErrorRate: number = Infinity;

    train(trainingData: TrainingData[]) {
        const featureValues: Record<number, Set<number>> = {};

        for (const { features, label } of trainingData) {
            for (const [feature, value] of Object.entries(features)) {
                if (!featureValues[feature]) {
                    featureValues[feature] = new Set<number>();
                }
                featureValues[feature].add(value);
            }
        }

        for (const [feature, values] of Object.entries(featureValues)) {
            for (const value of values) {
                const rule = { feature: Number(feature), value };
                const errorRate = this.calculateErrorRate(trainingData, rule);

                if (errorRate < this.bestErrorRate) {
                    this.bestErrorRate = errorRate;
                    this.bestRule = rule;
                }
            }
        }
    }

    private calculateErrorRate(trainingData: TrainingData[], rule: { feature: number; value: number }): number {
        let errorCount = 0;

        for (const { features, label } of trainingData) {
            if (features[rule.feature] !== rule.value && label !== this.mostCommonLabel(trainingData, rule)) {
                errorCount++;
            }
        }

        return errorCount / trainingData.length;
    }

    private mostCommonLabel(trainingData: TrainingData[], rule: { feature: number; value: number }): number {
        const labelCounts: Record<number, number> = {};

        for (const { features, label } of trainingData) {
            if (features[rule.feature] === rule.value) {
                labelCounts[label] = (labelCounts[label] || 0) + 1;
            }
        }

        return Object.entries(labelCounts).reduce((mostCommonLabel, [label, count]) =>
                count > labelCounts[mostCommonLabel] ? Number(label) : mostCommonLabel
            , 0);
    }

    classify(features: Record<number, number>): number {
        if (!this.bestRule) {
            throw new Error('Classifier has not been trained');
        }

        const { feature, value } = this.bestRule;
        return features[feature] === value ? this.mostCommonLabel([{ features, label: 0 }], this.bestRule) : 0;
    }
}
