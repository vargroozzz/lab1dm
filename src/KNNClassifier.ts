interface TrainingData {
    features: number[];
    label: number;
}

export class KNNClassifier {
    private trainingData: TrainingData[];

    constructor(trainingData: TrainingData[]) {
        this.trainingData = trainingData;
    }

    classify(features: number[], k: number): number {
        const nearestNeighbors = this.getNearestNeighbors(features, k);
        const labelCounts = this.countLabels(nearestNeighbors);
        return this.getMostCommonLabel(labelCounts);
    }

    private getNearestNeighbors(features: number[], k: number): TrainingData[] {
        return this.trainingData
            .map(trainingExample => ({
                ...trainingExample,
                distance: this.euclideanDistance(features, trainingExample.features)
            }))
            .sort((a, b) => a.distance - b.distance)
            .slice(0, k);
    }

    private countLabels(neighbors: TrainingData[]): Record<number, number> {
        return neighbors.reduce((labelCounts, { label }) => {
            labelCounts[label] = (labelCounts[label] || 0) + 1;
            return labelCounts;
        }, {} as Record<number, number>);
    }

    private getMostCommonLabel(labelCounts: Record<number, number>): number {
        return Object.entries(labelCounts).reduce((mostCommonLabel, [label, count]) =>
                count > labelCounts[mostCommonLabel] ? Number(label) : mostCommonLabel
            , 0);
    }

    private euclideanDistance(features1: number[], features2: number[]): number {
        if (features1.length !== features2.length) {
            throw new Error('Features have different lengths');
        }

        const sumOfSquaredDifferences = features1.reduce((sum, feature1, index) => {
            const feature2 = features2[index];
            return sum + (feature1 - feature2) ** 2;
        }, 0);

        return Math.sqrt(sumOfSquaredDifferences);
    }
}
