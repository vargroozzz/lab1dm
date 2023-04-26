type TrainingData = {
    label: number;
    text: string;
}

export class NaiveBayesClassifier {
    private vocabulary: Set<string> = new Set<string>();
    private classCounts = new Map<number, number>();
    private wordCounts= new Map<number, Map<string, number>>();
    private totalCount: number = 0;

    train(trainingData: TrainingData[]) {
        for (const { label, text } of trainingData) {
            this.classCounts.set(label, (this.classCounts.get(label) || 0) + 1);
            this.totalCount++;

            const words = text.split(' ');
            for (const word of words) {
                this.vocabulary.add(word);

                const wordCountMap = this.wordCounts.get(label) || new Map<string, number>();
                wordCountMap.set(word, (wordCountMap.get(word) || 0) + 1);
                this.wordCounts.set(label, wordCountMap);
            }
        }
    }

    classify(text: string): string {
        const words = text.split(' ');

        let maxProb = -Infinity;
        let maxLabel = this.classCounts.keys()?.[0] ?? 0;
        for (const [label, count] of this.classCounts) {
            let logProb = Math.log(count / this.totalCount);

            let wordCountMap = this.wordCounts.get(label);
            if (!wordCountMap) {
                wordCountMap = new Map<string, number>();
            }

            for (const word of words) {
                const wordCount = wordCountMap.get(word) || 0;
                const wordProb = (wordCount + 1) / (count + this.vocabulary.size);
                logProb += Math.log(wordProb);
            }

            if (logProb > maxProb) {
                maxProb = logProb;
                maxLabel = label;
            }
        }

        return maxLabel;
    }
}
