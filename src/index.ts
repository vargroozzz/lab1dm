import * as console from "node:console";
import {DecisionTreeClassifier} from "./DecisionTreeClassifier.js";
import {NaiveBayesClassifier} from "./NaiveBayesClassifier.js";
import {OneRuleClassifier} from "./OneRuleClassifier.js";
import {KNNClassifier} from "./KNNClassifier.js";



const trainingData = [
    {
        features:[0, 0, 0, 0],
        label: 1
    },{
        features:[0, 0, 0, 1],
        label: 0
    },{
        features:[0, 0, 1, 0],
        label: 1
    },{
        features:[0, 0, 1, 1],
        label: 1
    },{
        features:[0, 1, 0, 0],
        label: 0
    },{
        features:[0, 1, 0, 1],
        label: 0
    },{
        features:[0, 1, 1, 0],
        label: 1
    },{
        features:[0, 1, 1, 1],
        label: 1
    },{
        features:[1, 0, 0, 0],
        label: 0
    },{
        features:[1, 0, 1, 0],
        label: 0
    },
];

const nbData = trainingData.map(({ features, label }) => ({text: features.join(' '), label}));
const orcData = trainingData.map(
    ({ features, label }) => (
        {features: features.reduce(
            (acc, value) => (
                {...acc, [value]: Number.isFinite(acc[value]) ? acc[value] + 1 : 0}), {}), label}
    )
);

const classifyData = [1, 1, 1, 1]

const dtc = new DecisionTreeClassifier(trainingData);
const nbc = new NaiveBayesClassifier()
const orc = new OneRuleClassifier()
const knnc = new KNNClassifier(trainingData)

nbc.train(nbData)
orc.train(orcData)

console.log("DecisionTreeClassifier:", dtc.classify(classifyData));
console.log("NaiveBayesClassifier:", nbc.classify(classifyData.join(' ')));
console.log("OneRuleClassifier:", orc.classify(classifyData.reduce(
    (acc, value) => (
        {...acc, [value]: Number.isFinite(acc[value]) ? acc[value] + 1 : 0}), {})));
console.log("KNNClassifier:", knnc.classify(classifyData, 3));