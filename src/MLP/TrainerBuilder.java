package MLP;

import Function.LossFunction;
import MLP.Data.LabeledTestData;
import MLP.Data.LabeledTrainingDataset;

/**
 * Permet de créer facilement un Trainer.
 */
public class TrainerBuilder {

    public Trainer product;

    public TrainerBuilder() {
        this.product = new Trainer();
    }

    public Trainer product() {
        return this.product;
    }

    public TrainerBuilder setLossFunction(LossFunction lossFunction) {
        this.product.lossFunction = lossFunction;
        return this;
    }

    public TrainerBuilder setTestData(LabeledTestData testData){
        this.product.testData = testData;
        return this;
    }

    public TrainerBuilder setTrainingData(LabeledTrainingDataset labeledTrainingDataset) {
        this.product.labeledTrainingDataset = labeledTrainingDataset;
        return this;

    }

    public TrainerBuilder setOptimizer(Optimizer optimizer){
        this.product.optimizer = optimizer;
        return this;
    }

    public TrainerBuilder setEpoch(int numberOfEpochs){
        this.product.epochs = numberOfEpochs;
        return this;
    }

    public TrainerBuilder setBatchSize(int batchSize){
        this.product.batchSize = batchSize;
        return this;
    }



    public TrainerBuilder setVerbose() {
        this.product.verbose = true;
        return this;
    }

    public Trainer build(){
        assert (this.hasLossFunction()) : "Le Trainer doit avoir une fonction de coût !";
        assert (this.hasTestData()) : "Le Trainer doit avoir un ensemble de test !";
        assert (this.hasTrainData()) : "Le Trainer doit avoir un ensemble d'entraînement !";
        assert (this.hasOptimizer()) : "Le Trainer doit avoir un Optimizer !";

        return this.product;
    }

    private boolean hasLossFunction() {
        return (this.product().lossFunction != null);
    }

    private boolean hasTrainData() {
        return (this.product().labeledTrainingDataset != null);
    }

    private boolean hasTestData() {
        return (this.product.testData != null);
    }

    private boolean hasOptimizer() {
        return (this.product.optimizer != null);
    }





}
