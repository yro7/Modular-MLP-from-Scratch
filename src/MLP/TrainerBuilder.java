package MLP;
import Function.LossFunction;
import MLP.Data.LabeledDataset;
import MLP.Optimizers.Optimizer;

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

    public TrainerBuilder setDataset(LabeledDataset dataset) {
        this.product.dataset = dataset;
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
        this.product.dataset.testDataloader.batchSize = batchSize;
        this.product.dataset.trainDataLoader.batchSize = batchSize;
        this.product.batchSize = batchSize;
        return this;
    }



    public TrainerBuilder setVerbose() {
        this.product.verbose = true;
        return this;
    }

    public Trainer build(){
        assert (this.hasLossFunction()) : "Le Trainer doit avoir une fonction de coût !";
        assert (this.hasDataset()) : "Le Trainer doit avoir dataset !";
        assert (this.hasOptimizer()) : "Le Trainer doit avoir un Optimizer !";

        return this.product;
    }

    private boolean hasLossFunction() {
        return (this.product().lossFunction != null);
    }


    private boolean hasDataset() {
        return (this.product.dataset != null);
    }

    private boolean hasOptimizer() {
        return (this.product.optimizer != null);
    }





}
