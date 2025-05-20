package mlps.trainer;
import functions.LossFunction;
import mlps.data.LabeledDataset;
import mlps.Logger;
import mlps.optimizers.Optimizer;
import mlps.regularizations.ParameterRegularization;

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

    public TrainerBuilder setParameterRegularization(ParameterRegularization parameterRegularization) {
        this.product.parameterRegularization = parameterRegularization;
        return this;
    }

    public TrainerBuilder setBatchSize(int batchSize){
        this.product.dataset.testDataloader.batchSize = batchSize;
        this.product.dataset.trainDataLoader.batchSize = batchSize;
        this.product.batchSize = batchSize;
        return this;
    }


    /**
     * TRUE Si l'entraînement communique l'évolution de sa loss au cours des epoch.
     * Dans ce cas-là, le modèle sera confronté à l'ensemble de test à la fin de chaque epoch.
     * Rend l'entraînement plus lent mais permet de suivre l'avancement
     * @return
     */
    public TrainerBuilder setVerbose() {
        this.product.verbose = true;
        return this;
    }

    public Trainer build(){

        // Vérification des paramètres essentiels
        assert (this.hasLossFunction()) : "Le Trainer doit avoir une fonction de coût !";
        assert (this.hasDataset()) : "Le Trainer doit avoir un ataset !";
        assert (this.hasOptimizer()) : "Le Trainer doit avoir un Optimizer !";

        //
        if(this.product.batchSize == 1 || this.product.epochs == 1) {
            Logger.warn("Warn optimizer : batchSize ou epochs définies à seulement 1.");
        }

        if(!this.hasParameterRegularization()) {
            Logger.warn("Warn optimizer : aucune régularisation des paramètres définie.");
        }


        this.product.state = Trainer.TrainerState.CONFIGURED;
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

    private boolean hasParameterRegularization() {
        return this.product.parameterRegularization != null;
    }



}
