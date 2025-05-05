package MLP.Trainer;

import Function.LossFunction;
import MLP.Data.LabeledDataset;
import MLP.Data.Loaders.Dataloader;
import MLP.MLP;
import MLP.Optimizers.Optimizer;
import MLP.Regularizations.ParameterRegularization;
import Matrices.ActivationMatrix;
import MLP.Logger;

import java.util.Optional;

import static MLP.Data.Loaders.Dataloader.LabeledBatch;

/**
 * Représente l'objet permettant d'entraîner un modèle.
 * En fonction des paramètres du Trainer; l'entraînement ne se déroulera
 * pas de la même manière.
 */
public class Trainer {

    /**
     * Précise l'état du Trainer.
     * Un trainer UNCONFIGURED ne peut pas être utilisé pour l'entraînement.
     * Un trainer CONFIGURED peut être utilisé pour entraîner un modèle mais ne contient pas de
     *      résultats d'entraînement (temps, évolution du loss, etc...).
     * Un trainer POST_TRAINING contiendra ces données là après entraînement d'un modèle;
     */
    public enum TrainerState {
        UNCONFIGURED,
        CONFIGURED,
        POST_TRAINING
    }

    /**
     * Représente l'état du {@link Trainer}, voir {@link TrainerState}.
     */
    protected TrainerState state;

    // et utiliser des Optional pour les valeurs
    // après entraînement

    // TODO try-with-ressources pour tester la capacité de la ram à charger une si grosse batch
    /** Taille de batch à utiliser pour diviser les ensembles d'entraînement & de tests **/
    protected int batchSize;

    /** Nombre de fois où on va entraîner le modèle sur l'ensemble d'entraînement **/
    protected int epochs;

    /** Optimizer utilisé pour la mise à jour des poids/biais du NN dans {@link MLP#updateParameters}.**/
    protected Optimizer optimizer;

    /** Dataset scindé en train/test utilisé pour l'entraînement du NN.**/
    protected LabeledDataset dataset;

    /** Lossfunction utilisée sur la sortie du réseau**/
    protected LossFunction lossFunction;

    /**
     * Régularisation appliquée en sortie du réseau et lors du calcul des gradients (backprop)
     */
    protected ParameterRegularization parameterRegularization;

    /** TRUE Si l'entraînement communique l'évolution de sa loss au cours des epoch.
     * Dans ce cas-là, le modèle sera confronté à l'ensemble de test à la fin de chaque epoch.
     * Rend l'entraînement plus lent mais permet de suivre l'avancement. **/
    protected boolean verbose;

    /**
     * Représente les résultats d'entraînement. Sont éventuellement à null, et doivent
     * être accédés avec {@link Trainer#getOptionalResults}.
     */
    private TrainingResults trainingResults;



    /**
     * Renvoie un nouveau Trainer avec des variables par défaut
     */
    public Trainer() {
        this.state = TrainerState.UNCONFIGURED;
        this.verbose = false;
        this.epochs = 1;
        this.batchSize = 1;
        this.trainingResults = null;
    }

    public static TrainerBuilder builder() {
        return new TrainerBuilder();
    }

    public int numberOfEpochs() {
        return this.epochs;
    }

    public Dataloader getTrainingData(){
        return this.dataset.trainDataLoader;
    }

    /**
     * Entraîne le modèle en fonction du {@link Trainer}.
     **/
    public void train(MLP mlp) {

        Logger.debug("Debut de l'entraînement du mlp " + mlp + " avec l'optimizer " + this + ".");
       // if(Logger.  ;)


        long timeStartTraining = System.currentTimeMillis();

        for(int i = 0; i < this.numberOfEpochs(); i++) {

            for(int j = 0; j < this.getTrainingData().getNumberOfBatches(); j++) {
                System.out.println("Epoch n°" + i + ", Batch n°" + j);
                LabeledBatch batch = this.getTrainingData().getBatch(j);
                mlp.updateParameters(batch.getInput(), batch.getOutput(), this.getLossFunction(), this.getOptimizer(), this.parameterRegularization);
                if(verbose) {
                    Evaluation evaluation = this.evaluate(mlp);
                    System.out.print("  "); evaluation.print();

                    if(i == epochs/2 && j == 0) {

                    }
                }
            }

        }

    //    this.timeEndTraining = System.currentTimeMillis();

            Evaluation evaluation = this.evaluate(mlp);
            System.out.println("Evaluation de fin d'entraînement du modèle :");
            evaluation.print();
            System.out.println();
         //   System.out.println("Temps d'entraînement : " + this.getTrainingTime()/1000 + " secondes.");
    }

    private void print() {
        System.out.println("Optimizer : " + this.optimizer);
        System.out.println("Loss Function : " + this.lossFunction);
        System.out.println("Régularisation paramètres : " + this.parameterRegularization);

        System.out.println("Number of Epochs : " + this.numberOfEpochs());
        System.out.println("batch size : " + this.batchSize);


        System.out.println("Nombre d'items dans l'ensemble d'entraînement : " + this.getTrainingData().size);
        System.out.println("Nombre de batch : " + this.getTrainingData().getNumberOfBatches());
        System.out.println("Nombre d'items dans l'ensemble de test : " + this.getTestData().size);
        System.out.println("Nombre de batch : " + this.getTestData().getNumberOfBatches());

        System.out.println();
        System.out.println();

    }

    private Optimizer getOptimizer() {
        return this.optimizer;
    }

    private LossFunction getLossFunction() {
        return this.lossFunction;
    }

    public Evaluation evaluate(MLP mlp){

        double TP = 0;
        double TN = 0;
        double FP = 0;
        double FN = 0;

        Dataloader testData = this.getTestData();
        for(int j = 0; j < testData.getNumberOfBatches(); j++) {
            LabeledBatch batch = testData.getBatch(j);
            MLP.FeedForwardResult result = mlp.feedForward(batch.getInput());

            ActivationMatrix outputHardmaxed = result.getNetworkOutput().hardmax();


            double[][] dataPred = outputHardmaxed.getData();
            double[][] dataTheorique = batch.getOutput().getData();


            for(int i = 0; i < outputHardmaxed.getNumberOfRows(); i++){
                for(int k = 0; k < outputHardmaxed.getNumberOfColumns(); k++){
                    int dp = (int) dataPred[i][k];

                    if (dp == dataTheorique[i][k]) {
                        if(dp == 0) TN ++; else TP++;
                    }

                    else {
                        if(dp == 0) FN ++; else FP++;
                        }
                    }
            }
        }

        // Valable uniquiement pour les problèmes de type classification pour l'instant
        // TODO évaluation générale ?
        return new Evaluation(TP, FP, TN, FN, this.dataset.testDataloader.outputDimension);
    }

    public Dataloader getTestData() {
        return this.dataset.testDataloader;
    }

    /**public Optional<TrainingResults> getOptionalResults {

    }**/

    public class TrainingResults {

        /**
         * Temps de fin d'entraînement
         */
        public long timeEndTraining;

        /**
         * Temps de début d'entraînement
         */
        public long timeStartTraining;

        /**
         * Evolution du coût au cours de l'entraînement, où lossOverTime[i]
         * est le coût lors du passage du i-ème batch.
         */
        public double[] lossOverTime;


        /**
         * Renvoie le temps qui fut nécessaire pour finir l'entraînement.
         * @return
         */
        public long getTrainingTime() {
            assert(this.timeStartTraining != -1) : "Le Trainer n'a pas encore commencé son entraînement !";
            assert(this.timeEndTraining != -1) : "Le Trainer n'a pas encore fini son entraînement !";
            return this.timeEndTraining - this.timeStartTraining;
        }


    }


}



