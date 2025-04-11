package MLP;

import Function.LossFunction;
import MLP.Data.LabeledDataset;
import MLP.Data.Loaders.Dataloader;
import MLP.Optimizers.Optimizer;
import Matrices.ActivationMatrix;

import static MLP.Data.Loaders.Dataloader.LabeledBatch;

/**
 * Représente l'objet permettant d'entraîner un modèle.
 * En fonction des paramètres du Trainer; l'entraînement ne se déroulera
 * pas de la même manière.
 */
public class Trainer {

    // TODO try-with-ressources pour tester la capacité de la ram à charger une si grosse batch
    /** Taille de batch à utiliser pour diviser les ensembles d'entraînement & de tests **/
    public int batchSize;

    /** Nombre de fois où on va entraîner le modèle sur l'ensemble d'entraînement **/
    public int epochs;

    /** Optimizer utilisé pour la mise à jour des poids/biais du NN dans {@link MLP#updateParameters}.**/
    public Optimizer optimizer;

    /** Dataset scindé en train/test utilisé pour l'entraînement du NN.**/
    public LabeledDataset dataset;

    /** Lossfunction utilisée sur la sortie du réseau**/
    public LossFunction lossFunction;

    /** TRUE Si l'entraînement communique l'évolution de sa loss au cours des epoch.
     * Dans ce cas-là, le modèle sera confronté à l'ensemble de test à la fin de chaque epoch.
     * Rend l'entraînement plus lent mais permet de suivre l'avancement. **/
    public boolean verbose;


    long timeStartTraining;
    long timeEndTraining;

    /**
     * Renvoie un nouveau Trainer avec des variables par défaut
     */
    public Trainer() {
        this.verbose = false;
        this.epochs = 1;
        this.batchSize = 1;
        this.timeEndTraining = -1;
        this.timeStartTraining = -1;
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
        System.out.println("Début de l'entraînement du MLP.");
        System.out.println("Number of Epochs : " + this.numberOfEpochs());
        System.out.println("batch size : " + this.batchSize);


        System.out.println("Nombre d'items dans l'ensemble d'entraînement : " + this.getTrainingData().size);
        System.out.println("Nombre de batch : " + this.getTrainingData().getNumberOfBatches());
        System.out.println("Nombre d'items dans l'ensemble de test : " + this.getTestData().size);
        System.out.println("Nombre de batch : " + this.getTestData().getNumberOfBatches());

        System.out.println();
        System.out.println();

        this.timeStartTraining = System.currentTimeMillis();

        for(int i = 0; i < this.numberOfEpochs(); i++) {

            for(int j = 0; j < this.getTrainingData().getNumberOfBatches(); j++) {
                System.out.println("Epoch n°" + i + ", Batch n°" + j);
                LabeledBatch batch = this.getTrainingData().getBatch(j);
                mlp.updateParameters(batch.getInput(), batch.getOutput(), this.getLossFunction(), this.getOptimizer());
                if(verbose) {
                    Evaluation evaluation = this.evaluate(mlp);
                    System.out.print("  "); evaluation.print();

                    if(i == epochs/2 && j == 0) {

                    }
                }
            }

        }

        this.timeEndTraining = System.currentTimeMillis();

            Evaluation evaluation = this.evaluate(mlp);
            System.out.println("Evaluation de fin d'entraînement du modèle :");
            evaluation.print();
            System.out.println();
            System.out.println("Temps d'entraînement : " + this.getTrainingTime()/1000 + " secondes.");
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

        return new Evaluation(TP, FP, TN, FN);
    }

    public Dataloader getTestData() {
        return this.dataset.testDataloader;
    }

    /**
     * Représente le résultat d'une évaluation d'un {@link MLP}
     * contre un ensemble de données de test.
     */
    public static class Evaluation {

        double TP;
        double FP;
        double TN;
        double FN;

        public Evaluation(double TP, double FP, double TN, double FN) {
            this.TP = TP;
            this.FP = FP;
            this.TN = TN;
            this.FN = FN;
        }

        public void print(){

            double size = TP + TN + FP + FN;

            double accuracy = (TP + TN) /(size);
            double precision = TP /(TP + FP);
            double recall = TP /(TP + FN);
            double f1Score = 2*(precision*recall)/(precision+recall);


            System.out.println("Accuracy : " + accuracy);
            System.out.println("Precision : " + precision);
            System.out.println("Recall : " + recall);
            System.out.println("f1Score : " + f1Score);
            System.out.println();

            ;
            System.out.println("Nombre de prédictions correctes : " + this.TP + "/" + size/10);
        }
    }

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



