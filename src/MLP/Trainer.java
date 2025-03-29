package MLP;

import Function.LossFunction;
import MLP.Data.LabeledDataset;
import MLP.Data.LabeledTestData;
import MLP.Data.LabeledTrainingDataset;
import static MLP.Data.LabeledDataset.LabeledBatch;

/**
 * Représente l'objet permettant d'entraîner un modèle.
 * En fonction des paramètres du Trainer; l'entraînement ne se déroulera
 * pas de la même manière.
 */
public class Trainer {

    /** Taille de batch à utiliser pour diviser l'ensemble d'entraînement **/
    public int batchSize;

    /** Nombre de fois où on va entraîner le modèle sur l'ensemble d'entraînement **/
    public int epochs;

    public Optimizer optimizer;
    public LabeledTestData testData;
    public LabeledTrainingDataset labeledTrainingDataset;
    public LossFunction lossFunction;

    /** TRUE Si l'entraînement communique l'évolution de sa loss au cours des epoch.
     * Dans ce cas-là, le modèle sera confronté à l'ensemble de test à la fin de chaque epoch.
     * Rend l'entraînement plus lent mais permet de suivre l'avancement. **/
    public boolean verbose;

    /**
     * Renvoie un nouveau Trainer avec des variables par défaut
     */
    public Trainer() {
        this.verbose = false;
        this.epochs = 1;
        this.batchSize = 1;
    }

    public static TrainerBuilder builder() {
        return new TrainerBuilder();
    }

    public int numberOfEpochs() {
        return this.epochs;
    }

    public LabeledTrainingDataset getTrainingData(){
        return this.labeledTrainingDataset;
    }

    /**
     * Entraîne le modèle en fonction du {@link Trainer}.
     **/
    public void train(MLP mlp) {
        for(int i = 0; i < this.numberOfEpochs(); i++) {
            LabeledBatch batch = this.getTrainingData().getBatch(i);
            mlp.updateParameters(batch.getInput(), batch.getOutput(), this.lossFunction);

            if(verbose) {
                Evaluation evaluation = this.evaluate(mlp);
                evaluation.print();
            }

        }
    }

    public Evaluation evaluate(MLP mlp){
        Evaluation res = new Evaluation();
        return res;
    }

    /**
     * Représente le résultat d'une évaluation d'un {@link MLP}
     * contre un ensemble de données de test.
     */
    public static class Evaluation {

        public void print(){
            System.out.println("blablabla print eval etc");
        }
    }
}



