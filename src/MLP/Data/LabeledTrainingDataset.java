package MLP.Data;

import Matrices.ActivationMatrix;

import java.io.IOException;
import java.util.List;

/**
 * Représente un ensemble de données d'entraînement.
 * Ces données seront fournies au modèle lors de son entraînement.
 * Les données sont organisées sous forme de batch, où chaque batch
 * est représenté par une {@link ActivationMatrix}.
 */
public abstract class LabeledTrainingDataset<InputType,OutputType> extends LabeledDataset<InputType, OutputType> {

    int batchSize;

    public LabeledTrainingDataset(int batchSize, int size, int inputDimension, int outputDimension,
                                  String path, String labelPath) {
        this(size, inputDimension, outputDimension, path, labelPath);
        this.batchSize = batchSize;

    }

    /**
     * Récupère un tableau d'entrées et le met sous forme de batch.
     * Chaque entrée sera vectorisée et formera une ligne du batch résultant.
     * @param objects
     * @return
     */
    public ActivationMatrix objects_To_Batch(List<InputType> objects) {
        double[] firstItem = this.vectorizeInput(objects.getFirst());
        int itemSizes = firstItem.length;
        int batchSize = objects.size();
        double[][] data = new double[batchSize][itemSizes];
        for(int i = 0; i < batchSize; i++) {
            data[i] = this.vectorizeInput(objects.get(i));
        }

        return new ActivationMatrix(data);
    }

    /**
     * Charge en mémoire et renvoie la batch numéro i du {@link LabeledTrainingDataset},
     * sous forme de couple (Entrée, Sortie Attendue)
     * @param batchNumber
     * @return
     */
    public LabeledBatch getBatch(int batchNumber) {
        assert(batchNumber < this.getNumberOfBatches()) : "Le numéro de la batch doit être inférieur au nombre de batch du dataset !";

        int begin = batchNumber * batchSize;
        int end = (batchNumber + 1) * batchSize;

        List<LabeledDataSample<InputType, OutputType>> objects = loadList(begin, end);
        double[] firstItem = this.vectorizeInput(objects.getFirst().getInput());

        ActivationMatrix batch = new ActivationMatrix(batchSize, firstItem.length);
        ActivationMatrix labels = new ActivationMatrix(batchSize, outputDimension);

        for(int i = 0; i < batchSize; i++) {
            LabeledDataSample<InputType, OutputType> object = objects.get(i);

            batch.getData()[i] = this.vectorizeInput(object.getInput());
            labels.getData()[i] = this.vectorizeOutput(object.getOutput());
        }

        return new LabeledBatch(batch, labels);
    }


    /**
     * Renvoie le nombre de batch dans l'ensemble d'entraînement,
     * c'est à dire la taille de l'ensemble divisé par la taille de chaque batch.
     * @return
     */
    public int getNumberOfBatches() {
        return this.size / batchSize;
    }


}
