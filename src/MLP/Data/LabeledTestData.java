package MLP.Data;

import MLP.Pair;
import Matrices.ActivationMatrix;

/**
 * Représente un ensemble de données de test.
 * Après entrainement, le modèle sera confronté à cet ensemble
 * pour évaluer sa précision.
 */
public abstract class LabeledTestData<InputType, OutputType> extends LabeledDataset<InputType, OutputType> {


    public LabeledTestData(int size, int inputDimension, int outputDimension, String path, String labelPath) {
        super(size, inputDimension, outputDimension, path, labelPath);
    }
}
