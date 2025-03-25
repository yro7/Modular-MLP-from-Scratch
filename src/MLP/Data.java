package MLP;

import Matrices.ActivationMatrix;

import java.util.List;

/**
 * Représente un ensemble de données, soit pour l'entraînement ({@link TrainingData})
 * soit pour l'évaluation du model ({@link TestData}).
 */
public class Data {

    List<ActivationMatrix> dataList;
}
