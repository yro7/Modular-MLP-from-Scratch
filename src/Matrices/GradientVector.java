package Matrices;

import Function.ActivationFunction;
import Function.InitializationFunction;

/**
 * Représente un vecteur de gradient d'une {@link Layer} du {@link MLP} lors de la rétropropagation.
 *
 */
public class GradientVector extends Matrix<GradientVector> {

    public GradientVector(int rows) {
        super(rows, 1);

    }

    public GradientVector(int rows, int numberOfNeuronsInPreviousLayer, ActivationFunction af){
        this(rows);
        this.applyToElements((i,j) -> this.getData()[i][j] = af.applyRandomBias(rows,numberOfNeuronsInPreviousLayer));
    }

    @Override
    protected GradientVector createInstance(int rows, int cols) {
        return new GradientVector(rows);
    }
}
