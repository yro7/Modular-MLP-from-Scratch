package Matrices;

import Function.ActivationFunction;
import Function.InitializationFunction;

/**
 * Représente un vecteur de biais d'une {@link Layer} du {@link MLP}.
 *
 */
public class BiasVector extends Matrix<BiasVector> {

    public BiasVector(int rows) {
        super(rows, 1);

    }

    public BiasVector(int rows, int numberOfNeuronsInPreviousLayer, ActivationFunction af){
        this(rows);
        this.applyToElements((i,j) -> this.getData()[i][j] = af.applyRandomBias(rows,numberOfNeuronsInPreviousLayer));
    }

    @Override
    protected BiasVector createInstance(int rows, int cols) {
        return new BiasVector(rows);
    }
}
