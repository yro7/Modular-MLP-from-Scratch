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

    public BiasVector(double[][] data){
        assert(data[0].length == 1) : "Un Vecteur de Biais devrait être de largeur 1 ! Taille rentrée: "
                + data.length + " x " + data[0].length + ".";
        super(data);
    }

    public BiasVector(int rows, int numberOfNeuronsInPreviousLayer, ActivationFunction af){
        this(rows);
        this.applyToElements((i,j) -> this.getData()[i][j] = af.applyRandomBias(rows,numberOfNeuronsInPreviousLayer));
    }

    public static BiasVector createZeroBiasVector(int dimension) {
        double[][] data = new double[dimension][1];
        return new BiasVector(data);
    }

    @Override
    protected BiasVector createInstance(int rows, int cols) {
        assert(cols == 1) : "Un Vecteur de Biais devrait être de largeur 1 ! Taille rentrée: "
                + rows + " x " + cols + ".";
        return new BiasVector(rows);
    }

    public void printDimensions(String name) {
        super.printDimensions("BiasVector", name);
    }


}

