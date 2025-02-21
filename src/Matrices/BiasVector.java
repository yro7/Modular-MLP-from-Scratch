package Matrices;

/**
 * Représente un vecteur de biais d'une {@link Layer} du {@link MLP}.
 *
 */
public class BiasVector extends Matrix<BiasVector> {

    public BiasVector(int rows) {
        super(rows, 1);

    }

    @Override
    protected BiasVector createInstance(int rows, int cols) {
        return new BiasVector(rows);
    }
}
