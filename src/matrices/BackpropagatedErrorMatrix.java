package matrices;


/**
 * Représente une {@link Matrix} de terme d'une {@link Layer} d'un {@link mlps}.
 *
 * Chaque neuronne est représentée par une ligne de la matrice.
 * La colonne i d'une matrice de poids représente les poids de chaque neuronne de la couche actuelle avec
 * le neuronne i de la couche précédente.
 */
public class BackpropagatedErrorMatrix extends Matrix<BackpropagatedErrorMatrix> {
    @Override
    protected BackpropagatedErrorMatrix createInstance(int rows, int cols) {
        return new BackpropagatedErrorMatrix(rows, cols);
    }

    public BackpropagatedErrorMatrix(int rows, int cols) {
        super(rows, cols);
    }
}
