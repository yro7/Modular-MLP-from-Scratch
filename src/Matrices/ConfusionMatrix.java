package Matrices;

public class ConfusionMatrix extends Matrix<ConfusionMatrix> {

    public ConfusionMatrix(int rows, int cols) {
        assert(rows == 2 & cols == 2) : "Une matrice de confusion devrait être de taille 2 x 2 !";
        super(rows, cols);
    }

    @Override
    protected ConfusionMatrix createInstance(int rows, int cols) {
        return new ConfusionMatrix(2, 2);
    }
}
