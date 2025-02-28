package Matrices;

public class GradientMatrix extends Matrix<GradientMatrix> {

    public GradientMatrix(int rows, int cols) {
        super(rows, cols);
    }

    public GradientMatrix(double[][] data) {
        super(data);
    }

    @Override
    protected GradientMatrix createInstance(int rows, int cols) {
        return new GradientMatrix(rows, cols);
    }
}
