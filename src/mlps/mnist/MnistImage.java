package mlps.mnist;

import matrices.Matrix;

/**
 * Représente une image MNIST de taille 28 x 28 pixels
 */
public class MnistImage extends Matrix<MnistImage> {

    public MnistImage() {
        super(28, 28);
    }

    @Override
    protected MnistImage createInstance(int rows, int cols) {
        return new MnistImage();
    }
}
