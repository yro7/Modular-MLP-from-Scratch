package MLP.MNIST;

import Matrices.ActivationMatrix;

/**
 * Représente une liste de vecteurs MNIST (des vecteurs de 784 lignes, de la
 * largeur du batch, qui représentent une image MNIST de 28x28 applatie).
 * Contient aussi le label de l'image, càd s'il s'agit d'un 1, 2, 0...
 */

public class MnistMatrix extends ActivationMatrix {

    private int label;

    public MnistMatrix(int nRows, int nCols) {
        super(nRows, nCols);
    }

    public double getValue(int r, int c) {
        return this.getData()[r][c];
    }

    public void setValue(int row, int col, double value) {
        this.getData()[row][col] = value;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

}