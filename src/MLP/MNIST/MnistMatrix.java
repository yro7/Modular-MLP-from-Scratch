package MLP.MNIST;

import Matrices.ActivationMatrix;

/**
 * Représente une matrice de vecteurs Mnist, de taille
 * (taille du batch x 784).
 * Contient aussi le label de l'image, càd s'il s'agit d'un 1, 2, 0...
 */

public class MnistMatrix extends ActivationMatrix {

    private int label;

    public MnistMatrix(int rows) {
        super(rows, 784);
    }

    public double getValue(int r, int c) {
        return this.getData()[r][c];
    }

    public int getLabel() {
        return label;
    }


}