package MLP.MNIST;

import Matrices.ActivationMatrix;

/**
 * Représente un vecteur MNIST (un vecteur de 784 éléments qui représente
 * une image MNIST de 28x28 applatie).
 * Contient aussi le label de l'image, càd s'il s'agit d'un 1, 2, 0...
 */
public class MnistVector extends ActivationMatrix {

    private int label;

    public MnistVector() {
        super(1, 784); // Crée une matrice 784x1 (vecteur en ligne)
    }

    public double getValue(int idx) {
        return this.getData()[0][idx];
    }

    public void setValue(int idx, double value) {
        this.getData()[0][idx] = value;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public void setValues(double[] rowVector) {
        this.getData()[0] = rowVector;
    }
}