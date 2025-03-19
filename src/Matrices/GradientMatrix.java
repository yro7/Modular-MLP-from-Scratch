package Matrices;


/**
 * Représente un gradient de poids pour la mise à jour des poids des couches du réseau.
 * Par extension, est aussi utilisé pour représenter le coût de chaque couche.
 */
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

    public void printDimensions(String name) {
        super.printDimensions("Gradient", name);
    }

    /**
     * Calcule le gradient d'un {@link BiasVector} à partir d'une matrice
     * delta_l qui représente le terme d'erreur d'une couche, lors de
     * l'algorithme de descente de gradient.
     * Voir {@link MLP#gradientDescent}.
     * @immutable Ne modifie pas la matrice actuelle, en renvoie une nouvelle.
     */
    public BiasVector  sumErrorTerm() {
        double[][] biasVectorData = this.sumOverRows();
        return new BiasVector(biasVectorData);
    }

}
