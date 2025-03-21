package Matrices;

import MLP.Layer;
import Function.ActivationFunction;

/**
 * Représente la matrice des poids d'une {@link Layer} d'un {@link MLP}.
 *
 * Contient n lignes et p colonnes.
 * Si l'ancienne couche était de taille n, et la nouvelle de taille p,
 * alors la matrice de poids sera de taille n x p.
 *
 * Chaque neuronne est représentée par une colonne de la matrice.
 * La colonne i d'une matrice de poids correspond au neurone i de la couche du réseau.
 * La valeur en (i,j) correspond au poids entre le neuronne i de la couche précédente et le neuronne j de cette couche.
 */
public class WeightMatrix extends Matrix<WeightMatrix> {

    public WeightMatrix(int rows, int cols) {
        super(rows, cols);
    }

    @Override
    protected WeightMatrix createInstance(int numberOfNeuronsInPreviousLayer, int numberOfNeuronsInNewLayer) {
        return new WeightMatrix(numberOfNeuronsInPreviousLayer, numberOfNeuronsInNewLayer);
    }

    public WeightMatrix(double[][] data) {
        super(data);
    }

    /**
     * Permet d'initialiser une nouvelle matrice selon la fonction d'activation de la nouvelle couche du réseau de neurones.
     *
     * @param numberOfNeuronsInPreviousLayer Le nombre de neurones dans la couche précédente, et donc la hauteur de la matrice
     * @param numberOfNeuronsInNewLayer Le nombre de neurones dans la couche actuelle, et donc le nombre de poids (la largeur) de la matrice.
     * @param activationFunction La fonction d'activation utilisée pour initialiser les poids de la nouvelle matrice
     */
    public WeightMatrix(int numberOfNeuronsInPreviousLayer, int numberOfNeuronsInNewLayer, ActivationFunction activationFunction){
        super(numberOfNeuronsInPreviousLayer,numberOfNeuronsInNewLayer);
        this.applyToElements((i,j) -> {
            this.getData()[i][j] = activationFunction.getInitializationFunction()
                    .getRandomWeight.apply(numberOfNeuronsInPreviousLayer,numberOfNeuronsInNewLayer);
        });
    }

    public void printDimensions(String name) {
        super.printDimensions("Weight", name);
    }

    public static WeightMatrix createIdentityMatrix(int dimension) {
        double[][] data = new double[dimension][dimension];
        for(int i = 0; i < dimension; i++){
            data[i][i] = 1.0;
        }
        return new WeightMatrix(data);
    }


}
