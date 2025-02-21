package Matrices;

import Function.ActivationFunction;

/**
 * Représente la matrice des poids d'une {@link Layer} d'un {@link MLP}.
 *
 * Chaque neuronne est représentée par une ligne de la matrice.
 * La colonne i d'une matrice de poids représente les poids de chaque neuronne de la couche actuelle avec
 * le neuronne i de la couche précédente.
 */
public class WeightMatrix extends Matrix<WeightMatrix> {

    public WeightMatrix(int rows, int cols) {
        super(rows, cols);
    }

    @Override
    protected WeightMatrix createInstance(int rows, int cols) {
        return new WeightMatrix(rows, cols);
    }

    /**
     * Permet d'initialiser une nouvelle matrice selon la fonction d'activation de la nouvelle couche du réseau de neurones.
     *
     * @param rows Le nombre de neurones dans la nouvelle couche, et donc la hauteur de la matrice
     * @param cols Le nombre de neurones dans la couche précédente, et donc le nombre de poids (la largeur) de la matrice.
     * @param activationFunction La fonction d'activation utilisée dans la nouvelle
     */
    public WeightMatrix(int rows, int cols, ActivationFunction activationFunction){
        super(rows,cols);
        this.applyToElements((i,j) -> {
            this.getData()[i][j] = activationFunction.getInitializationFunction().getRandomWeight.apply(rows,cols);
        });
    }

}
