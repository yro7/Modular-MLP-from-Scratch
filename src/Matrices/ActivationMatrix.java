package Matrices;

/**
 * Représente une matrice d'activation d'une {@link Layer} d'un {@link MLP}.
 * Plutôt qu'un ActivationVector, la ActivationMatrix permet de process plusieurs entrées
 * en même temps. Voir <a href="https://en.wikipedia.org/wiki/Online_machine_learning#Batch_learning">Batch Learning</a>.
 */
public class ActivationMatrix extends Matrix {

    public ActivationMatrix(int rows, int cols) {
        super(rows, cols);
    }



}
