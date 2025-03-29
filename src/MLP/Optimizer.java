package MLP;


/**
 * Représente l'optimiseur qui sera utilisé lors de la mise à jour des paramètres
 * du MLP (voir {@link MLP#updateParameters}.
 */
public class Optimizer {

    int temp;

    public Optimizer(int temp) {
        this.temp = temp;
    }
}
