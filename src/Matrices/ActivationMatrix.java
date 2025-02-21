package Matrices;

/**
 * Représente une matrice d'activation d'une {@link Layer} d'un {@link MLP}.
 * Plutôt qu'un ActivationVector, la ActivationMatrix permet de process plusieurs entrées
 * en même temps. Voir <a href="https://en.wikipedia.org/wiki/Online_machine_learning#Batch_learning">Batch Learning</a>.
 */
public class ActivationMatrix extends Matrix<ActivationMatrix> {

    public ActivationMatrix(int rows, int cols) {
        super(rows, cols);
    }

    @Override
    protected ActivationMatrix createInstance(int rows, int cols) {
        return new ActivationMatrix(rows, cols);
    }

    public ActivationMatrix(double[][] data) {
        super(data);
    }


    /**
     * Ajoute un Vecteur de Biais à une matrice d'activation.
     * Cette fonction ajoute le vecteur biais à chaque colonne de la matrice d'activation.
     * C'est une opération intermédiaire.
     * @param biasVector
     * @return
     */
    public ActivationMatrix addBiasVector(BiasVector biasVector){
        assert(this.getNumberOfRows() == biasVector.getNumberOfRows()) : "Vecteur d'entrée incorrect : "
                + "Hauteur du vecteur bias : " + biasVector.getNumberOfRows()
                + " Hauteur de la matrice : " + this.getNumberOfRows();

        this.applyToElements((i,j) -> this.getData()[i][j] = this.getData()[i][j]+biasVector.getData()[i][0]);
        return this;
    }

    /**
     * Si W est la matrice de poids et A est la matrice d'activation actuelle,
     * renvoie WxA.
     * @param weightMatrix la matrice par laquelle multiplier la matrice d'activation.
     * @return
     */
    public ActivationMatrix multiplyAtRightByWeightMatrix(WeightMatrix weightMatrix){
        return this.multiplyAtRight(weightMatrix);
    }


}
