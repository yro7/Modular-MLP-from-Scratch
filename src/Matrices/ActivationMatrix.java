package Matrices;

/**
 * Représente une matrice d'activation d'une couche (Layer) d'un réseau de neurones (MLP).
 * Plutôt qu'un simple vecteur d'activation, cette classe permet de traiter plusieurs entrées
 * simultanément. Voir <a href="https://en.wikipedia.org/wiki/Online_machine_learning#Batch_learning">Batch Learning</a>.
 *
 * Une ActivationMatrix contient typiquement les valeurs d'activation pour tous les neurones
 * d'une couche donnée, pour chaque exemple dans un lot (batch) d'entrées.
 */
public class ActivationMatrix extends Matrix<ActivationMatrix> {

    /**
     * Constructeur créant une matrice d'activation de taille rows x cols remplie de zéros.
     *
     * @param rows Nombre de lignes (correspond généralement au nombre de neurones)
     * @param cols Nombre de colonnes (correspond généralement à la taille du batch)
     */
    public ActivationMatrix(int rows, int cols) {
        super(rows, cols);
    }

    /**
     * Constructeur à partir d'un tableau bidimensionnel.
     *
     * @param data Tableau 2D de valeurs double représentant les activations
     */
    public ActivationMatrix(double[][] data) {
        super(data);
    }

    /**
     * Méthode factory pour créer une nouvelle instance d'ActivationMatrix.
     * Utilisée en interne par les méthodes héritées de Matrix.
     *
     * @param rows Nombre de lignes de la nouvelle instance
     * @param cols Nombre de colonnes de la nouvelle instance
     * @return Une nouvelle instance d'ActivationMatrix
     */
    @Override
    protected ActivationMatrix createInstance(int rows, int cols) {
        return new ActivationMatrix(rows, cols);
    }

    /**
     * Ajoute un vecteur de biais à chaque colonne de cette matrice d'activation.
     * Cette opération est typiquement effectuée dans le calcul forward d'un réseau de neurones.
     *
     * @param biasVector Le vecteur de biais à ajouter
     * @return Cette matrice modifiée
     * @throws AssertionError si les dimensions ne correspondent pas
     * @mutable Cette méthode modifie la matrice actuelle
     * @intermédiaire Renvoie this pour permettre le chaînage
     */
    public ActivationMatrix addBiasVector(BiasVector biasVector) {
        assert(this.getNumberOfRows() == biasVector.getNumberOfRows()) : "Vecteur d'entrée incorrect : "
                + "Hauteur du vecteur bias : " + biasVector.getNumberOfRows()
                + " Hauteur de la matrice : " + this.getNumberOfRows();

        this.applyToElements((i, j) -> this.getData()[i][j] = this.getData()[i][j] + biasVector.getData()[i][0]);
        return this;
    }

    /**
     * Calcule le produit matriciel entre une matrice de poids et cette matrice d'activation.
     * Cette opération est typiquement effectuée dans le calcul forward d'un réseau de neurones.
     *
     * Si A est cette matrice d'activation et W est la matrice de poids,
     * calcule le produit W×A.
     *
     * @param weightMatrix La matrice de poids à multiplier à gauche
     * @return Une nouvelle matrice d'activation résultant du produit
     * @throws AssertionError si les dimensions sont incompatibles
     * @immutable Ne modifie pas la matrice d'activation actuelle
     * @intermédiaire Renvoie une nouvelle matrice pour permettre le chaînage
     */
    public ActivationMatrix multiplyAtRightByWeightMatrix(WeightMatrix weightMatrix) {
        return this.multiplyAtRight(weightMatrix);
    }

    /**
     * Affiche les dimensions de la matrice d'activation dans la console, pour le débogage.
     *
     * @param name Le nom à afficher pour identifier la matrice
     * @terminale Ne renvoie rien et termine la chaîne d'opérations
     */
    public void printDimensions(String name) {
        super.printDimensions("Activation", name);
    }
}