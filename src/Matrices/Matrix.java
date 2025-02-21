package Matrices;

import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Classe utilitaire pour implémenter tous les algorithmes de gestion de matrices.
 */
        // CRTP pour method chaining avec préservation de type
        // c'est pas très joli mais c'est utile
    // TODO faire un type de matrix générique qui peut prendre <? extends Number> pour ne pas
    // TODO etre restreint à des doubles.
public abstract class Matrix<T extends Matrix<T>> {

    double[][] data;

    public Matrix(int rows, int cols) {
        data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.data = data;
    }



    public Matrix(Matrix<?> source){
        this(source.getNumberOfRows(), source.getNumberOfColumns());
        applyToElements((i,j) -> this.data[i][j] = source.data[i][j]);
    }


    protected abstract T createInstance(int rows, int cols);

    @SuppressWarnings("unchecked")
    protected T self() {
        return (T) this;
    }

    public double[][] getData() {
        return data;
    }

    public int getNumberOfColumns(){
        return this.getData()[0].length;
    }

    public int getNumberOfRows(){
        return this.getData().length;
    }

    // Interface qui permet d'itérer sur les éléments de la matrice
    @FunctionalInterface
    public  interface ElementOperation {
        void apply(int i, int j);
    }

    /**
     * Performe l'action donnée pour chaque élément de la matrice.
     * Contrairement à {@link #forEach(Consumer)} )}, elle "gère uniquement les indices i,j.
     * C'est une opération intermédiaire.
     */
    public void applyToElements(ElementOperation operation){

        for(int i = 0; i < this.getNumberOfRows(); i++){
            for(int j = 0; j < this.getNumberOfColumns(); j++){
                operation.apply(i,j);
            }
        }
    }

    /**
     * Performe l'action donnée pour chaque élément de la matrice.
     * Contrairement à {@link #applyToElements(ElementOperation)}, elle gère les valeurs aux indices i,j.
     * C'est une opération intermédiaire.
     * @param action l'action a effectuer.
     */
    public T forEach(Consumer<? super Double> action){
        applyToElements((i,j) -> action.accept(this.data[i][j]));
        return self();
    }

    /**
     * Renvoie une nouvelle matrice dont les coefficients sont égaux à la matrice actuelle.
     * C'est une opération intermédiaire.
     * @return
     */
    public T clone(){
        T res = this.createInstance(this.getNumberOfRows(), this.getNumberOfColumns());
        applyToElements((i,j) -> res.getData()[i][j] = this.getData()[i][j]);
        return res;
    }

    /**
     * Cette fonction et {@link #cloneWeight()}} permettent de renvoyer une nouvelle {@link ActivationMatrix} qui a les mêmes valeurs que la matrice actuelle.
     * Avoir 2 sous-classes spécifiant "Activation" et "Weight" permet de séparer les matrices en fonction de leur usage,
     * ainsi que de séparer le code en plusieurs parties pour plus de lisibilité.
     * @return une {@link ActivationMatrix} aux mêmes valeurs que la matrice actuelle.
     */
    public ActivationMatrix cloneActivation(){
        return (ActivationMatrix) this.clone();
    }
    /**
     * Cette fonction et {@link #cloneActivation()} ()}} permettent de renvoyer une nouvelle {@link WeightMatrix} qui a les mêmes valeurs que la matrice actuelle.
     * Avoir 2 sous-classes spécifiant "Activation" et "Weight" permet de séparer les matrices en fonction de leur usage,
     * ainsi que de séparer le code en plusieurs parties pour plus de lisibilité.
     * @return une {@link WeightMatrix} aux mêmes valeurs que la matrice actuelle.
     */
    public WeightMatrix cloneWeight(){
        return (WeightMatrix) this.clone();
    }

    /**
     * Performe l'action donnée à partir des termes de la matrice
     * et de l'autre matrice passée en argument.
     * C'est une opération intermédiaire.
     * @param function la fonction à appliquer
     * @param matrix la deuxième matrice à utiliser
     * @return la même matrice modifiée
     */
    public T elementWiseOperation(BiFunction<Double,Double,Double> function, Matrix<?> matrix){
        verifyDimensions(matrix);
        applyToElements((i,j) -> function.apply(this.data[i][j], matrix.getData()[i][j]));
        return self();
    }

    /**
     * Applique la fonction donnée à chaque élément de la matrice,
     * puis la renvoie.
     * C'est une opération intermédiaire.
     * @param function la fonction à appliquer
     * @return la même matrice modifiée par la fonction
     */
    public T applyFunction(Function<Double,Double> function){
        this.forEach(function::apply);
        return self();
    }

    /**
     * Renvoie une NOUVELLE matrice qui correspond au produit de la matrice actuelle
     * ainsi que de la matrice passée en argument.
     * C'est une opération intermédiaire. (/!\ non commutative).
     *
     * Si A est la matrice this, B la matrice entrée,
     * renvoie AxB. (du type de A).
     *
     * Attention, les dimensions de la nouvelle matrice ne sont pas forcément égales
     * aux dimensions de l'ancienne.
     * @param matrix la matrice par laquelle on multiplie
     * @return une nouvelle matrice produit des 2.
     */
    // TODO optimiser à fond la multiplication
    public T multiply(Matrix<?> matrix){
        assert(this.getNumberOfColumns() == matrix.getNumberOfRows()) : "Matrices incompatibles pour un produit AxB :"
                + " Nombre de colonnes de A : " + this.getNumberOfColumns()
                + " Nombre de lignes de B : " + matrix.getNumberOfRows();
        int newNumberOfRows = this.getNumberOfRows();
        int newNumberOfColumns = matrix.getNumberOfColumns();
        T newMatrix = createInstance(newNumberOfRows, newNumberOfColumns);
        newMatrix.applyToElements((i,j) -> {
            for(int k = 0; k < newNumberOfRows; k++){
                newMatrix.getData()[i][j] += this.getData()[i][k]*matrix.getData()[k][j];
            }
        });

        return newMatrix;
    }

    /**
     * Renvoie une NOUVELLE matrice qui correspond au produit de la matrice actuelle
     * ainsi que de la matrice passée en argument.
     * C'est une opération intermédiaire. (/!\ non commutative).
     *
     * Si A est la matrice this, B la matrice entrée,
     * renvoie BxA. (du type de A).
     *
     * Attention, les dimensions de la nouvelle matrice ne sont pas forcément égales
     * aux dimensions de l'ancienne.
     * @param matrix la matrice par laquelle on multiplie
     * @return une nouvelle matrice produit des 2.
     */
    public T multiplyAtRight(Matrix<?> matrix) {
        assert(matrix.getNumberOfColumns() == this.getNumberOfRows()) : "Matrices incompatibles pour un produit AxB :"
                + " Nombre de colonnes de A : " + this.getNumberOfColumns()
                + " Nombre de lignes de B : " + matrix.getNumberOfRows();

        int newNumberOfRows = matrix.getNumberOfRows();
        int newNumberOfColumns = this.getNumberOfColumns();
        T newMatrix = createInstance(newNumberOfRows, newNumberOfColumns);

        newMatrix.applyToElements((i,j) -> {
            double sum = 0.0;
            for(int k = 0; k < this.getNumberOfRows(); k++) {
                sum += matrix.getData()[i][k] * this.getData()[k][j];
            }
            newMatrix.getData()[i][j] = sum;
        });

        return newMatrix;
    }

    /**
     * Renvoie la somme, élément par élément, des éléments de la matrice.
     * C'est une opération terminale.
     * @return un double qui correspond à l'ensemble des
     */
    public double sum(){
        // Nécessaire de passer par un tableau pour pouvoir
        // Utiliser la variable dans le lambda #forEach
        double[] res = {0.0};
        this.forEach(d -> res[0] += d);
        return res[0];
    }



    /**
     * Soustrait une autre {@link Matrix} terme à terme à la matrice actuelle.
     * C'est une opération intermédiaire.
     * @param matrix la matrice de même dimension que this, qu'on soustrait
     * @return la même {@link Matrix} qui correspond à la différence terme à terme.
     */
    public T substract(Matrix<?> matrix){
        verifyDimensions(matrix);
        return elementWiseOperation((d1,d2) -> d1 - d2, matrix);
    }

    /**
     * Multiplie une autre {@link Matrix} terme à terme à la matrice actuelle.
     * C'est une opération intermédiaire.
     * @param matrix la matrice de même dimension que this, qu'on multiplie
     * @return la même {@link Matrix} qui correspond à la différence terme à terme.
     */
    public T hadamardProduct(Matrix<?> matrix){
        verifyDimensions(matrix);
        return elementWiseOperation((d1, d2) -> d1 * d2, matrix);
    }

    /**
     * Vérifie que la matrice passée en argument possède les mêmes
     * dimensions que la matrice actuelle et lève une {@link AssertionError} dans le cas contraire.
     * C'est une opération terminale.
     * @param matrix
     */
    private void verifyDimensions(Matrix<?> matrix) {
        assert(this.hasSameDimensions(matrix)) : "Les matrices ne sont pas de même dimensions !"
                + " Matrice A : " + this.getNumberOfRows()+ " * " + this.getNumberOfColumns()
                + " Matrice B : " + matrix.getNumberOfRows()+ " * " + matrix.getNumberOfColumns();

    }

    /**
     * Vérifie que 2 matrices ont des dimensions compatibles pour les opérations.
     * C'est une opération terminale.
     * {@link #elementWiseOperation}.
     * @param matrix la matrice à tester
     * @return True si elles ont les mêmes dimensions, False sinon
     */
    private boolean hasSameDimensions(Matrix<?> matrix) {
        return this.getNumberOfColumns() == matrix.getNumberOfColumns()
                && this.getNumberOfRows() == matrix.getNumberOfRows();
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le logarithme népérien (log base e) de l'ancien;
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public T log(){
        return this.applyFunction(Math::log);
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le cosinus hyperbolique de l'ancien;
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public T cosh(){
        return this.applyFunction(Math::cosh);
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le carré de l'ancien.
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public T square(){
        return this.applyFunction(d -> Math.pow(d,2));
    }

    public void print(){
        this.applyToElements((i,j) -> {
            if(j == 0) System.out.println();
            System.out.print(this.getData()[i][j] + ",  ");
        });
        System.out.println();
    }

    /**
     * Renvoie le nombre d'éléments dans la matrice (n x p si la matrice est une matrice de taille (n,p)).
     * C'est une opération terminale.
     * @return l'entier qui correspond au nombre d'éléments de la matrice.
     */
    public int size(){
        return this.getNumberOfColumns()*this.getNumberOfRows();
    }

    public <T extends Matrix<?>> T createIdentity(int n){
        T identity = (T) createInstance(n,n);
        identity.applyToElements((i,j) -> {
            identity.getData()[i][j] = (i == j ? 1.0 : 0.0);
        });

        return identity;
    }
}
