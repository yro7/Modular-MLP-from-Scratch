package Matrices;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Classe utilitaire pour implémenter tous les algorithmes de gestion de matrices.
 */

public class Matrix {

    private final double[][] data;

    public Matrix(int rows, int cols) {
        data = new double[rows][cols];
    }

    /**
     * Construit une nouvelle matrice de même dimension que la matrice d'entrée.
     * @param matrix la matrice d'entrée
     */
    public Matrix(Matrix matrix) {
        data = new double[matrix.getNumberOfRows()][matrix.getNumberOfColumns()];
    }

    public double[][] getData() {
        return data;
    }

    public int getNumberOfColumns(){
        return this.getData().length;
    }

    public int getNumberOfRows(){
        return this.getData()[0].length;
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
    public Matrix forEach(Consumer<? super Double> action){
        applyToElements((i,j) -> action.accept(this.data[i][j]));
        return this;
    }

    /**
     * Renvoie une nouvelle matrice dont les coefficients sont égaux à la matrice actuelle.
     * C'est une opération intermédiaire.
     * @return
     */
    @Override
    public Matrix clone(){
        Matrix res = new Matrix(this);
        applyToElements((i,j) -> res.data[i][j] = this.data[i][j]);
        return res;
    }

    /**
     * Performe l'action donnée à partir des termes de la matrice
     * et de l'autre matrice passée en argument.
     * C'est une opération intermédiaire.
     * @param function la fonction à appliquer
     * @param matrix la deuxième matrice à utiliser
     * @return la même matrice modifiée
     */
    public Matrix elementWiseOperation(BiFunction<Double,Double,Double> function, Matrix matrix){
        verifyDimensions(matrix);
        applyToElements((i,j) -> function.apply(this.data[i][j],matrix.data[i][j]));
        return this;
    }

    /**
     * Renvoie une nouvelle matrice dont chaque élément est le résultat
     * de la fonction appliquée à l'élément de la matrice initiale.
     * C'est une opération intermédiaire.
     * @param function la fonction à appliquer
     * @return le nouveau vecteur d'activation.
     */
    public Matrix applyFunction(Function<Double,Double> function){
        Matrix res = new Matrix(this);
        res.forEach(function::apply);
        return res;
    }

    /**
     * Renvoie une nouvelle matrice qui correspond au produit de la matrice actuelle
     * ainsi que de la matrice passée en argument.
     *
     * C'est une opération intermédiaire. (/!\ non commutatif).
     * Attention, les dimensions de la nouvelle matrice ne sont pas forcément égales
     * aux dimensions de l'ancienne.
     * @param matrix la matrice par laquelle on multiplie
     * @return une nouvelle matrice produit des 2.
     */
    public Matrix multiply(Matrix matrix){
        return new Matrix(0,0);
    }



    /**
     * Soustrait une autre {@link Matrix} terme à terme à la matrice actuelle.
     * C'est une opération intermédiaire.
     * @param matrix la matrice de même dimension que this, qu'on soustrait
     * @return une nouvelle {@link Matrix} qui correspond à la différence terme à terme.
     */
    public Matrix substract(Matrix matrix){
        verifyDimensions(matrix);
        return elementWiseOperation((d1,d2) -> d1 - d2, matrix);
    }

    /**
     * Vérifie que la matrice passée en argument possède les mêmes
     * dimensions que la matrice actuelle.
     * @param matrix
     */
    private void verifyDimensions(Matrix matrix) {
        assert(this.hasSameDimensions(matrix)) : "Les matrices ne sont pas de même dimensions !"
                + " Matrice A : " + this.getNumberOfRows()+ " * " + this.getNumberOfColumns()
                + " Matrice B : " + matrix.getNumberOfRows()+ " * " + matrix.getNumberOfColumns();

    }

    private boolean hasSameDimensions(Matrix matrix) {
        return this.getNumberOfColumns() == matrix.getNumberOfColumns()
                && this.getNumberOfRows() == matrix.getNumberOfRows();
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le logarithme népérien (log base e) de l'ancien;
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public Matrix log(){
        return this.applyFunction(Math::log);
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le cosinus hyperbolique de l'ancien;
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public Matrix cosh(){
        return this.applyFunction(Math::cosh);
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le carré de l'ancien;
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public Matrix square(){
        return this.applyFunction(d -> Math.pow(d,2));
    }

    public void print(){
        this.applyToElements((i,j) -> {
            if(j == 0) System.out.println();
            System.out.print(this.getData()[i][j] + ", ");
        });
    }

}
