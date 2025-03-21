package Matrices;

import MLP.Layer;
import Function.ActivationFunction;

/**
 * Représente un vecteur de biais d'une {@link Layer} du {@link MLP}.
 * En pratique devrait toujours contenir en mémoire un double[][] de taille 1 x longueur
 * Où longueur est le nombre de neurones dans le couche.
 */
public class BiasVector extends Matrix<BiasVector> {

    public BiasVector(int length) {
        super(1, length);
    }

    public BiasVector(double[][] data){
        assert(data.length == 1) : "Un Vecteur de Biais devrait être de hauteur 1 ! Taille rentrée: "
                + data.length + " x " + data[0].length + ".";
        super(data);
    }

    /**
     * Initialise un nouveau vecteur de biais selon la fonction d'initialisation
     * de la fonction d'activation passée en argument.
     * @param length
     * @param numberOfNeuronsInPreviousLayer
     * @param af
     */
    public BiasVector(int length, int numberOfNeuronsInPreviousLayer, ActivationFunction af){
        this(length);
        this.applyToElements((i,j) -> this.getData()[i][j] = af.applyRandomBias(length,numberOfNeuronsInPreviousLayer));
    }

    public static BiasVector createZeroBiasVector(int dimension) {
        double[][] data = new double[1][dimension];
        return new BiasVector(data);
    }

    @Override
    protected BiasVector createInstance(int rows, int length) {
        assert(rows == 1) : "Un Vecteur de Biais devrait être de hauteur 1 ! Taille rentrée: "
                + rows + " x " + length + ".";
        return new BiasVector(length);
    }

    public void printDimensions(String name) {
        super.printDimensions("BiasVector", name);
    }

    public int getLength(){
        return this.data[0].length;
    }

    /**
     * Récupère le biais n°i du vecteur.
     *
     * @return La valeur contenue à la case i du vecteur.
     */
    public double getData(int i) {
        return this.data[0][i];
    }


}

