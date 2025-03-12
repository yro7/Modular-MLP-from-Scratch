import Function.ActivationFunction;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.WeightMatrix;

import java.util.function.Function;

public class Layer {

    public WeightMatrix weightMatrix;

    /**
     * Le biais de chaque neurone. Si la couche possède n neurones, la {@link Matrices.Matrix} {@link BiasVector} sera de dimension n*1.
     */
    public BiasVector biasVector;
    /** La <a href="https://en.wikipedia.org/wiki/Activation_function">Fonction d'Activationf</a> à utiliser dans cette couche du réseau de neurones.
     * Cette architecture ne permet donc pas d'avoir une {@link ActivationFunction} différente par neurone de la couche
     * (la couche est neuron-agnostique, elle ne possède pas d'objet "neurone" à proprement parler), mais à part dans certaines architectures
     * précises (voir <a href="https://www.ibm.com/think/topics/mixture-of-experts">Mixture of Experts</a>) cela n'est généralement pas une bonne chose à avoir.
     */
    private ActivationFunction activationFunction;

    public WeightMatrix getWeightMatrix() {
        return weightMatrix;
    }

    /**
     *
     * @param numberOfNeurons Le nombre de neurones de la nouvelle couche.
     * @param numberOfNeuronsOfPreviousLayer le nombre de neurones de la couche précédente.
     * @param activationFunction la fonction d'activation à appliquer à la fin du calcul
     */
    public Layer(int numberOfNeurons, int numberOfNeuronsOfPreviousLayer, ActivationFunction activationFunction){
        this.weightMatrix = new WeightMatrix(numberOfNeurons, numberOfNeuronsOfPreviousLayer, activationFunction);
        this.activationFunction = activationFunction;
        this.biasVector = new BiasVector(numberOfNeurons, numberOfNeuronsOfPreviousLayer, activationFunction);
    }

    public void print() {
        System.out.println("Activation function : " + this.getActivationFunction());
        System.out.println("Taille : " + this.getWeightMatrix().getNumberOfRows());
        System.out.println("Taille de la couche précédente : " + this.getWeightMatrix().getNumberOfColumns());
        System.out.println("Weights of the layer : ");
        this.getWeightMatrix().print();
        System.out.println("Biais: ");
        this.biasVector.print();
    }

    /**
     * Renvoie WxA + B où W est la matrice de poids, A la matrice d'activation actuelle,
     * B le vecteur biais, f la fonction d'activation de la couche.
     *
     * Calcule donc les activations des neuronnes de cette couche.
     * @param activationsOfPreviousLayer Le vecteur d'activation de la couche précédente
     * @return Le nouveau vecteur d'activation de cette couche.
     * @immutable ne modifie pas la matrice passée en argument
     */
    public ActivationMatrix computePreFunctionNewActivationMatrix(ActivationMatrix activationsOfPreviousLayer) {
        return activationsOfPreviousLayer
                .multiplyAtRightByWeightMatrix(weightMatrix)  // Performe A' = W*A
                .addBiasVector(this.biasVector); // A' = A + B
    }

    public ActivationFunction getActivationFunction(){
        return this.activationFunction;
    }

    public int size(){
        return this.getWeightMatrix().getNumberOfRows();
    }

    public Function<Double,Double> getDerivativeOfAF(){
        return this.getActivationFunction().getDerivative();
    }

    public BiasVector getBiasVector(){
        return this.biasVector;
    }
}
