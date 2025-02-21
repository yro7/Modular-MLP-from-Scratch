import Function.ActivationFunction;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.WeightMatrix;

public class Layer {

    private WeightMatrix weightMatrix;

    /**
     * Le biais de chaque neurone. Si la couche possède n neurones, la {@link Matrices.Matrix} {@link BiasVector} sera de dimension n*1.
     */
    private BiasVector biasVector;
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
     * @param activationFunction
     */
    public Layer(int numberOfNeurons, int numberOfNeuronsOfPreviousLayer, ActivationFunction activationFunction){
        this.weightMatrix = new WeightMatrix(numberOfNeurons, numberOfNeuronsOfPreviousLayer, activationFunction);
    }

    public void print() {
        System.out.println("Layer n° " + this);
        System.out.println("Weights of the layer : ");
        this.getWeightMatrix().print();
    }

    /**
     * Pour chaque neurone dans la couche, calcule l'activation en fonction des activations de la couche précédente.
     * On calcule le produit matriciel de la matrice d'activation et de la matrice de poids de la couche actuelle,
     * Puis on y ajoute le biais et on y applique la fonction d'activation.
     * @param activationsOfPreviousLayer Le vecteur d'activation de la couche précédente
     * @return Le nouveau vecteur d'activation de cette couche.
     */
    public ActivationMatrix computeNewActivationMatrix(ActivationMatrix activationsOfPreviousLayer) {
        return activationsOfPreviousLayer
                .multiplyByWeightMatrix(weightMatrix)
                .applyFunction(d -> this.activationFunction.apply(d))
                .addBiasVector(this.biasVector);
    }

}
