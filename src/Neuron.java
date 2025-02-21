import Function.ActivationFunction;
import Matrices.ActivationVector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Neuron {

    /**
     * Représente le poids des connexions entre le neuronne n°i de la couche précédente et le neuronne actuel.
     * Ainsi weights(1) est le poids entre le premier neuronne de la couche précédente et le neuronne de cette couche.
     */
    private List<Double> weights;
    private double bias;
    private final ActivationFunction activationFunction;


    public Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public Neuron(ActivationFunction activationFunction, int numberOfInputs, int layerSize) {
        this.activationFunction = activationFunction;
        initialize(numberOfInputs, layerSize);
    }

    public double getWeight(int i) {
        return weights.get(i);
    }

    public ActivationFunction getActivationFunction(){
        return this.activationFunction;
    }

    public double getBias() {
        return bias;
    }



    public double feed(double input) {
        return activationFunction.apply(input+getBias());
    }

    /**
     *
     * @param activationsOfPreviousLayer Les activations de la couche précédente du réseau
     * @return L'activation associée.
     */
    public double feedPreviousLayer(ActivationVector activationsOfPreviousLayer) {
        int n = activationsOfPreviousLayer.size();
        double activation = IntStream.range(0,n)
                .mapToDouble(i -> this.getWeight(i) * activationsOfPreviousLayer.get(i))
                .sum();
        return feed(activation);
    }



    public void initialize(int previousLayerSize, int layerSize){
        double bias = getRandomBias(previousLayerSize, layerSize);
        List<Double> weights = new ArrayList<>();
        // Pour chaque neurone de la couche précédent, on initialise un poids aléatoire
        // selon la fonction d'activation du réseau.
        for(int i = 0; i < previousLayerSize; i++){
            weights.add(this.getRandomWeight(previousLayerSize, layerSize));
        }
        this.bias = bias;
        this.weights = weights;
    }


    public double getRandomBias(int numberOfInputs, int layerSize){
        return this.getActivationFunction().initializationFunction.getRandomBias.apply(numberOfInputs, layerSize);
    }

    public double getRandomWeight(int numberOfInputs, int layerSize){
        return this.getActivationFunction().initializationFunction.getRandomWeight.apply(numberOfInputs, layerSize);
    }

    public List<Double> getWeights(){
        return this.weights;
    }
}
