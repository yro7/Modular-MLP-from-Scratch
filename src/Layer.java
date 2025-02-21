import Function.ActivationFunction;
import Matrices.ActivationVector;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    private List<Neuron> neurons;
    private ActivationFunction activationFunction;
    // TODO "WeightMatrix" plutot que list neurons
    // TODO ActivationFunction et on enlève la psosibilité d'en avoir 1 différente par neurone du réseau.

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public void setNeurons(List<Neuron> neurons) {
        this.neurons = neurons;
    }

    public Layer(int numberOfNeurons, ActivationFunction activationFunction){
        List<Neuron> neurons = new ArrayList<>();
        for(int i = 0; i < numberOfNeurons; i ++){
            neurons.add(new Neuron(activationFunction));
        }
        this.neurons = neurons;
    }

    public void print() {
        int n = this.getNeurons().size();
        System.out.println("Size " + n);
        for(int i = 0; i < n; i++){
            Neuron neuronI = this.getNeuron(i);
            System.out.println("Neuron " + i + " W/B : " + neuronI.getWeights() + " / " + neuronI.getBias());
        }
    }

    private Neuron getNeuron(int i) {
        return this.getNeurons().get(i);
    }


    public void initialize(int previousLayerSize, int layerSize){
        this.getNeurons().forEach(n -> n.initialize(previousLayerSize, layerSize));
    }

    /**
     * Pour chaque neurone dans la couche, calcule l'activation en fonction des activations de la couche précédente.
     * @param activationsOfPreviousLayer Le vecteur d'activation de la couche précédente
     * @return Le nouveau vecteur d'activation de cette couche.
     */
    public ActivationVector computeActivationVectorOfPreviousLayer(ActivationVector activationsOfPreviousLayer) {
        ActivationVector newActivationVector = new ActivationVector();
        for(Neuron neuron : this.getNeurons()){
            double activation = neuron.feedPreviousLayer(activationsOfPreviousLayer);
            newActivationVector.add(activation);
        }
        return newActivationVector;
    }
}
