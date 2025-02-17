import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Layer {

    private List<Neuron> neurons;


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
            System.out.println("Neuron " + i + " W/B : " + neuronI.getWeight() + " / " + neuronI.getBias());
        }
    }

    private Neuron getNeuron(int i) {
        return this.getNeurons().get(i);
    }


    public void initialize(int previousLayerSize, int layerSize){
        this.getNeurons().forEach(n -> n.initialize(previousLayerSize, layerSize));
    }
}
