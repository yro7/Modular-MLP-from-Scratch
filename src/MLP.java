import java.util.List;
import java.util.Vector;

public class MLP {

    private final int dimInput;
    private final List<Layer> layers;

    public MLP(List<Layer> layers, int dimInput){
        this.layers = layers;
        this.dimInput = dimInput;
    }

    /**
     * Envoie au réseau de neurones un vecteur d'activations initial (la donnée d'entrée mise sous forme de vecteur
     * aux dimensions de la première couche du réseau) et renvoie les activations des
     * @param input
     * @return
     */
    public ActivationVector feedForward(ActivationVector input){
        assert(input.size() == dimInput);
        ActivationVector activationsOfPreviousLayer = input;
        for(Layer layer : layers) {
            activationsOfPreviousLayer = layer.computeActivationVectorOfPreviousLayer(activationsOfPreviousLayer);
        }

        return activationsOfPreviousLayer;

    }

    public static MLPBuilder builder(int dimInput){
        return new MLPBuilder(dimInput);
    }


    public void print(){

        int n = layers.size();
        for(int i = 0; i < n; i ++){
            System.out.println("Layer n°" + i + " : ");
            layers.get(i).print();
            System.out.println();
        }
    }


}
