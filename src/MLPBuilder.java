import java.util.ArrayList;
import java.util.List;

public class MLPBuilder {

    private final List<Layer> layers = new ArrayList<>();
    private final int dimInput;
    private final int previousLayerSize;

    public MLPBuilder(int dimInput){
        this.dimInput = dimInput;
        this.previousLayerSize = dimInput;
    }

    public MLP build(){
        return new MLP(this.layers, dimInput);
    }

    public MLPBuilder addLayer(int neurons, ActivationFunction af){
        Layer newLayer = new Layer(neurons, af);
        newLayer.initialize(previousLayerSize, neurons);
        this.layers.add(newLayer);
        return this;
    }



}
