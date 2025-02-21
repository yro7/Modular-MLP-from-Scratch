import Function.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public class MLPBuilder {

    private final List<Layer> layers = new ArrayList<>();
    private final int dimInput;
    private int previousLayerSize;

    public MLPBuilder(int dimInput) {
        this.dimInput = dimInput;
        this.previousLayerSize = dimInput;
    }

    public MLP build(){
        return new MLP(this.layers, dimInput);
    }


    public MLPBuilder addLayer(int numberOfNeuronsOfNewLayer, ActivationFunction af){
        Layer newLayer = new Layer(numberOfNeuronsOfNewLayer, previousLayerSize, af);
        this.layers.add(newLayer);
        this.previousLayerSize = numberOfNeuronsOfNewLayer;
        return this;
    }



}
