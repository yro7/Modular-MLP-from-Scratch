import Function.ActivationFunction;
import Function.InitializationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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

    public MLPBuilder setRandomSeed(long seed){
        ActivationFunction.randomGenerator = new Random(seed);
        return this;
    }


    public MLPBuilder addLayer(int numberOfNeuronsOfNewLayer, ActivationFunction af){
        Layer newLayer = new Layer(numberOfNeuronsOfNewLayer, previousLayerSize, af);
        this.layers.add(newLayer);
        this.previousLayerSize = numberOfNeuronsOfNewLayer;
        return this;
    }



}
