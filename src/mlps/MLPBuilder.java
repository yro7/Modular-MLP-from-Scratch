package mlps;

import functions.ActivationFunction;
import functions.InitializationFunction;
import matrices.BiasVector;
import matrices.WeightMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static functions.ActivationFunction.SoftMax;

public class MLPBuilder {

    public static Random randomGenerator = new Random();

    private final List<Layer> layers = new ArrayList<>();
    private final int dimInput;
    private int previousLayerSize;


    public MLPBuilder(int dimInput) {
        this.dimInput = dimInput;
        this.previousLayerSize = dimInput;
    }

    public MLP build(){
        assert (this.layers.size() > 1) : "Le MLP doit avoir au moins une couche cachée.";

        for(Layer l : layers){
            assert(l.getActivationFunction() != SoftMax || l.size() != 1) : "Utilisation de SoftMax dans "
                    + "une couche avec un seul neuronne !";
        }

        return new MLP(this.layers, dimInput);
    }

    /**
     * Est utilisé lors de l'initialization aléatoire des poids du réseau de neurones, dans
     * {@link InitializationFunction}.
     * @param seed
     * @return
     */
    public MLPBuilder setRandomSeed(long seed){
        MLPBuilder.randomGenerator = new Random(seed);
        return this;
    }



    public MLPBuilder addLayer(int numberOfNeuronsOfNewLayer, ActivationFunction af){
        Layer newLayer = new Layer(previousLayerSize, numberOfNeuronsOfNewLayer, af);
        this.layers.add(newLayer);
        this.previousLayerSize = numberOfNeuronsOfNewLayer;
        return this;
    }

    public MLPBuilder addLayer(WeightMatrix weightMatrix, BiasVector biasVector, ActivationFunction func){
        int layerSize = weightMatrix.getNumberOfRows();
        Layer newLayer = new Layer(weightMatrix, biasVector, func);
        this.layers.add(newLayer);
        this.previousLayerSize = layerSize;
        return this;
    }

    public MLPBuilder addIdentityLayer(int dimension) {
        this.addLayer(WeightMatrix.createIdentityMatrix(dimension),
                BiasVector.createZeroBiasVector(dimension),
                ActivationFunction.Identity);
        return this;
    }

    /**
     * Créé un MLP "identité" qui n'applique aucune transformée à l'entrée.
     * @param numberOfHiddensLayers
     * @return
     */
    public static MLP createIdentityMLP(int dimension, int numberOfHiddensLayers) {
        MLPBuilder builder = MLP.builder(dimension);
        for(int i = 0; i < numberOfHiddensLayers; i++){
            builder.addIdentityLayer(dimension);
        }

        return builder.build();
    }



}
