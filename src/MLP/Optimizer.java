package MLP;


import Function.LossFunction;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.GradientMatrix;

import java.util.List;

import static Function.ActivationFunction.SoftMax;
import static Function.LossFunction.CE;

/**
 * Représente l'optimiseur qui sera utilisé lors de la mise à jour des paramètres
 * du MLP (voir {@link MLP#updateParameters}.
 */
public class Optimizer {

    int temp;

    public Optimizer(int temp) {
        this.temp = temp;
    }


    public void updateParameters(ActivationMatrix input, ActivationMatrix expectedOutput,
                                 LossFunction lossFunction, MLP mlp) {

        assert lossFunction != CE || (mlp.getLastLayer().getActivationFunction() == SoftMax) : "La couche de sortie du réseau " +
                "devrait être Softmax si la fonction de coût utilisée est la Cross Entropie !";

        List<Pair<GradientMatrix, BiasVector>> gradients = mlp.backpropagate(input, expectedOutput, lossFunction);

        // TODO IMPLEMENT OPTIMIZERS
        double learningRate = 0.0001;

        for (int l = 0; l < mlp.getLayers().size(); l++) {
            GradientMatrix weightCorrection = gradients.get(l).getA().clone().multiply(learningRate);
            BiasVector biasGradient = gradients.get(l).getB().clone().multiply(learningRate);

            Layer currentLayer = mlp.getLayer(l);

            currentLayer.getWeightMatrix().substract(weightCorrection);
            currentLayer.getBiasVector().substract(biasGradient);
        }


    }

}
