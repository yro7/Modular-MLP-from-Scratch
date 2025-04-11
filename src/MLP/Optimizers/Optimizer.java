package MLP.Optimizers;


import Function.LossFunction;
import MLP.MLP;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.GradientMatrix;

import java.util.List;

import static Function.ActivationFunction.SoftMax;
import static Function.LossFunction.CE;
import static Function.LossFunction.CE;
import MLP.Pair;
import static MLP.MLP.BackProResult;
import MLP.Layer;

/**
 * Représente l'optimiseur qui sera utilisé lors de la mise à jour des paramètres
 * du MLP (voir {@link MLP#updateParameters}.
 */
public abstract class Optimizer {

    public abstract void updateParameters(BackProResult gradients, MLP mlp);

    public void updateParametersBody(ActivationMatrix input, ActivationMatrix expectedOutput,
                                                 LossFunction lossFunction, MLP mlp) {

        assert lossFunction != CE || (mlp.getLastLayer().getActivationFunction() == SoftMax) : "La couche de sortie du réseau " +
                "devrait être Softmax si la fonction de coût utilisée est la Cross Entropie !";

        BackProResult gradients = mlp.backpropagate(input, expectedOutput, lossFunction);
        this.updateParameters(gradients, mlp);
    }

}
