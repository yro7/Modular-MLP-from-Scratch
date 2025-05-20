package mlps.optimizers;


import functions.LossFunction;
import mlps.MLP;
import mlps.regularizations.ParameterRegularization;
import matrices.ActivationMatrix;

import static functions.ActivationFunction.SoftMax;
import static functions.LossFunction.CE;
import static mlps.MLP.BackProResult;

/**
 * Représente l'optimiseur qui sera utilisé lors de la mise à jour des paramètres
 * du MLP (voir {@link MLP#updateParameters}.
 */
public abstract class Optimizer {

    public abstract void updateParameters(BackProResult gradients, MLP mlp);

    public void updateParametersBody(ActivationMatrix input, ActivationMatrix expectedOutput,
                                     LossFunction lossFunction, MLP mlp, ParameterRegularization regularization) {

        assert lossFunction != CE || (mlp.getLastLayer().getActivationFunction() == SoftMax) : "La couche de sortie du réseau " +
                "devrait être Softmax si la fonction de coût utilisée est la Cross Entropie !";

        BackProResult gradients = mlp.backpropagate(input, expectedOutput, lossFunction, regularization);

        this.updateParameters(gradients, mlp);
    }

}
