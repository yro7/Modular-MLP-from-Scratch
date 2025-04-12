package MLP.Optimizers;

import MLP.MLP;
import Matrices.BiasVector;
import Matrices.GradientMatrix;

import MLP.Layer;
import static MLP.MLP.BackProResult;

/**
 * Voir <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">Descente de Gradient Stochastique</a>.
 * Optimizer le plus simple, mltiplie les gradients par un learning rate avant de les soustraire aux paramètres du réseau.
 */
public class SGD extends Optimizer {

    public final double learningRate;

    /**
     *
     * @param learningRate Le taux d'apprentissage utilisé par la SGD.
     */
    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void updateParameters(BackProResult gradients, MLP mlp) {

        for(int l = 0; l < mlp.getLayers().size(); l++ ) {
            GradientMatrix weightCorrection = gradients.getWeightGradient(l).multiply(learningRate);
            BiasVector biasGradient = gradients.getBiasGradient(l).multiply(learningRate);

            Layer layer = mlp.getLayer(l);

            layer.getWeightMatrix().substract(weightCorrection);
            layer.getBiasVector().substract(biasGradient);
        }

    }
}
