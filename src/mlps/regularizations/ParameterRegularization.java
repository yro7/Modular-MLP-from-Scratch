package mlps.regularizations;

import mlps.MLP;
import matrices.GradientMatrix;
import matrices.WeightMatrix;

import java.util.function.BiConsumer;


/**
 * Représente une Régularization appliquée aux Paramètres du MLP.
 * La régularization sera utilisée lors de {@link MLP.MLP#backpropagate} et appliquera
 * une pénalité aux gradients calculés à chaque couche.
 * Exemple : L1, L2
 */
public abstract class ParameterRegularization implements BiConsumer<GradientMatrix, WeightMatrix>{

    public ParameterRegularization(double regularizationRate) {
        this.regularizationRate = regularizationRate;
    }

    public double regularizationRate;

    /**
     * Modifie le gradient pour y ajouter une pénalité en fonction de la matrice de poids
     * de la couche du {@link MLP}.
     *
     *
     * @param gradientMatrix the first input argument
     * @param weightMatrix the second input argument
     */
    @Override
    public void accept(GradientMatrix gradientMatrix, WeightMatrix weightMatrix) {
        WeightMatrix penalty = this.computePenalty(weightMatrix);
        gradientMatrix.add(penalty);
    }

    public abstract double computePenalty(MLP mlp);

    public abstract WeightMatrix computePenalty(WeightMatrix gradientMatrix);

    /**
     * Régression LASSO
     * Ajoute au gradient un lambda en fonction de si
     * un poids neuronal est positif ou négatif.
     */
    public static class L1 extends ParameterRegularization {

        public L1(double regularizationRate) {
            super(regularizationRate);
        }

        @Override
        public double computePenalty(MLP mlp) {
            return mlp.getWeightsNorm() * regularizationRate;
        }


        @Override
        public WeightMatrix computePenalty(WeightMatrix weightMatrix) {
            return weightMatrix.clone().sign().multiply(regularizationRate);
        }
    }

    /**
     * Régression Ridge
     * Ajoute au gradient le taux de régularization * le poids neuronal.
     */
    public class L2 extends ParameterRegularization {

        public L2(double regularizationRate) {
            super(regularizationRate);
        }

        @Override
        public double computePenalty(MLP mlp) {
            return mlp.mapWeightsToDouble(d -> Math.pow(d,2)) * regularizationRate;
        }

        @Override
        public WeightMatrix computePenalty(WeightMatrix weightMatrix) {
            return weightMatrix.clone().multiply(regularizationRate);
        }
    }

    /**
     * Voir <a href="https://en.wikipedia.org/wiki/Elastic_net_regularization">Elastic Net Regularization</a>
     * Combine à la fois une régularization L1 et une L2.
     */
    public static class ElasticNet extends ParameterRegularization {

        public double regularizationRateL2;

        public ElasticNet(double regularizationRateL1, double regularizationRateL2) {
            super(regularizationRateL1);
            this.regularizationRateL2 = regularizationRateL2;
        }

        @Override
        public double computePenalty(MLP mlp) {
            return mlp.getWeightsNorm() * regularizationRateL2 +
                    mlp.mapWeightsToDouble(d -> Math.pow(d,2))*regularizationRateL2;
        }

        @Override
        public WeightMatrix computePenalty(WeightMatrix gradientMatrix) {
            return gradientMatrix.clone().sign().multiply(regularizationRate) // L1
                    .addMultipliedMatrix(gradientMatrix, 2*regularizationRate); // L2
        }

    }

}
