package MLP.Optimizers;

import MLP.MLP;
import static MLP.MLP.BackProResult;
import Matrices.BiasVector;
import Matrices.GradientMatrix;
import MLP.Layer;
import Matrices.Matrix;
import Matrices.Utils;

/**
 * Adaptive Moment Estimation
 * Voir <a href="https://arxiv.org/pdf/1412.6980">ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION</a>.
 *
 */
public class Adam extends Optimizer {


    public final double learningRate;
    /**
     * Exponential Decay Rate for First Moment Estimates
     */
    public final double beta1;
    public final double beta2;

    /**
     * Utilisé pour éviter la division par 0. 1e-8 comme dans l'article introduisant Adam.
     */
    public static final double epsilon = 1e-8;

    /**
     * Représente à quelle itération l'optimiseur est
     */
    public int iteration;

    /**
     * Représente les moments de premier ordre (le biais des gradients) estimés des poids du MLP
     */
    public GradientMatrix[] firstOrderMomentsWeights;

    /**
     * Représente les moments de second ordre (la variance des gradients) estimés des poids du MLP
     */
    public GradientMatrix[] secondOrderMomentsWeights;

    /**
     * Représente les moments de premier ordre (le biais des gradients) estimés des biais du MLP
     */
    public BiasVector[] firstOrderMomentsBias;
    /**
     * Représente les moments de second ordre (la variance des gradients) estimés des biais du MLP
     */
    public BiasVector[] secondOrderMomentsBias;

    public BackProResult lastGradients;

    /**
     *
     */
    public Adam(double learningRate, double beta1, double beta2) {
        assert(learningRate > 0): "Le learning rate devrait être strictement positif.";
        assert(beta1 >= 0 && beta1 < 1) : "beta1 devrait appartenir à [0,1(.";
        assert(beta2 > 0 && beta2 < 1) : "beta2 devrait appartenir à [0,1(.";

        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.iteration = 0;
        this.firstOrderMomentsWeights = null;
        this.secondOrderMomentsWeights = null;
        this.firstOrderMomentsBias = null;
        this.secondOrderMomentsBias = null;
    }

    /**
     * Renvoie une nouvelle instance d'ADAM avec des paramètres par défault :
     * LR = 0.001,
     * beta1 = 0.9,
     * beta2 = 0.999.
     */
    public Adam(){
        this(0.001, 0.9, 0.999);
    }

    @Override
    public void updateParameters(BackProResult gradients, MLP mlp) {
        if(lastGradients == null) initialize(gradients);

        int size = gradients.size();

        GradientMatrix[] weightCorrections = new GradientMatrix[size];
        BiasVector[] biasCorrections = new BiasVector[size];

        for(int i = 0; i < size; i++) {

            GradientMatrix weightGradient = gradients.getWeightGradient(i);
            BiasVector biasGradient = gradients.getBiasGradient(i);
 // TODO faire que des opérations in-place
            // TODO implémenter l'initialization bias correction

            // m_t = b1*m_t-1 + (1-b1)*gt
            updateMomentum(firstOrderMomentsWeights[i], weightGradient, beta1);
            updateMomentum(secondOrderMomentsWeights[i], weightGradient, beta2);

            updateMomentum(firstOrderMomentsBias[i], biasGradient, beta1);
            updateMomentum(secondOrderMomentsBias[i], biasGradient, beta2);

            // Calcule la correction finale. Nécessaire de cloner pour éviter
            // de modifier les moments précédemment calculés.
            weightCorrections[i] = firstOrderMomentsWeights[i].clone()
                    // Bias correction: m_t/(1-beta1^t)
                    .divide(1-Math.pow(beta1, iteration))
                    .multiply(learningRate)
                    .hadamardQuotient(
                            secondOrderMomentsWeights[i].clone().sqrt().add(epsilon)
                    );

            biasCorrections[i] = firstOrderMomentsBias[i].clone()
                    // Bias correction: v_t/(1-beta^t)
                    .divide(1-Math.pow(beta1, iteration))
                    .multiply(learningRate)
                    .hadamardQuotient(
                            secondOrderMomentsBias[i].clone().sqrt().add(epsilon)
                    );

        }

        // Met à jour les paramètres avec la correction calculée plus tôt.
        for(int i = 0; i < size; i++) {
            Layer l = mlp.getLayer(i);
            l.getWeightMatrix().substract(weightCorrections[i]);
            l.getBiasVector().substract(biasCorrections[i]);
        }

        lastGradients = gradients;
        this.iteration++;
    }

    /**
     * Initialize l'optimizeur en fonction des premiers gradients reçus.
     * Initialize notamment les moments à 0.
     * @param gradients
     */
    public void initialize(BackProResult gradients) {
        int size = gradients.size();

        this.firstOrderMomentsWeights = new GradientMatrix[size];
        this.secondOrderMomentsWeights = new GradientMatrix[size];
        this.firstOrderMomentsBias = new BiasVector[size];
        this.secondOrderMomentsBias = new BiasVector[size];

        for(int i = 0; i < size; i++) {
            GradientMatrix weightGradient = gradients.getWeightGradient(i);
            this.firstOrderMomentsWeights[i] = new GradientMatrix(Utils.zeroArray(
                    weightGradient.getNumberOfRows(),
                    weightGradient.getNumberOfColumns()
            ));

            this.secondOrderMomentsWeights[i] = new GradientMatrix(Utils.zeroArray(
                    weightGradient.getNumberOfRows(),
                    weightGradient.getNumberOfColumns()
            ));
            // Initialize bias moments
            BiasVector biasGradient = gradients.getBiasGradient(i);
            this.firstOrderMomentsBias[i] = new BiasVector(Utils.zeroArray(
                    1,
                    biasGradient.getNumberOfColumns()
            ));
            this.secondOrderMomentsBias[i] = new BiasVector(Utils.zeroArray(
                    1,
                    biasGradient.getNumberOfColumns()
            ));
        }

        this.lastGradients = gradients;
        this.iteration = 1;
    }


    /**
     * Updates momentum values according to the Adam algorithm
     * @param momentum The momentum matrix to update (either first or second order)
     * @param gradient The current gradient
     * @param beta The decay rate (beta1 for first-order, beta2 for second-order)
     */
    public void updateMomentum(Matrix<?> momentum, Matrix<?> gradient, double beta) {
        Matrix<?> gradientCopy = gradient.clone();

        // Si on update la variance, mettre au carré le clone
        if(beta == beta2) {
            gradientCopy.square();
        }

        // Mets à jour le moment: m_t = beta*m_{t-1} + (1-beta)*g_t
        momentum.multiply(beta)
                .add(gradientCopy.multiply(1-beta));
    }



}
