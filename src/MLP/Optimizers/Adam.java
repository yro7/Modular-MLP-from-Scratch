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
     * Garde en mémoire beta1^t et le multiplie par beta1 à chaque itération,
     * pour éviter de recalculer le beta1^t chaque fois.
     */
    public double beta1_t;
    public double beta2_t;

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
        this.beta1_t = beta1;
        this.beta2_t = beta2;
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

            // m_t = b1*m_t-1 + (1-b1)*gt
            // Mettre à jour les moments de 1er & 2nd ordre, pour les poids et les biais.
            updateMomentum(firstOrderMomentsWeights[i], weightGradient, beta1);
            updateMomentum(secondOrderMomentsWeights[i], weightGradient, beta2);

            updateMomentum(firstOrderMomentsBias[i], biasGradient, beta1);
            updateMomentum(secondOrderMomentsBias[i], biasGradient, beta2);


            this.beta1_t *= beta1;
            this.beta2_t *= beta2;

            // Compute bias-corrected first/second moment estimate pour les poids
            GradientMatrix firstOrderMomentsWeightsCorrected =
                    firstOrderMomentsWeights[i].clone()
                            .divide(1-beta1_t);
            GradientMatrix secondOrderMomentsWeightsCorrected =
                    secondOrderMomentsWeights[i].clone()
                            .divide(1-beta2_t);

            // Compute bias-corrected first/second moment estimate pour els biais
            BiasVector firstOrderMomentsBiasCorrected =
                    firstOrderMomentsBias[i].clone()
                            .divide(1-beta1_t);
            BiasVector secondOrderMomentsBiasCorrected =
                    secondOrderMomentsBias[i].clone()
                            .divide(1-beta2_t);

            // Calcule la correction finale pour les poids et biais
            weightCorrections[i] = firstOrderMomentsWeightsCorrected.multiply(learningRate)
                    .hadamardQuotient(secondOrderMomentsWeightsCorrected.sqrt().add(epsilon));
            biasCorrections[i] = firstOrderMomentsBiasCorrected.multiply(learningRate)
                    .hadamardQuotient(secondOrderMomentsBiasCorrected.sqrt().add(epsilon));



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

        // Multiplier le moment par beta
        momentum.multiply(beta);

        // Si on update la variance, multiplier par le carré du gradient
        if(beta == beta2) {
            momentum.addMultipliedMatrix(gradient.clone().square(), (1-beta));
        }

        // Sinon, multiplier par le gradient
        // Mettre à jour le moment: m_t = beta*m_{t-1} + (1-beta)*g_t
        momentum.addMultipliedMatrix(gradient, (1-beta));
    }



}
