package Tests;

import Function.ActivationFunction;
import Function.LossFunction;
import MLP.Layer;
import MLP.MLP;
import MLP.Pair;
import Matrices.*;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static Function.ActivationFunction.*;
import static Function.LossFunction.MSE;
import static org.junit.Assert.*;

public class MLPTest {

    private MLP mlp;
    private int inputDim;
    private int numberOfLayers;
    private int outputDim;
    private int batchSize;
    private Random random;
    private MLP mlp1;
    private MLP mlp2;

    @Before
    public void setUp() {
        random = new Random(69);


        mlp = MLP.builder(43)
                .setRandomSeed(2)
                .addLayer(7, ReLU)
                .addLayer(3, ActivationFunction.ReLU)
                .addLayer(12, ReLU)
                .addLayer(17, ActivationFunction.ReLU)
                .addLayer(24, ActivationFunction.Sigmoid)
                .addLayer(24, ActivationFunction.Sigmoid)
                .addLayer(24, ActivationFunction.Sigmoid)
                .addLayer(36, ActivationFunction.Sigmoid)
                .build();

        inputDim = mlp.getDimInput();
        numberOfLayers = mlp.getLayers().size();
        outputDim = mlp.getLayers().getLast().size();
        batchSize = 15;

    }

    private void initializeRandomly(WeightMatrix matrix) {
        matrix.applyToElements((i, j) -> matrix.getData()[i][j] = random.nextDouble() * 0.1 - 0.05);
    }

    private void initializeRandomly(BiasVector vector) {
        vector.applyToElements((i, j) -> vector.getData()[i][j] = random.nextDouble() * 0.1 - 0.05);
    }

    @Test
    public void testFeedForward() {


        List<WeightMatrix> weightsClone = new ArrayList<>();
        List<BiasVector> biasesClone = new ArrayList<>();

        for(Layer layer : mlp.getLayers()) {
            weightsClone.add(layer.getWeightMatrix().clone());
            biasesClone.add(layer.getBiasVector().clone());
        }

        // Créer l'entrée
        ActivationMatrix input = new ActivationMatrix(creerTableau(mlp.getDimInput(), batchSize));

        List<Pair<ActivationMatrix, ActivationMatrix>> activations = mlp.feedForward(input);

        for(int l = 0; l < mlp.getLayers().size(); l++){

            WeightMatrix cloneWeight = weightsClone.get(l);
            WeightMatrix weightOfLayer = mlp.getLayer(l).getWeightMatrix();

            BiasVector cloneBias = biasesClone.get(l);
            BiasVector biasOfLayer = mlp.getLayer(l).getBiasVector();

            assert(cloneWeight.equals(weightOfLayer)) : "Le feedforward a modifié les poids du réseau.";
            assert(cloneBias.equals(biasOfLayer)) : "Le feedforward a modifié les biais du réseau.";
            assert(activations.get(l).getA().getNumberOfRows() == weightOfLayer.getNumberOfRows()) : "Le nombre de neurones dans le résultat du feedforward n'est pas égal au nombre de neuronnes de la couche.";
        }

        assert(activations.size() == mlp.getLayers().size());
    }



    @Test
    public void testMLPBuilder() {
        // Create MLP using builder
        MLP builtMlp = MLP.builder(inputDim)
                .addLayer(numberOfLayers, ReLU)
                .addLayer(outputDim, Sigmoid)
                .build();

        // Verify dimensions
        assertEquals("Input dimension should match", inputDim, builtMlp.getLayer(0).getWeightMatrix().getNumberOfColumns());
        assertEquals("Hidden layer dimensions should match", numberOfLayers, builtMlp.getLayer(0).size());
        assertEquals("Output layer dimensions should match", outputDim, builtMlp.getLayer(1).size());
        assertEquals("MLP should have 2 layers", 2, builtMlp.getLayers().size());

        // Verify activation functions
        assertEquals("Hidden layer activation function should be ReLU",
                ReLU, builtMlp.getLayer(0).getActivationFunction());
        assertEquals("Output layer activation function should be Sigmoid",
                Sigmoid, builtMlp.getLayer(1).getActivationFunction());
    }

    // Test with different activation functions
    @Test
    public void testDifferentActivationFunctions() {
        // Create MLP with tanh and softmax
        MLP customMlp = MLP.builder(inputDim)
                .addLayer(numberOfLayers, TanH)
                .addLayer(outputDim, ReLU)
                .build();

        // Create input matrix
        ActivationMatrix input = createRandomActivationMatrix(inputDim, batchSize);

        // Run feedForward
        List<Pair<ActivationMatrix, ActivationMatrix>> activations = customMlp.feedForward(input);

        // Verify tanh activations (between -1 and 1)
        ActivationMatrix hiddenActivations = activations.get(0).getA();
        for (int i = 0; i < numberOfLayers; i++) {
            for (int j = 0; j < batchSize; j++) {
                double value = hiddenActivations.getData()[i][j];
                assertTrue("Tanh activation should be between -1 and 1", value >= -1 && value <= 1);
            }
        }
    }

    // Helper methods

    private ActivationMatrix createRandomActivationMatrix(int rows, int cols) {
        return createRandomActivationMatrix(rows, cols, -1, 1);
    }

    private ActivationMatrix createRandomActivationMatrix(int rows, int cols, double min, double max) {
        ActivationMatrix matrix = new ActivationMatrix(rows, cols);
        double range = max - min;
        matrix.applyToElements((i, j) -> matrix.getData()[i][j] = random.nextDouble() * range + min);
        return matrix;
    }

    public static double[][] creerTableau(int n, int p){
        double[][] res = new double[n][p];
        int compteur = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                res[i][j] = compteur++;
            }
        }
        return res;
    }
}