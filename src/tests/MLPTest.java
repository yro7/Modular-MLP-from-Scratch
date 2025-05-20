package tests;

import static org.junit.jupiter.api.Assertions.*;

import matrices.BiasVector;
import matrices.WeightMatrix;
import mlps.Layer;
import mlps.MLP;
import mlps.optimizers.SGD;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import matrices.ActivationMatrix;
import functions.LossFunction;
import mlps.optimizers.Optimizer;
import mlps.trainer.Trainer;

import java.util.ArrayList;
import java.util.List;

public class MLPTest {

    private MLP mlp;
    private int dimInput = 3;
    private List<Layer> layers;

    @BeforeEach
    public void setUp() {
        // Initialisation des couches pour le test
        layers = new ArrayList<>();
        layers.add(new Layer(new WeightMatrix(3, 4), new BiasVector(4))); // Couche 1
        layers.add(new Layer(new WeightMatrix(4, 2), new BiasVector(2))); // Couche 2

        mlp = new MLP(layers, dimInput);
    }

    @Test
    public void testFeedForward() {
        ActivationMatrix input = new ActivationMatrix(new double[][]{{1, 0, 1}});
        MLP.FeedForwardResult result = mlp.feedForward(input);

        assertNotNull(result);
        assertEquals(2, result.results.size()); // Vérifie que le résultat contient les activations pour chaque couche
    }

    @Test
    public void testComputeLoss() {
        ActivationMatrix input = new ActivationMatrix(new double[][]{{1, 0, 1}});
        ActivationMatrix expectedOutput = new ActivationMatrix(new double[][]{{1, 0}});
        LossFunction lossFunction = LossFunction.MSE;

        double loss = mlp.computeLoss(input, expectedOutput, lossFunction);
        assertTrue(loss >= 0); // Vérifie que la perte est un nombre positif
    }

    @Test
    public void testTrain() {
        Trainer trainer = new Trainer() {
            @Override
            public void train(MLP mlp) {
                // Implémentation simplifiée de l'entraînement
            }
        };

        MLP trainedMlp = mlp.train(trainer);
        assertNotNull(trainedMlp);
    }

    @Test
    public void testUpdateParameters() {
        ActivationMatrix input = new ActivationMatrix(new double[][]{{1, 0, 1}});
        ActivationMatrix expectedOutput = new ActivationMatrix(new double[][]{{1, 0}});
        LossFunction lossFunction = LossFunction.MSE;
        Optimizer optimizer = new SGD(0.1);

        mlp.updateParameters(input, expectedOutput, lossFunction, optimizer, null);
        assertTrue(true);
    }

    @Test
    public void testBackpropagate() {
        ActivationMatrix input = new ActivationMatrix(new double[][]{{1, 0, 1}});
        ActivationMatrix expectedOutput = new ActivationMatrix(new double[][]{{1, 0}});
        LossFunction lossFunction = LossFunction.MSE;
        MLP.BackProResult result = mlp.backpropagate(input, expectedOutput, lossFunction, null);
        assertNotNull(result);
        assertEquals(2, result.size()); // Vérifie que le résultat contient les gradients pour chaque couche
    }

    @Test
    public void testSerializeAndImportModel() {
        String modelName = "testModel";
        mlp.serialize(modelName);

        MLP importedMlp = MLP.importModel(modelName);
        assertNotNull(importedMlp);
        assertEquals(mlp.getDimInput(), importedMlp.getDimInput());
        assertEquals(mlp.getLayers().size(), importedMlp.getLayers().size());
    }
}
