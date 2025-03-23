package Tests;

import Function.ActivationFunction;
import Matrices.ActivationMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ActivationFunctionTest {

    private ActivationFunction softmax;
    private ActivationMatrix testMatrix;
    private static final double DELTA = 1e-6; // Précision pour les comparaisons de doubles

    @BeforeEach
    public void setUp() {
        softmax = ActivationFunction.SoftMax;

        // Créer une matrice de test 2x3
        double[][] data = {
                {1.0, 2.0, 3.0},
                {4.0, 1.0, -1.0}
        };
        testMatrix = new ActivationMatrix(data);
    }

    @Test
    public void testSoftMaxValues() {
        // Appliquer softmax
        ActivationMatrix result = softmax.apply(testMatrix.clone());

        // Calcul manuel des valeurs attendues pour vérification
        // Pour la première ligne [1.0, 2.0, 3.0]
        double exp1 = Math.exp(1.0);
        double exp2 = Math.exp(2.0);
        double exp3 = Math.exp(3.0);
        double sum1 = exp1 + exp2 + exp3;

        // Pour la deuxième ligne [4.0, 1.0, -1.0]
        double exp4 = Math.exp(4.0);
        double exp5 = Math.exp(1.0);
        double exp6 = Math.exp(-1.0);
        double sum2 = exp4 + exp5 + exp6;

        // Vérifier les valeurs
        assertEquals(exp1 / sum1, result.getData()[0][0], DELTA);
        assertEquals(exp2 / sum1, result.getData()[0][1], DELTA);
        assertEquals(exp3 / sum1, result.getData()[0][2], DELTA);
        assertEquals(exp4 / sum2, result.getData()[1][0], DELTA);
        assertEquals(exp5 / sum2, result.getData()[1][1], DELTA);
        assertEquals(exp6 / sum2, result.getData()[1][2], DELTA);
    }

    @Test
    public void testSoftMaxSumsToOne() {
        // Appliquer softmax
        ActivationMatrix result = softmax.apply(testMatrix.clone());

        // Vérifier que chaque ligne somme à 1
        double sum1 = result.getData()[0][0] + result.getData()[0][1] + result.getData()[0][2];
        double sum2 = result.getData()[1][0] + result.getData()[1][1] + result.getData()[1][2];

        assertEquals(1.0, sum1, DELTA);
        assertEquals(1.0, sum2, DELTA);
    }

    @Test
    public void testSoftMaxPreservesDominance() {
        // Appliquer softmax
        ActivationMatrix result = softmax.apply(testMatrix.clone());

        // Vérifier que l'ordre des valeurs est préservé
        // Dans la première ligne, 3.0 > 2.0 > 1.0, donc result[0][2] > result[0][1] > result[0][0]
        assertTrue(result.getData()[0][2] > result.getData()[0][1]);
        assertTrue(result.getData()[0][1] > result.getData()[0][0]);

        // Dans la deuxième ligne, 4.0 > 1.0 > -1.0, donc result[1][0] > result[1][1] > result[1][2]
        assertTrue(result.getData()[1][0] > result.getData()[1][1]);
        assertTrue(result.getData()[1][1] > result.getData()[1][2]);
    }

    @Test
    public void testSoftMaxDerivative() {
        // Appliquer softmax
        ActivationMatrix activated = softmax.apply(testMatrix.clone());

        // Appliquer la dérivée
        ActivationMatrix derivative = softmax.getDerivative().apply(activated.clone());

        // Vérifier que la dérivée suit la formule d*(1-d) pour chaque élément
        for (int i = 0; i < activated.getData().length; i++) {
            for (int j = 0; j < activated.getData()[i].length; j++) {
                double expected = activated.getData()[i][j] * (1 - activated.getData()[i][j]);
                assertEquals(expected, derivative.getData()[i][j], DELTA);
            }
        }
    }

    @Test
    public void testSoftMaxWithExtremeValues() {
        // Créer une matrice avec des valeurs extrêmes
        double[][] extremeData = {
                {100.0, 0.0, -100.0}  // Valeurs extrêmes pour tester la stabilité numérique
        };
        ActivationMatrix extremeMatrix = new ActivationMatrix(extremeData);

        // Appliquer softmax
        ActivationMatrix result = softmax.apply(extremeMatrix);

        // Vérifier que le résultat ne contient pas de NaN ou d'Infinity
        for (double[] row : result.getData()) {
            for (double val : row) {
                assertFalse(Double.isNaN(val), "SoftMax ne devrait pas produire de NaN");
                assertFalse(Double.isInfinite(val), "SoftMax ne devrait pas produire d'Infinity");
            }
        }

        // La plus grande valeur devrait dominer (proche de 1.0)
        assertTrue(result.getData()[0][0] > 0.99, "La plus grande valeur devrait être proche de 1.0");

        // La somme devrait toujours être 1.0
        double sum = result.getData()[0][0] + result.getData()[0][1] + result.getData()[0][2];
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void testSoftMaxHandlesZeros() {
        // Créer une matrice avec des zéros
        double[][] zeroData = {
                {0.0, 0.0, 0.0}
        };
        ActivationMatrix zeroMatrix = new ActivationMatrix(zeroData);

        // Appliquer softmax
        ActivationMatrix result = softmax.apply(zeroMatrix);

        // Tous les éléments devraient être égaux (1/3)
        assertEquals(1.0/3.0, result.getData()[0][0], DELTA);
        assertEquals(1.0/3.0, result.getData()[0][1], DELTA);
        assertEquals(1.0/3.0, result.getData()[0][2], DELTA);
    }
}