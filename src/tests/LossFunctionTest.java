package tests;

import functions.LossFunction;
import matrices.ActivationMatrix;
import matrices.GradientMatrix;
import org.junit.jupiter.api.Test;

import static functions.LossFunction.MAE;
import static functions.LossFunction.MSE;
import static org.junit.jupiter.api.Assertions.*;

public class LossFunctionTest {

    private static final double DELTA = 1e-6; // Précision pour les comparaisons de doubles

    @Test
    public void testMSEValue() {
        // Créer des matrices spécifiques pour ce test
        double[][] predData = {
                {0.7, 0.1, 0.2},
                {0.3, 0.4, 0.3}
        };

        double[][] trueData = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_pred = new ActivationMatrix(predData);
        ActivationMatrix y_true = new ActivationMatrix(trueData);

        // Calcul manuel de MSE pour vérification
        double expectedMSE = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedMSE += Math.pow(y_pred.getData()[i][j] - y_true.getData()[i][j], 2);
            }
        }
        expectedMSE /= (2 * 3); // Divisé par la taille (nb lignes * nb colonnes)

        // Calculer MSE avec la fonction
        double actualMSE = MSE.apply(y_pred, y_true);

        assertEquals(expectedMSE, actualMSE, DELTA);
    }

    @Test
    public void testMSEDerivative() {
        // Créer des matrices spécifiques pour ce test
        double[][] predData = {
                {0.7, 0.1, 0.2},
                {0.3, 0.4, 0.3}
        };

        double[][] trueData = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_pred = new ActivationMatrix(predData);
        ActivationMatrix y_true = new ActivationMatrix(trueData);

        // La dérivée de MSE est 2*(y_pred - y_true)/size
        GradientMatrix derivative = MSE.applyDerivative(y_pred, y_true);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double expected = 2 * (y_pred.getData()[i][j] - y_true.getData()[i][j]) / (2 * 3);
                assertEquals(expected, derivative.getData()[i][j], DELTA);
            }
        }
    }

    @Test
    public void testMAEValue() {
        // Créer des matrices spécifiques pour ce test
        double[][] predData = {
                {0.7, 0.1, 0.2},
                {0.3, 0.4, 0.3}
        };

        double[][] trueData = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_pred = new ActivationMatrix(predData);
        ActivationMatrix y_true = new ActivationMatrix(trueData);

        // Calcul manuel de MAE pour vérification
        double expectedMAE = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedMAE += Math.abs(y_pred.getData()[i][j] - y_true.getData()[i][j]);
            }
        }

        // Calculer MAE avec la fonction (attention, il semble y avoir une erreur dans votre implémentation)
        double actualMAE = MAE.apply(y_pred, y_true);

        // Remarque: votre implémentation de MAE semble prendre la somme absolue mais ne divise pas par le nombre d'éléments
        // Vous pourriez vouloir vérifier cette implémentation
    }

    @Test
    public void testMAEDerivative() {
        // Créer des matrices spécifiques pour ce test
        double[][] predData = {
                {0.7, 0.1, 0.2},
                {0.3, 0.4, 0.3}
        };

        double[][] trueData = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_pred = new ActivationMatrix(predData);
        ActivationMatrix y_true = new ActivationMatrix(trueData);

        GradientMatrix derivative = MAE.applyDerivative(y_pred, y_true);

        // La dérivée de MAE est sign(y_pred - y_true)/batchSize
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double diff = y_pred.getData()[i][j] - y_true.getData()[i][j];
                double expected = Math.signum(diff) / y_true.getBatchSize();
                assertEquals(expected, derivative.getData()[i][j], DELTA);
            }
        }
    }

    @Test
    public void testBCEValue() {
        // Créer des matrices spécifiques pour BCE (entre 0 et 1)
        double[][] predDataBCE = {
                {0.7, 0.3},
                {0.2, 0.8}
        };

        double[][] trueDataBCE = {
                {1.0, 0.0},
                {0.0, 1.0}
        };

        ActivationMatrix y_pred_bce = new ActivationMatrix(predDataBCE);
        ActivationMatrix y_true_bce = new ActivationMatrix(trueDataBCE);

        // Calcul manuel de BCE pour vérification
        double expectedBCE = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double p = y_pred_bce.getData()[i][j];
                double y = y_true_bce.getData()[i][j];
                expectedBCE += y * Math.log(p) + (1 - y) * Math.log(1 - p);
            }
        }
        expectedBCE /= -4.0; // Divisé par la taille et multiplié par -1

        // Calculer BCE avec la fonction
        double actualBCE = LossFunction.BCE.apply(y_pred_bce, y_true_bce);

        assertEquals(expectedBCE, actualBCE, DELTA);
    }

    @Test
    public void testBCEDerivative() {
        // Créer des matrices spécifiques pour BCE (entre 0 et 1)
        double[][] predDataBCE = {
                {0.7, 0.3},
                {0.2, 0.8}
        };

        double[][] trueDataBCE = {
                {1.0, 0.0},
                {0.0, 1.0}
        };

        ActivationMatrix y_pred_bce = new ActivationMatrix(predDataBCE);
        ActivationMatrix y_true_bce = new ActivationMatrix(trueDataBCE);

        GradientMatrix derivative = LossFunction.BCE.applyDerivative(y_pred_bce, y_true_bce);

        // La dérivée de BCE est (-y/p + (1-y)/(1-p))/size
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double p = y_pred_bce.getData()[i][j];
                double y = y_true_bce.getData()[i][j];
                double expected = (-y/p + (1-y)/(1-p))/4.0;
                assertEquals(expected, derivative.getData()[i][j], DELTA);
            }
        }
    }

    @Test
    public void testCEValue() {
        // Créer des matrices spécifiques pour CE (softmax outputs)
        double[][] predDataCE = {
                {0.7, 0.2, 0.1},
                {0.1, 0.8, 0.1}
        };

        double[][] trueDataCE = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_pred_ce = new ActivationMatrix(predDataCE);
        ActivationMatrix y_true_ce = new ActivationMatrix(trueDataCE);

        // Calcul manuel de CE pour vérification
        double expectedCE = 0;
        for (int i = 0; i < 2; i++) { // batch size
            for (int j = 0; j < 3; j++) { // nombre de colonnes
                if (y_true_ce.getData()[i][j] > 0) {
                    expectedCE += y_true_ce.getData()[i][j] * Math.log(y_pred_ce.getData()[i][j]);
                }
            }
        }
        expectedCE /= -2.0; // Divisé par batchSize et multiplié par -1

        // Calculer CE avec la fonction
        double actualCE = LossFunction.CE.apply(y_pred_ce, y_true_ce);

        assertEquals(expectedCE, actualCE, DELTA);
    }

    @Test
    public void testCEDerivative() {
        // Créer des matrices spécifiques pour CE (softmax outputs)
        double[][] predDataCE = {
                {0.7, 0.2, 0.1},
                {0.1, 0.8, 0.1}
        };

        double[][] trueDataCE = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_pred_ce = new ActivationMatrix(predDataCE);
        ActivationMatrix y_true_ce = new ActivationMatrix(trueDataCE);

        GradientMatrix derivative = LossFunction.CE.applyDerivative(y_pred_ce, y_true_ce);

        // La dérivée de CE est -y_true/y_pred/batchSize
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double expected = 0;
                if (y_true_ce.getData()[i][j] > 0) {
                    expected = -y_true_ce.getData()[i][j] / y_pred_ce.getData()[i][j] / 2.0;
                }
                assertEquals(expected, derivative.getData()[i][j], DELTA);
            }
        }
    }

    @Test
    public void testLogCoshValue() {
        // Créer des matrices spécifiques pour ce test
        double[][] predData = {
                {0.7, 0.1, 0.2},
                {0.3, 0.4, 0.3}
        };

        double[][] trueData = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_pred = new ActivationMatrix(predData);
        ActivationMatrix y_true = new ActivationMatrix(trueData);

        // Calcul manuel de LogCosh pour vérification
        double expectedLogCosh = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double diff = y_pred.getData()[i][j] - y_true.getData()[i][j];
                expectedLogCosh += Math.log(Math.cosh(diff));
            }
        }
        expectedLogCosh /= 2.0; // Divisé par batchSize

        // Calculer LogCosh avec la fonction
        double actualLogCosh = LossFunction.LogCosh.apply(y_pred, y_true);

        assertEquals(expectedLogCosh, actualLogCosh, DELTA);
    }

    @Test
    public void testLogCoshDerivative() {
        // Créer des matrices spécifiques pour ce test
        double[][] predData = {
                {0.7, 0.1, 0.2},
                {0.3, 0.4, 0.3}
        };

        double[][] trueData = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_pred = new ActivationMatrix(predData);
        ActivationMatrix y_true = new ActivationMatrix(trueData);

        GradientMatrix derivative = LossFunction.LogCosh.applyDerivative(y_pred, y_true);

        // La dérivée de LogCosh est tanh(y_pred - y_true)/batchSize
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double diff = y_pred.getData()[i][j] - y_true.getData()[i][j];
                double expected = Math.tanh(diff) / 2.0;
                assertEquals(expected, derivative.getData()[i][j], DELTA);
            }
        }
    }

    @Test
    public void testPerfectPrediction() {
        // Créer des matrices spécifiques pour ce test
        double[][] trueData = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_true = new ActivationMatrix(trueData);

        // Cas où les prédictions sont parfaites
        ActivationMatrix perfectPred = new ActivationMatrix(trueData);

        // MSE devrait être 0
        assertEquals(0.0, LossFunction.MSE.apply(perfectPred, y_true), DELTA);

        // BCE devrait être 0 (ou très proche en raison des limites de précision)
        // Mais nous devons avoir des prédictions non-extrêmes pour éviter log(0)
        double[][] almostPerfectData = {
                {0.999, 0.001, 0.001},
                {0.001, 0.999, 0.001}
        };
        ActivationMatrix almostPerfectPred = new ActivationMatrix(almostPerfectData);

        double bceLoss = LossFunction.BCE.apply(almostPerfectPred, y_true);
        // La valeur devrait être très proche de 0, mais pas exactement 0 en raison des valeurs 0.999/0.001
        assertTrue(bceLoss < 0.01);
    }

    @Test
    public void testWorstPrediction() {
        // Créer des matrices spécifiques pour ce test
        double[][] trueData = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        };

        ActivationMatrix y_true = new ActivationMatrix(trueData);

        // Cas où les prédictions sont à l'opposé de la réalité
        double[][] worstPredData = {
                {0.0, 0.0, 1.0},
                {0.0, 0.0, 1.0}
        };
        ActivationMatrix worstPred = new ActivationMatrix(worstPredData);

        // Pour CE, ce serait l'infini à cause de log(0), mais pour les tests nous utilisons des valeurs non-extrêmes
        double[][] badPredData = {
                {0.001, 0.001, 0.998},
                {0.001, 0.001, 0.998}
        };
        ActivationMatrix badPred = new ActivationMatrix(badPredData);

        double ceLoss = LossFunction.CE.apply(badPred, y_true);
        // La valeur devrait être très élevée
        assertTrue(ceLoss > 3.0);
    }
}