package tests;

import static org.junit.jupiter.api.Assertions.*;

import matrices.Matrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class MatrixTest {

    private Matrix<?> matrixA;
    private Matrix<?> matrixB;
    private Matrix<?> matrixC;
    private Matrix<?> matrixD;
    private Matrix<?> matrixSquare;

    @BeforeEach
    public void setUp() {
        // Matrices de test standard (2x3)
        double[][] dataA = {
                {1, 2, 3},
                {4, 5, 6}
        };
        double[][] dataB = {
                {7, 8, 9},
                {10, 11, 12}
        };

        // Matrice pour la multiplication (3x2)
        double[][] dataC = {
                {1, 2},
                {3, 4},
                {5, 6}
        };

        // Matrice avec des valeurs négatives et zéro
        double[][] dataD = {
                {-3, 0, 2},
                {5, -1, 0}
        };

        // Matrice carrée pour tester les opérations qui nécessitent une matrice carrée
        double[][] dataSquare = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };

        matrixA = new ConcreteMatrix(dataA);
        matrixB = new ConcreteMatrix(dataB);
        matrixC = new ConcreteMatrix(dataC);
        matrixD = new ConcreteMatrix(dataD);
        matrixSquare = new ConcreteMatrix(dataSquare);
    }

    // Tests des constructeurs et des méthodes de base
    @Test
    public void testConstructor() {
        // Test du constructeur avec dimensions
        Matrix<?> matrix = new ConcreteMatrix(3, 4);
        assertEquals(3, matrix.getNumberOfRows());
        assertEquals(4, matrix.getNumberOfColumns());

        // Vérifier que la matrice est initialisée avec des zéros
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals(0.0, matrix.getData()[i][j], 1e-9);
            }
        }
    }

    @Test
    public void testConstructorWithData() {
        assertEquals(2, matrixA.getNumberOfRows());
        assertEquals(3, matrixA.getNumberOfColumns());
        assertEquals(1.0, matrixA.getData()[0][0], 1e-9);
        assertEquals(6.0, matrixA.getData()[1][2], 1e-9);
    }

    @Test
    public void testCopyConstructor() {
        Matrix<?> copy = new ConcreteMatrix(matrixA);
        assertNotSame(matrixA.getData(), copy.getData());
        assertArrayEquals(matrixA.getData()[0], copy.getData()[0], 1e-9);
        assertArrayEquals(matrixA.getData()[1], copy.getData()[1], 1e-9);
    }

    @Test
    public void testConstructorInvalidDimensions() {
        // Test avec un nombre de lignes négatif
        assertThrows(AssertionError.class, () -> new ConcreteMatrix(-1, 5));
    }

    @Test
    public void testConstructorInvalidDimensionsZeroColumns() {
        // Test avec un nombre de colonnes égal à zéro
        assertThrows(AssertionError.class, () -> new ConcreteMatrix(5, 0));
    }

    // Tests des méthodes de récupération de données
    @Test
    public void testGetData() {
        double[][] data = matrixA.getData();
        assertEquals(1.0, data[0][0], 1e-9);
        assertEquals(6.0, data[1][2], 1e-9);
    }

    @Test
    public void testGetNumberOfColumns() {
        assertEquals(3, matrixA.getNumberOfColumns());
        assertEquals(2, matrixC.getNumberOfColumns());
    }

    @Test
    public void testGetNumberOfRows() {
        assertEquals(2, matrixA.getNumberOfRows());
        assertEquals(3, matrixC.getNumberOfRows());
    }

    @Test
    public void testSize() {
        assertEquals(6, matrixA.size());
        assertEquals(9, matrixSquare.size());
    }

    // Tests des méthodes de clonage
    @Test
    public void testClone() {
        Matrix<?> clone = matrixA.clone();

        // Vérifier que c'est une copie profonde
        assertNotSame(matrixA, clone);
        assertNotSame(matrixA.getData(), clone.getData());

        // Vérifier que les données sont identiques
        for (int i = 0; i < matrixA.getNumberOfRows(); i++) {
            assertArrayEquals(matrixA.getData()[i], clone.getData()[i], 1e-9);
        }

        // Modifier la copie ne devrait pas affecter l'original
        clone.getData()[0][0] = 999;
        assertEquals(1.0, matrixA.getData()[0][0], 1e-9);
    }

    // Tests des opérations de transformation
    @Test
    public void testTranspose() {
        Matrix<?> transposed = matrixA.transpose();

        // Vérifier les dimensions
        assertEquals(3, transposed.getNumberOfRows());
        assertEquals(2, transposed.getNumberOfColumns());

        // Vérifier le contenu
        assertEquals(1.0, transposed.getData()[0][0], 1e-9);
        assertEquals(4.0, transposed.getData()[0][1], 1e-9);
        assertEquals(2.0, transposed.getData()[1][0], 1e-9);
        assertEquals(3.0, transposed.getData()[2][0], 1e-9);
        assertEquals(6.0, transposed.getData()[2][1], 1e-9);
    }

    @Test
    public void testApplyFunction() {
        // Test avec une fonction simple (x -> x + 1)
        Matrix<?> result = matrixA.applyFunction(d -> d + 1);

        assertEquals(2.0, result.getData()[0][0], 1e-9);
        assertEquals(4.0, result.getData()[0][2], 1e-9);
        assertEquals(7.0, result.getData()[1][2], 1e-9);

        // Vérifier que c'est la même instance (mutable)
        assertSame(matrixA, result);
    }

    @Test
    public void testForEach() {
        // Créer un tableau pour stocker la somme (nécessaire pour lambda)
        double[] sum = {0.0};

        // Appliquer forEach pour accumuler la somme
        Matrix<?> result = matrixA.forEach(d -> sum[0] += d);

        // Vérifier la somme
        assertEquals(21.0, sum[0], 1e-9);

        // Vérifier que c'est la même instance (mutable)
        assertSame(matrixA, result);
    }

    // Tests des opérations arithmétiques
    @Test
    public void testAddition() {
        Matrix<?> result = matrixA.clone().add(matrixB);

        // Vérifier quelques valeurs
        assertEquals(8.0, result.getData()[0][0], 1e-9);  // 1 + 7
        assertEquals(10.0, result.getData()[0][1], 1e-9); // 2 + 8
        assertEquals(14.0, result.getData()[1][0], 1e-9); // 4 + 10
        assertEquals(16.0, result.getData()[1][1], 1e-9); // 5 + 11
        assertEquals(18.0, result.getData()[1][2], 1e-9); // 6 + 12
    }

    @Test
    public void testAddScalar() {
        Matrix<?> result = matrixA.clone().add(10);

        // Vérifier quelques valeurs
        assertEquals(11.0, result.getData()[0][0], 1e-9);
        assertEquals(12.0, result.getData()[0][1], 1e-9);
        assertEquals(16.0, result.getData()[1][2], 1e-9);
    }

    @Test
    public void testSubtraction() {
        Matrix<?> result = matrixA.clone().substract(matrixB);

        // Vérifier quelques valeurs
        assertEquals(-6.0, result.getData()[0][0], 1e-9);  // 1 - 7
        assertEquals(-6.0, result.getData()[0][1], 1e-9);  // 2 - 8
        assertEquals(-6.0, result.getData()[0][2], 1e-9);  // 3 - 9
        assertEquals(-6.0, result.getData()[1][0], 1e-9);  // 4 - 10
        assertEquals(-6.0, result.getData()[1][1], 1e-9);  // 5 - 11
        assertEquals(-6.0, result.getData()[1][2], 1e-9);  // 6 - 12
    }

    @Test
    public void testHadamardProduct() {
        Matrix<?> result = matrixA.clone().hadamardProduct(matrixB);

        // Vérifier quelques valeurs
        assertEquals(7.0, result.getData()[0][0], 1e-9);   // 1 * 7
        assertEquals(16.0, result.getData()[0][1], 1e-9);  // 2 * 8
        assertEquals(27.0, result.getData()[0][2], 1e-9);  // 3 * 9
        assertEquals(40.0, result.getData()[1][0], 1e-9);  // 4 * 10
        assertEquals(55.0, result.getData()[1][1], 1e-9);  // 5 * 11
        assertEquals(72.0, result.getData()[1][2], 1e-9);  // 6 * 12
    }

    @Test
    public void testHadamardQuotient() {
        // Créer des matrices avec des valeurs non nulles pour éviter les divisions par zéro
        double[][] dataX = {{2, 4}, {6, 8}};
        double[][] dataY = {{1, 2}, {3, 4}};

        Matrix<?> matrixX = new ConcreteMatrix(dataX);
        Matrix<?> matrixY = new ConcreteMatrix(dataY);

        Matrix<?> result = matrixX.clone().hadamardQuotient(matrixY);

        // Vérifier les résultats
        assertEquals(2.0, result.getData()[0][0], 1e-9);  // 2 / 1
        assertEquals(2.0, result.getData()[0][1], 1e-9);  // 4 / 2
        assertEquals(2.0, result.getData()[1][0], 1e-9);  // 6 / 3
        assertEquals(2.0, result.getData()[1][1], 1e-9);  // 8 / 4
    }

    @Test
    public void testHadamardQuotientAtRight() {
        // Créer des matrices avec des valeurs non nulles pour éviter les divisions par zéro
        double[][] dataX = {{2, 4}, {6, 8}};
        double[][] dataY = {{1, 2}, {3, 4}};

        Matrix<?> matrixX = new ConcreteMatrix(dataX);
        Matrix<?> matrixY = new ConcreteMatrix(dataY);

        Matrix<?> result = matrixX.clone().hadamardQuotientAtRight(matrixY);

        // Vérifier les résultats
        assertEquals(0.5, result.getData()[0][0], 1e-9);  // 1 / 2
        assertEquals(0.5, result.getData()[0][1], 1e-9);  // 2 / 4
        assertEquals(0.5, result.getData()[1][0], 1e-9);  // 3 / 6
        assertEquals(0.5, result.getData()[1][1], 1e-9);  // 4 / 8
    }

    @Test
    public void testMultiplyScalar() {

        Matrix<?> result = matrixA.clone().multiply(2.0);

        // Vérifier quelques valeurs
        assertEquals(2.0, result.getData()[0][0], 1e-9);   // 1 * 2
        assertEquals(4.0, result.getData()[0][1], 1e-9);   // 2 * 2
        assertEquals(6.0, result.getData()[0][2], 1e-9);   // 3 * 2
        assertEquals(8.0, result.getData()[1][0], 1e-9);   // 4 * 2
        assertEquals(10.0, result.getData()[1][1], 1e-9);  // 5 * 2
        assertEquals(12.0, result.getData()[1][2], 1e-9);  // 6 * 2
    }

    @Test
    public void testDivideScalar() {
        Matrix<?> result = matrixA.clone().divide(2.0);

        // Vérifier quelques valeurs
        assertEquals(0.5, result.getData()[0][0], 1e-9);   // 1 / 2
        assertEquals(1.0, result.getData()[0][1], 1e-9);   // 2 / 2
        assertEquals(1.5, result.getData()[0][2], 1e-9);   // 3 / 2
        assertEquals(2.0, result.getData()[1][0], 1e-9);   // 4 / 2
        assertEquals(2.5, result.getData()[1][1], 1e-9);   // 5 / 2
        assertEquals(3.0, result.getData()[1][2], 1e-9);   // 6 / 2
    }

    @Test
    public void testMatrixMultiplication() {
        // A (2x3) * C (3x2) = Result (2x2)
        Matrix<?> result = matrixA.multiply(matrixC);

        // Vérifier les dimensions
        assertEquals(2, result.getNumberOfRows());
        assertEquals(2, result.getNumberOfColumns());

        // Vérifier les valeurs
        // [1 2 3] * [1 2] = [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [22, 28]
        // [4 5 6]   [3 4]   [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6] = [49, 64]
        //           [5 6]
        assertEquals(22.0, result.getData()[0][0], 1e-9);
        assertEquals(28.0, result.getData()[0][1], 1e-9);
        assertEquals(49.0, result.getData()[1][0], 1e-9);
        assertEquals(64.0, result.getData()[1][1], 1e-9);
    }

    @Test
    public void testMultiplyAtRight() {
        // C (3x2) * A (2x3) = Result (3x3)
        Matrix<?> result = matrixA.multiplyAtRight(matrixC);

        // Vérifier les dimensions
        assertEquals(3, result.getNumberOfRows());
        assertEquals(3, result.getNumberOfColumns());

        // Vérifier les valeurs
        // [1 2] * [1 2 3] = [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]
        // [3 4]   [4 5 6]   [3*1 + 4*4, 3*2 + 4*5, 3*3 + 4*6] = [19, 26, 33]
        // [5 6]             [5*1 + 6*4, 5*2 + 6*5, 5*3 + 6*6] = [29, 40, 51]
        assertEquals(9.0, result.getData()[0][0], 1e-9);
        assertEquals(12.0, result.getData()[0][1], 1e-9);
        assertEquals(15.0, result.getData()[0][2], 1e-9);
        assertEquals(19.0, result.getData()[1][0], 1e-9);
        assertEquals(26.0, result.getData()[1][1], 1e-9);
        assertEquals(33.0, result.getData()[1][2], 1e-9);
        assertEquals(29.0, result.getData()[2][0], 1e-9);
        assertEquals(40.0, result.getData()[2][1], 1e-9);
        assertEquals(51.0, result.getData()[2][2], 1e-9);
    }

    @Test
    public void testMultiplyIncompatibleMatrices() {
        // A (2x3) * B (2x3) - incompatible dimensions
        assertThrows(AssertionError.class, () -> matrixA.multiply(matrixB));
    }

    // Tests des opérations mathématiques spécifiques
    @Test
    public void testSum() {
        assertEquals(21.0, matrixA.sum(), 1e-9);  // 1+2+3+4+5+6 = 21
        assertEquals(57.0, matrixB.sum(), 1e-9);  // 7+8+9+10+11+12 = 57
    }

    @Test
    public void testSumOverRows() {
        double[] result = matrixA.sumOverRows();

        // Vérifier les dimensions
        assertEquals(2, result.length);

        // Vérifier les valeurs
        assertEquals(6.0, result[0], 1e-9);  // 1+2+3 = 6
        assertEquals(15.0, result[1], 1e-9); // 4+5+6 = 15
    }

    @Test
    public void testSumOverColumns() {
        double[] result = matrixA.sumOverColumns();

        // Vérifier les dimensions
        assertEquals(3, result.length);

        // Vérifier les valeurs
        assertEquals(5.0, result[0], 1e-9);  // 1+4 = 5
        assertEquals(7.0, result[1], 1e-9);  // 2+5 = 7
        assertEquals(9.0, result[2], 1e-9);  // 3+6 = 9
    }

    @Test
    public void testSignFunction() {
        Matrix<?> result = matrixD.clone().sign();

        // Vérifier quelques valeurs
        assertEquals(-1.0, result.getData()[0][0], 1e-9); // -3 -> -1
        assertEquals(0.0, result.getData()[0][1], 1e-9);  // 0 -> 0
        assertEquals(1.0, result.getData()[0][2], 1e-9);  // 2 -> 1
        assertEquals(1.0, result.getData()[1][0], 1e-9);  // 5 -> 1
        assertEquals(-1.0, result.getData()[1][1], 1e-9); // -1 -> -1
        assertEquals(0.0, result.getData()[1][2], 1e-9);  // 0 -> 0
    }

    @Test
    public void testLogFunction() {
        double[][] dataE = {
                {1, Math.E, Math.pow(Math.E, 2)},
                {Math.pow(Math.E, 3), 1, 1}
        };
        Matrix<?> matrixE = new ConcreteMatrix(dataE);
        Matrix<?> result = matrixE.clone().log();

        // Vérifier quelques valeurs
        assertEquals(0.0, result.getData()[0][0], 1e-9);   // ln(1) = 0
        assertEquals(1.0, result.getData()[0][1], 1e-9);   // ln(e) = 1
        assertEquals(2.0, result.getData()[0][2], 1e-9);   // ln(e²) = 2
        assertEquals(3.0, result.getData()[1][0], 1e-9);   // ln(e³) = 3
    }

    @Test
    public void testLogFunctionSpecialCases() {
        double[][] dataSpecial = {{0, 1}};
        Matrix<?> matrixSpecial = new ConcreteMatrix(dataSpecial);
        Matrix<?> result = matrixSpecial.clone().log();

        // Vérifier le traitement des cas spéciaux
        assertTrue(result.getData()[0][0] < -20);  // ln(0+epsilon) devrait être très négatif
        assertTrue(result.getData()[0][1] < 0);    // ln(1-epsilon) devrait être légèrement négatif
    }

    @Test
    public void testCoshFunction() {
        double[][] dataF = {
                {0, 1},
                {2, 3}
        };
        Matrix<?> matrixF = new ConcreteMatrix(dataF);
        Matrix<?> result = matrixF.clone().cosh();

        // Vérifier quelques valeurs
        assertEquals(Math.cosh(0), result.getData()[0][0], 1e-9);
        assertEquals(Math.cosh(1), result.getData()[0][1], 1e-9);
        assertEquals(Math.cosh(2), result.getData()[1][0], 1e-9);
        assertEquals(Math.cosh(3), result.getData()[1][1], 1e-9);
    }

    @Test
    public void testTanhFunction() {
        double[][] dataF = {
                {0, 1},
                {2, 3}
        };
        Matrix<?> matrixF = new ConcreteMatrix(dataF);
        Matrix<?> result = matrixF.clone().tanh();

        // Vérifier quelques valeurs
        assertEquals(Math.tanh(0), result.getData()[0][0], 1e-9);
        assertEquals(Math.tanh(1), result.getData()[0][1], 1e-9);
        assertEquals(Math.tanh(2), result.getData()[1][0], 1e-9);
        assertEquals(Math.tanh(3), result.getData()[1][1], 1e-9);
    }

    @Test
    public void testSquareFunction() {
        Matrix<?> result = matrixA.clone().square();

        // Vérifier quelques valeurs
        assertEquals(1.0, result.getData()[0][0], 1e-9);   // 1² = 1
        assertEquals(4.0, result.getData()[0][1], 1e-9);   // 2² = 4
        assertEquals(9.0, result.getData()[0][2], 1e-9);   // 3² = 9
        assertEquals(16.0, result.getData()[1][0], 1e-9);  // 4² = 16
        assertEquals(25.0, result.getData()[1][1], 1e-9);  // 5² = 25
        assertEquals(36.0, result.getData()[1][2], 1e-9);  // 6² = 36
    }

    @Test
    public void testNorm() {
        // ||A||² = 1² + 2² + 3² + 4² + 5² + 6² = 91
        // ||A|| = sqrt(91) ≈ 9.539
        assertEquals(Math.sqrt(91), matrixA.norm(), 1e-9);
    }

    @Test
    public void testTrace() {
        // Trace d'une matrice carrée
        assertEquals(15.0, matrixSquare.trace(), 1e-9);  // 1 + 5 + 9 = 15
    }

    @Test
    public void testTraceNonSquareMatrix() {
        // La trace n'est définie que pour les matrices carrées
        assertThrows(AssertionError.class, () -> matrixA.trace());
    }

    // Tests des méthodes utilitaires
    @Test
    public void testIdentityMatrix() {
        Matrix<?> identity = matrixA.createIdentity(3);

        // Vérifier les dimensions
        assertEquals(3, identity.getNumberOfRows());
        assertEquals(3, identity.getNumberOfColumns());

        // Vérifier le contenu
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == j) {
                    assertEquals(1.0, identity.getData()[i][j], 1e-9);
                } else {
                    assertEquals(0.0, identity.getData()[i][j], 1e-9);
                }
            }
        }
    }

    @Test
    public void testHasSameDimensions() {
        // Même dimensions
        assertTrue(matrixA.hasSameDimensions(matrixB));

        // Dimensions différentes
        assertFalse(matrixA.hasSameDimensions(matrixC));
        assertFalse(matrixA.hasSameDimensions(matrixSquare));
    }

    @Test
    public void testVerifyDimensionsFails() {
        // Devrait lancer une AssertionError car les dimensions sont différentes
        assertThrows(AssertionError.class, () -> matrixA.verifyDimensions(matrixC));
    }

    @Test
    public void testVerifyDimensionsSucceeds() {
        // Ne devrait pas lancer d'exception car les dimensions sont identiques
        assertDoesNotThrow(() -> matrixA.verifyDimensions(matrixB));
    }

    @Test
    public void testEquals() {
        // Copie de matrixA
        Matrix<?> copyA = new ConcreteMatrix(matrixA.getData());

        // Test égalité
        assertTrue(matrixA.equals(copyA));
        assertFalse(matrixA.equals(matrixB));

        // Test avec des dimensions différentes
        assertFalse(matrixA.equals(matrixC));
    }

    @Test
    public void testElementWiseOperation() {
        // Test avec une opération personnalisée (multiplication par 2 + addition)
        Matrix<?> result = matrixA.clone().elementWiseOperation((a, b) -> a * 2 + b, matrixB);

        // Vérifier quelques valeurs
        assertEquals(9.0, result.getData()[0][0], 1e-9);   // 1*2 + 7 = 9
        assertEquals(12.0, result.getData()[0][1], 1e-9);  // 2*2 + 8 = 12
        assertEquals(15.0, result.getData()[0][2], 1e-9);  // 3*2 + 9 = 15
        assertEquals(18.0, result.getData()[1][0], 1e-9);  // 4*2 + 10 = 18
        assertEquals(21.0, result.getData()[1][1], 1e-9);  // 5*2 + 11 = 21
        assertEquals(24.0, result.getData()[1][2], 1e-9);  // 6*2 + 12 = 24
    }

    @Test
    public void testDeepCopyArray() {
        double[][] original = {
                {1, 2},
                {3, 4}
        };

        double[][] copy = Matrix.double2DArrayDeepCopy(original);

        // Vérifier que ce n'est pas la même référence
        assertNotSame(original, copy);

        // Vérifier que les tableaux internes sont différents
        assertNotSame(original[0], copy[0]);
        assertNotSame(original[1], copy[1]);

        // Vérifier que les données sont identiques
        assertArrayEquals(original[0], copy[0], 1e-9);
        assertArrayEquals(original[1], copy[1], 1e-9);

        // Modifier l'original ne doit pas affecter la copie
        original[0][0] = 999;
        assertEquals(1.0, copy[0][0], 1e-9);
    }

    /**
     * Implémentation concrète minimale pour tester la classe abstraite Matrix.
     */
    static class ConcreteMatrix extends Matrix<ConcreteMatrix> {
        public ConcreteMatrix(int rows, int cols) {
            super(rows, cols);
        }

        public ConcreteMatrix(double[][] data) {
            super(data);
        }

        public ConcreteMatrix(Matrix<?> source) {
            super(source);
        }

        @Override
        protected ConcreteMatrix createInstance(int rows, int cols) {
            return new ConcreteMatrix(rows, cols);
        }
    }
}
