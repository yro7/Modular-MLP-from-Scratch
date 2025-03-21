package Tests;

import static org.junit.Assert.*;

import Matrices.Matrix;
import org.junit.Before;
import org.junit.Test;

public class MatrixTest {

    private Matrix<?> matrixA;
    private Matrix<?> matrixB;

    @Before
    public void setUp() {
        double[][] dataA = {
                {1, 2, 3},
                {4, 5, 6}
        };
        double[][] dataB = {
                {7, 8, 9},
                {10, 11, 12}
        };
        matrixA = new ConcreteMatrix(dataA);
        matrixB = new ConcreteMatrix(dataB);
    }

    @Test
    public void testMatrixInitialization() {
        assertEquals(2, matrixA.getNumberOfRows());
        assertEquals(3, matrixA.getNumberOfColumns());
        assertEquals(1.0, matrixA.getData()[0][0], 1e-9);
        assertEquals(6.0, matrixA.getData()[1][2], 1e-9);
    }

    @Test
    public void testClone() {
        Matrix<?> clone = matrixA.clone();
        matrixA.print();
        clone.print();

        assertNotSame(matrixA, clone);
        assertArrayEquals(matrixA.getData(), clone.getData());
    }

    @Test
    public void testTranspose() {
        Matrix<?> transposed = matrixA.transpose();
        assertEquals(3, transposed.getNumberOfRows());
        assertEquals(2, transposed.getNumberOfColumns());
        assertEquals(1.0, transposed.getData()[0][0], 1e-9);
        assertEquals(4.0, transposed.getData()[0][1], 1e-9);
    }

    @Test
    public void testAddition() {

        Matrix<?> result = matrixA.add(matrixB);

        result.print();
        assertEquals(8.0, result.getData()[0][0], 1e-9);
        assertEquals(16.0, result.getData()[1][1], 1e-9);
    }

    @Test
    public void testSubtraction() {
        Matrix<?> result = matrixA.substract(matrixB);
        assertEquals(-6.0, result.getData()[0][0], 1e-9);
        assertEquals(-6.0, result.getData()[1][1], 1e-9);
    }

    @Test
    public void testHadamardProduct() {
        Matrix<?> result = matrixA.hadamardProduct(matrixB);
        assertEquals(7.0, result.getData()[0][0], 1e-9);
        assertEquals(55.0, result.getData()[1][1], 1e-9);
    }

    @Test
    public void testMultiplyScalar() {
        Matrix<?> result = matrixA.multiply(2.0);
        assertEquals(2.0, result.getData()[0][0], 1e-9);
        assertEquals(12.0, result.getData()[1][2], 1e-9);
    }

    @Test
    public void testDivideScalar() {
        Matrix<?> result = matrixA.divide(2.0);
        assertEquals(0.5, result.getData()[0][0], 1e-9);
        assertEquals(3.0, result.getData()[1][2], 1e-9);
    }

    @Test
    public void testMatrixMultiplication() {
        double[][] dataC = {
                {1, 2},
                {3, 4},
                {5, 6}
        };

        matrixA.print();
        System.out.println();
        System.out.println();

        Matrix<?> matrixC = new ConcreteMatrix(dataC);
        Matrix<?> result = matrixA.multiply(matrixC);
        matrixC.print();
        System.out.println();
        result.print();


        assertEquals(2, result.getNumberOfRows());
        assertEquals(2, result.getNumberOfColumns());
        assertEquals(22.0, result.getData()[0][0], 1e-9);
        assertEquals(28.0, result.getData()[0][1], 1e-9);
        assertEquals(49.0, result.getData()[1][0], 1e-9);
        assertEquals(64.0, result.getData()[1][1], 1e-9);
    }

    @Test
    public void testSum() {
        assertEquals(21.0, matrixA.sum(), 1e-9);
    }

    @Test
    public void testSignFunction() {
        double[][] dataD = {
                {-3, 0, 2},
                {5, -1, 0}
        };
        Matrix<?> matrixD = new ConcreteMatrix(dataD);
        Matrix<?> result = matrixD.sign();

        assertEquals(-1.0, result.getData()[0][0], 1e-9);
        assertEquals(0.0, result.getData()[0][1], 1e-9);
        assertEquals(1.0, result.getData()[0][2], 1e-9);
    }

    @Test
    public void testLogFunction() {
        double[][] dataE = {
                {1, Math.E, Math.pow(Math.E, 2)},
                {Math.pow(Math.E, 3), 1, 1}
        };
        Matrix<?> matrixE = new ConcreteMatrix(dataE);
        Matrix<?> result = matrixE.log();

        matrixE.print();
        result.print();
        assertEquals(0.0, result.getData()[0][0], 1e-9);
        assertEquals(1.0, result.getData()[0][1], 1e-9);
        assertEquals(2.0, result.getData()[0][2], 1e-9);
    }

    @Test
    public void testCoshFunction() {
        double[][] dataF = {
                {0, 1},
                {2, 3}
        };
        Matrix<?> matrixF = new ConcreteMatrix(dataF);
        Matrix<?> result = matrixF.cosh();

        assertEquals(Math.cosh(0), result.getData()[0][0], 1e-9);
        assertEquals(Math.cosh(3), result.getData()[1][1], 1e-9);
    }

    @Test
    public void testSumOverRows() {
        // Effectuer la somme des lignes de la matrice
        double[][] result = matrixA.sumOverRows();

        // Vérifier les résultats de la somme pour chaque ligne
        assertEquals(6.0, result[0][0], 1e-9); // 1 + 2 + 3 = 6
        assertEquals(15.0, result[1][0], 1e-9); // 4 + 5 + 6 = 15
    }


    @Test
    public void testSquareFunction() {
        Matrix<?> result = matrixA.square();
        assertEquals(1.0, result.getData()[0][0], 1e-9);
        assertEquals(25.0, result.getData()[1][1], 1e-9);
    }

    @Test
    public void testIdentityMatrix() {
        Matrix<?> identity = new ConcreteMatrix(3, 3).createIdentity(3);
        assertEquals(1.0, identity.getData()[0][0], 1e-9);
        assertEquals(1.0, identity.getData()[1][1], 1e-9);
        assertEquals(1.0, identity.getData()[2][2], 1e-9);
        assertEquals(0.0, identity.getData()[0][1], 1e-9);
        assertEquals(0.0, identity.getData()[2][0], 1e-9);
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

        @Override
        protected ConcreteMatrix createInstance(int rows, int cols) {
            return new ConcreteMatrix(rows, cols);
        }
    }
}
