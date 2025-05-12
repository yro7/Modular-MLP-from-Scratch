package Tests;

import static org.junit.Assert.*;

import matrices.Matrix;
import org.junit.Before;
import org.junit.Test;

public class MatrixDimensionsTest {

    private ConcreteMatrix matrix2x3;
    private ConcreteMatrix matrix3x2;
    private ConcreteMatrix matrix3x3;
    private ConcreteMatrix matrix2x2;
    private ConcreteMatrix matrix1x1;

    @Before
    public void setUp() {
        double[][] data2x3 = {
                {1, 2, 3},
                {4, 5, 6}
        };

        double[][] data3x2 = {
                {1, 2},
                {3, 4},
                {5, 6}
        };

        double[][] data3x3 = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };

        double[][] data2x2 = {
                {1, 2},
                {3, 4}
        };

        double[][] data1x1 = {
                {1}
        };

        matrix2x3 = new ConcreteMatrix(data2x3);
        matrix3x2 = new ConcreteMatrix(data3x2);
        matrix3x3 = new ConcreteMatrix(data3x3);
        matrix2x2 = new ConcreteMatrix(data2x2);
        matrix1x1 = new ConcreteMatrix(data1x1);
    }

    @Test
    public void testDimensionGetters() {
        assertEquals(2, matrix2x3.getNumberOfRows());
        assertEquals(3, matrix2x3.getNumberOfColumns());

        assertEquals(3, matrix3x2.getNumberOfRows());
        assertEquals(2, matrix3x2.getNumberOfColumns());

        assertEquals(1, matrix1x1.getNumberOfRows());
        assertEquals(1, matrix1x1.getNumberOfColumns());
    }

    @Test(expected = AssertionError.class)
    public void testZeroRowsInitialization() {
        new ConcreteMatrix(0, 3);
    }

    @Test(expected = AssertionError.class)
    public void testZeroColumnsInitialization() {
        new ConcreteMatrix(3, 0);
    }

    @Test(expected = AssertionError.class)
    public void testNegativeRowsInitialization() {
        new ConcreteMatrix(-2, 3);
    }

    @Test(expected = AssertionError.class)
    public void testNegativeColumnsInitialization() {
        new ConcreteMatrix(3, -2);
    }

    @Test
    public void testTransposeDimensions() {
        Matrix<?> transposed2x3 = matrix2x3.transpose();
        assertEquals(3, transposed2x3.getNumberOfRows());
        assertEquals(2, transposed2x3.getNumberOfColumns());

        Matrix<?> transposed3x3 = matrix3x3.transpose();
        assertEquals(3, transposed3x3.getNumberOfRows());
        assertEquals(3, transposed3x3.getNumberOfColumns());

        Matrix<?> transposed1x1 = matrix1x1.transpose();
        assertEquals(1, transposed1x1.getNumberOfRows());
        assertEquals(1, transposed1x1.getNumberOfColumns());
    }

    @Test
    public void testCloneDimensions() {
        Matrix<?> clone2x3 = matrix2x3.clone();
        assertEquals(2, clone2x3.getNumberOfRows());
        assertEquals(3, clone2x3.getNumberOfColumns());
    }

    @Test(expected = AssertionError.class)
    public void testAdditionIncompatibleDimensions() {
        matrix2x3.add(matrix3x2);
    }

    @Test
    public void testAdditionCompatibleDimensions() {
        Matrix<?> result = matrix2x3.add(new ConcreteMatrix(data2x3()));
        assertEquals(2, result.getNumberOfRows());
        assertEquals(3, result.getNumberOfColumns());
    }

    @Test(expected = AssertionError.class)
    public void testSubtractionIncompatibleDimensions() {
        matrix2x3.substract(matrix3x3);
    }

    @Test(expected = AssertionError.class)
    public void testHadamardProductIncompatibleDimensions() {
        matrix2x3.hadamardProduct(matrix2x2);
    }

    @Test
    public void testMatrixMultiplicationValidDimensions() {
        Matrix<?> result = matrix2x3.multiply(matrix3x2);
        assertEquals(2, result.getNumberOfRows());
        assertEquals(2, result.getNumberOfColumns());
    }

    @Test(expected = AssertionError.class)
    public void testMatrixMultiplicationInvalidDimensions() {
        matrix2x3.multiply(matrix2x2);
    }

    @Test
    public void testMultiplyAtRightValidDimensions() {
        Matrix<?> result = matrix2x3.multiplyAtRight(matrix3x2);
        assertEquals(3, result.getNumberOfRows());
        assertEquals(3, result.getNumberOfColumns());
    }

    @Test(expected = AssertionError.class)
    public void testMultiplyAtRightInvalidDimensions() {
        matrix2x3.multiplyAtRight(matrix3x3);
    }

    @Test
    public void testMatrixSize() {
        assertEquals(6, matrix2x3.size());
        assertEquals(6, matrix3x2.size());
        assertEquals(9, matrix3x3.size());
        assertEquals(4, matrix2x2.size());
        assertEquals(1, matrix1x1.size());
    }

    @Test
    public void testSquareDimensions() {
        Matrix<?> squared = matrix2x3.square();
        assertEquals(2, squared.getNumberOfRows());
        assertEquals(3, squared.getNumberOfColumns());
    }

    @Test
    public void testIdentityMatrixDimensions() {
        Matrix<?> identity3 = matrix2x3.createIdentity(3);
        assertEquals(3, identity3.getNumberOfRows());
        assertEquals(3, identity3.getNumberOfColumns());

        Matrix<?> identity5 = matrix2x3.createIdentity(5);
        assertEquals(5, identity5.getNumberOfRows());
        assertEquals(5, identity5.getNumberOfColumns());
    }

    @Test
    public void testScalarOperationsDimensions() {
        Matrix<?> multiplied = matrix2x3.multiply(2.5);
        assertEquals(2, multiplied.getNumberOfRows());
        assertEquals(3, multiplied.getNumberOfColumns());

        Matrix<?> divided = matrix3x3.divide(2.0);
        assertEquals(3, divided.getNumberOfRows());
        assertEquals(3, divided.getNumberOfColumns());
    }

    @Test
    public void testMatrixCopy() {
        Matrix<?> copy = new ConcreteMatrix(matrix2x3);
        assertEquals(2, copy.getNumberOfRows());
        assertEquals(3, copy.getNumberOfColumns());

        for (int i = 0; i < matrix2x3.getNumberOfRows(); i++) {
            for (int j = 0; j < matrix2x3.getNumberOfColumns(); j++) {
                assertEquals(matrix2x3.getData()[i][j], copy.getData()[i][j], 1e-9);
            }
        }
    }

    /**
     * Helper method to create a 2x3 data array for testing
     */
    private double[][] data2x3() {
        return new double[][] {
                {2, 3, 4},
                {5, 6, 7}
        };
    }

    /**
     * Concrete implementation of Matrix for testing
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