package tests;

import matrices.Matrix;
import matrices.WeightMatrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest2 {

    @Test
    public void testAddMultipliedMatrix() {
        // Arrange
        double[][] dataA = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        };

        double[][] dataB = {
                {0.5, 0.3, 0.1},
                {0.2, 0.4, 0.6}
        };

        // Create copies to verify originals are not modified
        double[][] originalA = Matrix.double2DArrayDeepCopy(dataA);
        double[][] originalB = Matrix.double2DArrayDeepCopy(dataB);

        // Create matrices
        WeightMatrix matrixA = new WeightMatrix(dataA);
        WeightMatrix matrixB = new WeightMatrix(dataB);

        double scalar = 2.5;

        // Expected results: A += B * scalar
        double[][] expectedResult = {
                {1.0 + (0.5 * scalar), 2.0 + (0.3 * scalar), 3.0 + (0.1 * scalar)},
                {4.0 + (0.2 * scalar), 5.0 + (0.4 * scalar), 6.0 + (0.6 * scalar)}
        };

        // Act
        WeightMatrix result = matrixA.addMultipliedMatrix(matrixB, scalar);

        // Assert
        // 1. Check if result is as expected
        for (int i = 0; i < matrixA.getNumberOfRows(); i++) {
            for (int j = 0; j < matrixA.getNumberOfColumns(); j++) {
                assertEquals(expectedResult[i][j], result.getData(i, j), 1e-10);
            }
        }

        // 2. Check if the method returns the same object (for chaining)
        assertSame(matrixA, result);

        // 3. Check if the argument matrix was not modified
        for (int i = 0; i < matrixB.getNumberOfRows(); i++) {
            for (int j = 0; j < matrixB.getNumberOfColumns(); j++) {
                assertEquals(originalB[i][j], matrixB.getData(i, j), 1e-10);
            }
        }
    }

    @Test
    public void testAddMultipliedMatrixWithZeroScalar() {
        // Arrange
        double[][] dataA = {
                {1.0, 2.0},
                {3.0, 4.0}
        };

        double[][] dataB = {
                {5.0, 6.0},
                {7.0, 8.0}
        };

        WeightMatrix matrixA = new WeightMatrix(dataA);
        WeightMatrix matrixB = new WeightMatrix(dataB);

        // Act
        WeightMatrix result = matrixA.addMultipliedMatrix(matrixB, 0.0);

        // Assert
        // With a zero scalar, the values of matrixA should remain unchanged
        for (int i = 0; i < matrixA.getNumberOfRows(); i++) {
            for (int j = 0; j < matrixA.getNumberOfColumns(); j++) {
                assertEquals(dataA[i][j], result.getData(i, j), 1e-10);
            }
        }
    }

    @Test
    public void testAddMultipliedMatrixWithNegativeScalar() {
        // Arrange
        double[][] dataA = {
                {10.0, 20.0},
                {30.0, 40.0}
        };

        double[][] dataB = {
                {1.0, 2.0},
                {3.0, 4.0}
        };

        WeightMatrix matrixA = new WeightMatrix(dataA);
        WeightMatrix matrixB = new WeightMatrix(dataB);

        double scalar = -1.0;  // Negative scalar

        // Expected results: A += B * -1.0 (equivalent to A -= B)
        double[][] expectedResult = {
                {10.0 + (1.0 * scalar), 20.0 + (2.0 * scalar)},
                {30.0 + (3.0 * scalar), 40.0 + (4.0 * scalar)}
        };

        // Act
        WeightMatrix result = matrixA.addMultipliedMatrix(matrixB, scalar);

        // Assert
        for (int i = 0; i < matrixA.getNumberOfRows(); i++) {
            for (int j = 0; j < matrixA.getNumberOfColumns(); j++) {
                assertEquals(expectedResult[i][j], result.getData(i, j), 1e-10);
            }
        }
    }

    @Test
    public void testAddMultipliedMatrixWithDifferentDimensions() {
        // Arrange
        double[][] dataA = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        };

        double[][] dataB = {
                {0.5, 0.3},
                {0.2, 0.4}
        };

        WeightMatrix matrixA = new WeightMatrix(dataA);
        WeightMatrix matrixB = new WeightMatrix(dataB);

        // Act & Assert - should throw AssertionError due to dimension mismatch
        assertThrows(AssertionError.class, () -> matrixA.addMultipliedMatrix(matrixB, 2.0));
    }
}
