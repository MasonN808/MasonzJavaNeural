package matrix;

import neural.labs.lab03_06.Mop;
import neural.matrix.IMop;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

import java.util.stream.IntStream;

/**
 * Tests slice from start of matrix.
 * @author Mason.Nakamura
 */
//@FixMethodOrder(MethodSorters.DEFAULT)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class TransposeTest {
    // TODO: instantiate a concrete IMop here
    IMop mop = new Mop();

    // Matrix is this size to anticipate start, mid, end testing.
    final double[][] TEST_MATRIX_0 = {
            { 1,  2,  3},
            { 4,  5,  6},
            { 7,  8,  9},
            {10, 11, 12},
            {13, 14, 15}
    };

    // Singleton Matrix
    final double[][] TEST_MATRIX_1 = {
            { 1, 2, 3, 4, 5}
    };

    final double[][] TEST_MATRIX_2 = {
            {1}
    };

    final double[][] EXPECTED_MATRIX_0 = {
            { 1, 4, 7, 10, 13},
            { 2, 5, 8, 11, 14},
            { 3, 6, 9, 12, 15}
    };

    final double[][] EXPECTED_MATRIX_1 = {
            { 1},
            { 2},
            { 3},
            { 4},
            { 5}
    };

    final double[][] EXPECTED_MATRIX_2 = {
            {1}
    };

    /**
     * Tests that slice matches expectations.
     */
    @Test
    public void test_0() {
        final double[][] transpose = mop.transpose(TEST_MATRIX_0);

        int numRows = transpose.length;
        assert(numRows == EXPECTED_MATRIX_0.length);

        int numCols = transpose[0].length;
        assert(numCols == EXPECTED_MATRIX_0[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(transpose[rowno][colno] == EXPECTED_MATRIX_0[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" transpose",transpose);
    }

    @Test
    public void test_1() {
        final double[][] transpose = mop.transpose(TEST_MATRIX_1);

        int numRows = transpose.length;
        assert(numRows == EXPECTED_MATRIX_1.length);

        int numCols = transpose[0].length;
        assert(numCols == EXPECTED_MATRIX_1[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(transpose[rowno][colno] == EXPECTED_MATRIX_1[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" transpose",transpose);
    }

    @Test
    public void test_2() {
        final double[][] transpose = mop.transpose(TEST_MATRIX_2);

        int numRows = transpose.length;
        assert(numRows == EXPECTED_MATRIX_2.length);

        int numCols = transpose[0].length;
        assert(numCols == EXPECTED_MATRIX_2[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(transpose[rowno][colno] == EXPECTED_MATRIX_2[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" transpose",transpose);
    }
}
