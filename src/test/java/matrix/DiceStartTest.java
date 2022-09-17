package matrix;

import neural.labs.lab03_06.Mop;
import neural.matrix.IMop;
import org.junit.Test;

import java.util.stream.IntStream;

/**
 * Tests that slice on transpose does not commute.
 * @author Mason.Nakamura
 */
public class DiceStartTest {
    // TODO: instantiate a concrete IMop here
    IMop mop = new Mop();

    final double[][] TEST_MATRIX_0 = {
            { 1,  2,  3},
            { 4,  5,  6},
            { 7,  8,  9},
            {10, 11, 12},
            {13, 14, 15}
    };

    final double[][] TEST_MATRIX_1 = {
            {1}
    };


    final double[][] EXPECTED_MATRIX_0 = {
            { 1,  2},
            { 4,  5},
            { 7,  8},
            {10, 11},
            {13, 14}
    };

    final double[][] EXPECTED_MATRIX_1 = {
            { 2, 3},
            { 5, 6},
            { 8, 9},
            {11, 12},
            {14, 15}
    };

    final double[][] EXPECTED_MATRIX_2 = {
            {1}
    };

    final double[][] EXPECTED_MATRIX_3 = {
            {1, 2},
            {4, 5}
    };

    final double[][] EXPECTED_MATRIX_4 = {
            {1},
            {4}
    };



    /**
     * Tests transpose meets minimal expectations.
     */
    @Test
    public void test_0() {

        final double[][] dice = mop.dice(TEST_MATRIX_0,0,2);

        int numRows = dice.length;
        assert(numRows == EXPECTED_MATRIX_0.length);

        int numCols = dice[0].length;
        assert(numCols == EXPECTED_MATRIX_0[0].length);

        IntStream.range(0,numRows).forEach(rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(dice[rowno][colno] == EXPECTED_MATRIX_0[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" dice",dice);
    }

    @Test
    public void test_1() {
        final double[][] dice = mop.dice(TEST_MATRIX_0,1,3);

        int numRows = dice.length;
        assert(numRows == EXPECTED_MATRIX_1.length);

        int numCols = dice[0].length;
        assert(numCols == EXPECTED_MATRIX_1[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(dice[rowno][colno] == EXPECTED_MATRIX_1[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" dice",dice);
    }

    @Test
    public void test_2() {
        final double[][] dice = mop.dice(TEST_MATRIX_1,0,1);

        int numRows = dice.length;
        assert(numRows == EXPECTED_MATRIX_2.length);

        int numCols = dice[0].length;
        assert(numCols == EXPECTED_MATRIX_2[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(dice[rowno][colno] == EXPECTED_MATRIX_2[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" dice",dice);
    }

    @Test
    public void test_3() {
        final double[][] dice = mop.slice(mop.dice(TEST_MATRIX_0,0,2), 0, 2);

        int numRows = dice.length;
        assert(numRows == EXPECTED_MATRIX_3.length);

        int numCols = dice[0].length;
        assert(numCols == EXPECTED_MATRIX_3[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(dice[rowno][colno] == EXPECTED_MATRIX_3[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" dice, slice",dice);
    }

    @Test
    public void test_4() {
        final double[][] dice = mop.dice(mop.slice(TEST_MATRIX_0,0,2), 0, 2);

        int numRows = dice.length;
        assert(numRows == EXPECTED_MATRIX_3.length);

        int numCols = dice[0].length;
        assert(numCols == EXPECTED_MATRIX_3[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(dice[rowno][colno] == EXPECTED_MATRIX_3[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" slice, dice",dice);
    }

    @Test
    public void test_5() {
        final double[][] dice = mop.dice(mop.slice(mop.dice(TEST_MATRIX_0,0,2), 0, 2), 0 ,1);

        int numRows = dice.length;
        assert(numRows == EXPECTED_MATRIX_4.length);

        int numCols = dice[0].length;
        assert(numCols == EXPECTED_MATRIX_4[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(dice[rowno][colno] == EXPECTED_MATRIX_4[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" dice, slice, dice",dice);
    }

}
