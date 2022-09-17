package matrix;

import neural.labs.lab03_06.Mop;
import neural.matrix.IMop;
import org.junit.Test;

/**
 * Tests that slice on transpose does not commute.
 * @author Ron.Coleman
 */
public class CommuteTest {
    // TODO: instantiate a concrete IMop here
    IMop mop = new Mop();

    final double[][] TEST_MATRIX_0 = {
            { 1,  2,  3},
            { 4,  5,  6},
            { 7,  8,  9},
            {10, 11, 12},
            {13, 14, 15}
    };

    /**
     * Tests transpose meets minimal expectations.
     */
    @Test
    public void test_0() {
        double[][] sliceTranspose = mop.transpose(mop.slice(TEST_MATRIX_0,0,2));
        mop.print(this.getClass().getName()+" slice, transpose",sliceTranspose);

        double[][] transposeSlice = mop.slice(mop.transpose(TEST_MATRIX_0),0,2);
        mop.print(this.getClass().getName()+" slice, transpose",transposeSlice);

        assert(sliceTranspose.length != transposeSlice.length);
    }

    @Test
    public void test_1() {
        double[][] sliceTranspose = mop.transpose(mop.dice(TEST_MATRIX_0,0,2));
        mop.print(this.getClass().getName()+" dice, transpose",sliceTranspose);

        double[][] transposeSlice = mop.dice(mop.transpose(TEST_MATRIX_0),0,2);
        mop.print(this.getClass().getName()+" dice, transpose",transposeSlice);

        assert(sliceTranspose.length != transposeSlice.length);
    }

    @Test
    public void test_2() {
        double[][] sliceTranspose = mop.transpose(mop.transpose(mop.slice(TEST_MATRIX_0,0,2)));
        mop.print(this.getClass().getName()+" slice, transpose, transpose",sliceTranspose);

        double[][] transposeSlice = mop.slice(mop.transpose(mop.transpose(TEST_MATRIX_0)),0,2);
        mop.print(this.getClass().getName()+" slice, transpose, transpose",transposeSlice);

        assert(sliceTranspose.length == transposeSlice.length);
    }

    @Test
    public void test_3() {
        double[][] sliceTranspose = mop.transpose(mop.transpose(mop.dice(TEST_MATRIX_0,0,2)));
        mop.print(this.getClass().getName()+" dice, transpose, transpose",sliceTranspose);

        double[][] transposeSlice = mop.dice(mop.transpose(mop.transpose(TEST_MATRIX_0)),0,2);
        mop.print(this.getClass().getName()+" dice, transpose, transpose",transposeSlice);

        assert(sliceTranspose.length == transposeSlice.length);
    }
}
