/*
 * Copyright (c) Ron Coleman
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package matrix;

import neural.labs.lab03_06.Mop;
import neural.matrix.IMop;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Tests slice from start of matrix.
 * @author Ron.Coleman
 */
//@FixMethodOrder(MethodSorters.DEFAULT)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class SliceStartTest {
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
            {1}
    };


    final double[][] EXPECTED_MATRIX_0 = {
            { 1, 2, 3},
            { 4, 5, 6}
    };

    final double[][] EXPECTED_MATRIX_1 = {
            { 4, 5, 6},
            { 7, 8, 9}
    };

    final double[][] EXPECTED_MATRIX_2 = {
            {1}
    };

    /**
     * Tests that slice matches expectations.
     */
    @Test
    public void test_0() {
        final double[][] slice = mop.slice(TEST_MATRIX_0,0,2);

        int numRows = slice.length;
        assert(numRows == EXPECTED_MATRIX_0.length);

        int numCols = slice[0].length;
        assert(numCols == EXPECTED_MATRIX_0[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(slice[rowno][colno] == EXPECTED_MATRIX_0[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" slice",slice);
    }

    @Test
    public void test_1() {
        final double[][] slice = mop.slice(TEST_MATRIX_0,1,3);

        int numRows = slice.length;
        assert(numRows == EXPECTED_MATRIX_1.length);

        int numCols = slice[0].length;
        assert(numCols == EXPECTED_MATRIX_1[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(slice[rowno][colno] == EXPECTED_MATRIX_1[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" slice",slice);
    }

    @Test
    public void test_2() {
        final double[][] slice = mop.slice(TEST_MATRIX_1,0,1);

        int numRows = slice.length;
        assert(numRows == EXPECTED_MATRIX_2.length);

        int numCols = slice[0].length;
        assert(numCols == EXPECTED_MATRIX_2[0].length);

        IntStream.range(0,numRows).forEach( rowno -> {
            IntStream.range(0,numCols).forEach(colno -> {
                assert(slice[rowno][colno] == EXPECTED_MATRIX_2[rowno][colno]);
            });
        });

        mop.print(this.getClass().getName()+" slice",slice);
    }
}
