package neural.labs.lab03_06;

import java.util.Arrays;
import neural.matrix.IMop;

public class Mop implements IMop {

    @Override
    public double[][] slice(double[][] src, int startRow, int endRow) {

        double[][] src_copy = new double[endRow-startRow][src[0].length];

        int index = 0;
        for (int i=startRow; i<endRow; i++, index++){
            // Make a copy of the current row and append to new matrix
            System.arraycopy(src[i], 0, src_copy[index], 0, src[0].length);
        }

        return src_copy;
    }

    @Override
    public double[][] transpose(double[][] src) {

        // Switch the row and column lengths for transposed matrix
        double[][] src_copy = new double[src[0].length][src.length];

        for (int i=0; i<src[0].length; i++){
            for (int j=0; j<src.length; j++){
                src_copy[i][j] = src[j][i];
            }
        }

        return src_copy;
    }

    @Override
    public double[][] dice(double[][] src, int startCol, int endCol) {

        double[][] src_copy = new double[src.length][src[0].length];

        double[][] transposed_src = transpose(src);

        for (int i=startCol; i<endCol; i++){
            // Make a copy of the current row and append to new matrix
            System.arraycopy(src_copy[i], 0, transposed_src[i], 0, src[0].length);
        }

        // Transpose again to get back to offset the first transpose
        src_copy = transpose(src_copy);

        return src_copy;
    }

    @Override
    public void print(String msg, double[][] src) {

    }
}
