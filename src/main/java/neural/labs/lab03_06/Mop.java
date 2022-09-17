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

        double[][] transposed_src = transpose(src);

        return transpose(slice(transposed_src, startCol, endCol));
    }

    @Override
    public void print(String msg, double[][] src) {
        System.out.println("-----------------------------------");
        System.out.println(msg);
        for (int row_index=0; row_index<src.length; row_index++){
            // Print the next row
            System.out.println();
            for (int col_index=0; col_index<src[0].length; col_index++){
                System.out.print(src[row_index][col_index] + " ");
            }
        }
        System.out.println();
    }
}
