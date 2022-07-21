package jan.matrix;

public interface IMop {
    double[][] slice(double[][] src,int startRow,int endRow);
    double[][] transpose(double[][] src);
    double[][] dice(double[][] src,int startCol, int endCol);
    void print(String msg, double[][] src);
}