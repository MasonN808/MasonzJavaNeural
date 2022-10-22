package neural.mnist;

/**
 * Container of MNIST digits
 */
public class MDigit {

    int no;
    public int[] pixels;
    public int label;
    public double[] normalizedPixels;
    public double[] encodedLabel;

    public MDigit(int no, int[] pixels, int label) {
        this.no = no;
        this.pixels = pixels;
        this.label = label;
        this.normalizedPixels = new double[785];
        this.encodedLabel = new double[9];
    }

    @Override
    public String toString() {
        System.out.println("--- testing");
        System.out.println("#" + this.no + " label: " + this.label);
        System.out.println("  0123456789012345678901234567");
        for (int rowIndex=0; rowIndex < 28; rowIndex++) {
            System.out.print(rowIndex % 10 + " ");
            for (int colIndex=0; colIndex < 28; colIndex++) {
                if (this.pixels[colIndex + rowIndex*28] == 0) {
                    System.out.print(".");
                }
                else {
                    // Discretize the ubyte
                    int undiscretizedPixelValue = this.pixels[colIndex + rowIndex*28];
                    if (undiscretizedPixelValue <= 256/15) {
                        System.out.printf("%1d", 1);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*2){
                        System.out.printf("%1d", 2);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*3){
                        System.out.printf("%1d", 3);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*4){
                        System.out.printf("%1d", 4);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*5){
                        System.out.printf("%1d", 5);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*6){
                        System.out.printf("%1d", 6);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*7){
                        System.out.printf("%1d", 7);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*8){
                        System.out.printf("%1d", 8);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*9){
                        System.out.printf("%1d", 9);
                    }
                    else if (undiscretizedPixelValue <= (256/15)*10){
                        System.out.printf("%1s", "A");
                    }
                    else if (undiscretizedPixelValue <= (256/15)*11){
                        System.out.printf("%1s", "B");
                    }
                    else if (undiscretizedPixelValue <= (256/15)*12){
                        System.out.printf("%1s", "C");
                    }
                    else if (undiscretizedPixelValue <= (256/15)*13){
                        System.out.printf("%1s", "D");
                    }
                    else if (undiscretizedPixelValue <= (256/15)*14){
                        System.out.printf("%1s", "E");
                    }
                    else if (undiscretizedPixelValue <= 256){
                        System.out.printf("%1.1s", "F");
                    }
                }
            }
            System.out.println();
        }
        return null;
    }
}
