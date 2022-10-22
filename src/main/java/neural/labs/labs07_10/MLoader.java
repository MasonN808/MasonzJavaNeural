package neural.labs.labs07_10;

//import neural.labs.labs03_06.Mop;
import neural.matrix.IMop;
import neural.mnist.IMLoader;
import neural.mnist.MDigit;
import neural.util.IrisHelper;
import org.encog.mathutil.Equilateral;

import java.awt.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.CRC32;


/*
 * Author: Jonathan Murphy
 */

public class MLoader
        implements IMLoader {

    String imagePath;
    String labelsPath;
    int dataMagicNumber;
    int dataNumberOfItems;
    int nRows;
    int nCols;
    int labelMagicNumber;
    int labelNumberOfLabels;
    CRC32 crc;
    MDigit[] MList;
    static final Equilateral eq =
            new Equilateral(10,
                    0,
                    1);

    public MLoader(String imagePath, String labelsPath) {
        this.imagePath = imagePath;
        this.labelsPath = labelsPath;
        this.crc = new CRC32();
    }


    @Override
    public MDigit[] load() throws IOException {

        // Create the image data stream
        DataInputStream imageInputStream =
                new DataInputStream(
                        new BufferedInputStream(
                                new FileInputStream(imagePath)));

        // Create the labels data stream
        DataInputStream labelsInputStream =
                new DataInputStream(
                        new BufferedInputStream(
                                new FileInputStream(labelsPath)));

        // Read in the image data
        this.dataMagicNumber = imageInputStream.readInt();
        this.dataNumberOfItems = imageInputStream.readInt();
        this.nRows = imageInputStream.readInt();
        this.nCols = imageInputStream.readInt();

        // Read in the label data
        this.labelMagicNumber = labelsInputStream.readInt();
        this.labelNumberOfLabels = labelsInputStream.readInt();

        this.MList = new MDigit[this.dataNumberOfItems];

        // Initialize size of list of MDigit relative to number of images and number of pixels of each image
        MDigit[] MList = new MDigit[dataNumberOfItems];

        for (int itemNumber = 0; itemNumber < this.dataNumberOfItems; itemNumber++) {
            //rename
            int [] pixels = new int [nRows * nCols];

            for ( int pixi = 0; pixi < nRows * nCols; pixi++) {
                pixels[pixi] = imageInputStream.readUnsignedByte();
                this.crc.update(pixels[pixi]);
            }
            int label = labelsInputStream.readUnsignedByte();
            this.MList[itemNumber] = new MDigit(itemNumber, pixels, label);
        }
        return this.MList;
    }

    @Override
    public int getPixelsMagic() {
        return 0;
    }

    @Override
    public int getLabelsMagic() {
        return 0;
    }

    @Override
    public long getChecksum() {
        return this.crc.getValue();
    }

    @Override
    public Normal normalize() {
        double[][] pixels = new double[this.dataNumberOfItems][this.nRows * this.nCols];
        double[][] labels = new double[this.dataNumberOfItems][9];

        for (int itemNumber = 0; itemNumber < this.dataNumberOfItems; itemNumber++) {
            int[] arrayPixels = this.MList[itemNumber].pixels;
            double[] normalizedPixels = new double[this.nRows * this.nCols];
            double[] encodedLabel;

            for (int pixelIndex = 0; pixelIndex < this.nRows * this.nCols; pixelIndex++) {
                 normalizedPixels[pixelIndex] = ((double)arrayPixels[pixelIndex])/255.0;
            }

            // Store the normalized pixels
            pixels[itemNumber] = normalizedPixels;

            // Store the encoding
            labels[itemNumber] = eq.encode(this.MList[itemNumber].label);
        }

        return new Normal(pixels, labels);
    }

    public static void main(String args[]) throws IOException {
        MLoader mLoader = new MLoader("/Users/masonnakamura/IdeaProjects/MasonzJavaNeural/data/train-images.idx3-ubyte",
                "/Users/masonnakamura/IdeaProjects/MasonzJavaNeural/data/train-labels.idx1-ubyte");

        MDigit[] mList = mLoader.load();

        mLoader.normalize();
        System.out.println(Arrays.toString(mList[0].pixels));
        System.out.println(Arrays.toString(mList[0].normalizedPixels));
        System.out.println(Arrays.toString(mList[0].encodedLabel));
    }
}