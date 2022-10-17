package neural.labs.labs07_10;

//import neural.labs.labs03_06.Mop;
import neural.matrix.IMop;
import neural.mnist.IMLoader;
import neural.mnist.MDigit;

import java.awt.*;
import java.io.*;
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
            MList[itemNumber] = new MDigit(itemNumber, pixels, label);
        }
        return MList;
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
        return null;
    }

    public static void main(String args[]) throws IOException {
        MLoader mLoader = new MLoader("/Users/masonnakamura/IdeaProjects/MasonzJavaNeural/data/train-images.idx3-ubyte",
                "/Users/masonnakamura/IdeaProjects/MasonzJavaNeural/data/train-labels.idx1-ubyte");

        MDigit[] mList = mLoader.load();

        System.out.print(mList.length);
        mList[323].toString();
        System.out.println(mLoader.getChecksum());
    }
}