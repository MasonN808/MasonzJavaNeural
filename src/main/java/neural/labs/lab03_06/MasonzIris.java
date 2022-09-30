package neural.labs.lab03_06;

import neural.matrix.IMop;
import neural.util.EncogHelper;
import neural.util.IrisHelper;
import org.apache.commons.math3.stat.StatUtils;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizedField;
import org.encog.util.arrayutil.NormalizationAction;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static neural.util.EncogHelper.*;

/**
 * XOR: This example is essentially the "Hello World" of neural network
 * programming. This example shows how to construct an Encog neural network to
 * predict the report from the XOR operator. This example uses backpropagation
 * to train the neural network.
 *
 * This example attempts to use a minimum of Encog values to create and train
 * the neural network. This allows you to see exactly what is going on. For a
 * more advanced example, that uses Encog factories, refer to the XORFactory
 * example.
 *
 * The original version of this code does not appear to converge. I fixed this
 * problem by using two neurons in the hidden layer and instead of ramped activation,
 * sigmoid activation. This makes the network reflect the model in figure 1.1
 * in the book, d. 11. I also added more comments to make the code more explanatory.
 * @author Ron Coleman
 * @date 24 Oct 2017
 */
public class MasonzIris {
    /**
     * These learning parameters generally give good results according to literature,
     * that is, the training algorithm converges with the tolerance below.
     * */
    public final static double LEARNING_RATE = 0.25;
    public final static double LEARNING_MOMENTUM = 0.25;
    final static double NORMALIZED_HI = 1;
    final static double NORMALIZED_LO = -1;

    static final Equilateral eq =
            new Equilateral(IrisHelper.species2Cat.size(),
                    NORMALIZED_HI,
                    NORMALIZED_LO);

    final static Map<Integer, NormalizedField> normalizers =
            new HashMap<>();

    final static Map<Integer, double[]> encoded_arrays =
            new HashMap<>();

    public static double TRAINING_INPUTS[][];
    public static double TESTING_INPUTS[][];

    public static double TRAINING_IDEALS[][];
    public static double TESTING_IDEALS[][];

    public static double[][] normalize(double[][] src) {
        Mop mop = new Mop();
        // Get a column
        for (int column_index = 0; column_index < src[0].length; column_index++) {
            // Dice, transpose, then flatten the singleton matrix
            double[] transposed_column = mop.transpose(mop.dice(src, column_index, column_index + 1))[0];

            double max = StatUtils.max(transposed_column);
            double min = StatUtils.min(transposed_column);

            NormalizedField normalizedField = new NormalizedField(NormalizationAction.Normalize, null, max, min, NORMALIZED_HI, NORMALIZED_LO);

            // Put normalized field in hashtable
            normalizers.put(column_index, normalizedField);

            for (int row_index = 0; row_index < src.length; row_index++) {
                src[row_index][column_index] = normalizedField.normalize(src[row_index][column_index]);
            }
        }
        return src;
    }

    public static double[][] encode(double[][] src) {
        // TODO fix the +1 to length; very hacky
        double[][] encoded_matrix = new double[src.length][src[0].length+1];
        for (int row_index = 0; row_index < src.length; row_index++) {
            // Convert the double to an int for categorization
            int element = (int) src[row_index][0];
            double[] encoded_array = eq.encode(element);
            encoded_matrix[row_index][0] = encoded_array[0];
            encoded_matrix[row_index][1] = encoded_array[1];

            // Put into hashtable when we decode
            encoded_arrays.put(row_index, encoded_array);
        }
        return encoded_matrix;
    }

    public static void init() {
        IMop mop = new Mop();
        double[][] observations = IrisHelper.load("data/iris.csv");

        // Set the column indexes to remove the id column and the target column
        double[][] observations_ = mop.dice(mop.transpose(observations),0,4);

        double[][] inputs = normalize(observations_);

        // Omit the first row of column labels
        TRAINING_INPUTS = mop.slice(inputs,1,120);
        TESTING_INPUTS = mop.slice(inputs,120,150);

        // This is a column of doubles
        observations_ = mop.dice(mop.transpose(observations),4,5);

        double[][] outputs = encode(observations_);



        TRAINING_IDEALS = mop.slice(outputs,1,120);
        mop.print("TEST", TRAINING_IDEALS);
        TESTING_IDEALS = mop.slice(outputs,120,150);

//        report("training", TRAINING_INPUTS, "inputs");
        report("training",TRAINING_IDEALS, "outputs");
//        report("testing",TESTING_INPUTS, "inputs");
    }

    public static void report(String input_type, double[][] inputs, String input_string) {
        if (input_string.equals("inputs")) {
            System.out.println("--- " + input_type + " inputs");
            for (int id = 0; id < normalizers.size(); id++) {
                if (id == 0) {
                    System.out.println("SL: " + normalizers.get(0).getActualLow() + " - " + normalizers.get(0).getActualHigh());
                } else if (id == 1) {
                    System.out.println("SW: " + normalizers.get(1).getActualLow() + " - " + normalizers.get(1).getActualHigh());
                } else if (id == 2) {
                    System.out.println("PL: " + normalizers.get(2).getActualLow() + " - " + normalizers.get(2).getActualHigh());
                } else {
                    System.out.println("PW: " + normalizers.get(3).getActualLow() + " - " + normalizers.get(3).getActualHigh());
                }
            }

            DecimalFormat f = new DecimalFormat("##.00");
            System.out.println("#  |" + String.format("%-14s|", "SL") + String.format("%-14s|", "SW") +
                    String.format("%-14s|", "PL") + String.format("%-14s|", "PW"));

            for (int row_index = 0; row_index < inputs.length; row_index++) {
                System.out.print(String.format("%-3s|", row_index));
                for (int column_index = 0; column_index < inputs[0].length; column_index++) {
                    System.out.print(String.format("%-14s|", f.format(normalizers.get(column_index).deNormalize(inputs[row_index][column_index])) + " -> "
                            + f.format(inputs[row_index][column_index])));
                }
                System.out.println();
            }
        }

        else if (input_string.equals("outputs")){
            System.out.println("--- " + input_type + " outputs");
            System.out.print("#   ");
            for (int id = 1; id < inputs[0].length+1; id++) {
                System.out.print(String.format("%-8s", "t" + id));
            }

            System.out.print(String.format("%-8s", "decoding"));
            System.out.println();

            DecimalFormat f = new DecimalFormat("#0.0000");

            for (int row_index = 0; row_index < inputs.length; row_index++) {
                System.out.print(String.format("%-4s", row_index));

//                for (int column_index = 0; column_index < inputs[0].length; column_index++) {
//                    System.out.print(String.format("%-8s", f.format(inputs[row_index][column_index])));
//                }
                for (int column_index = 0; column_index < inputs[0].length; column_index++) {
                    System.out.print(String.format("%-8s", f.format(encoded_arrays.get(row_index)[column_index])));
                }

                int decoded_int = eq.decode(encoded_arrays.get(row_index));
                System.out.print(" " + decoded_int);

                switch (decoded_int) {
                    case (0) -> System.out.print(" -> setosa");
                    case (1) -> System.out.print(" -> verginica");
                    case (2) -> System.out.print(" -> versicolor");
                }
                System.out.println();
            }
        }

        else {
            System.out.println("Invalid input_string");
        }

    }

    /**
     * The main method.
     * @param args No arguments are used.
     */
    public static void main(final String args[]) {
        init();

//        // Instantiate the network
//        BasicNetwork network = new BasicNetwork();
//
//        // Input layer plus bias node
//        network.addLayer(new BasicLayer(null, true, 2));
//
//        // Hidden layer plus bias node
//        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
//
//        // Output layer
//        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
//
//        // No more layers to be added
//        network.getStructure().finalizeStructure();
//
//        // Randomize the weights
//        network.reset();
//
//        EncogHelper.describe(network);
//
//        // Create training observations
//        MLDataSet trainingSet = new BasicMLDataSet(TRAINING_INPUTS, TRAINING_IDEALS);
//
//        // Use a training object for the learning algorithm, backpropagation.
////        final BasicTraining training = new Backpropagation(network, trainingSet,LEARNING_RATE,LEARNING_MOMENTUM);
//        final BasicTraining training = new ResilientPropagation(network, trainingSet);
//
//        // Set learning batch size: 0 = batch, 1 = online, n = batch size
//        // See org.encog.neural.networks.training.BatchSize
//        // train.setBatchSize(0);
//
//        int epoch = 0;
//
//        double minError = Double.MAX_VALUE;
//
//        double error = 0.0;
//
//        int sameCount = 0;
//        final int MAX_SAME_COUNT = 5*LOG_FREQUENCY;
//
//        EncogHelper.log(epoch, error,false, false);
//        do {
//            training.iteration();
//
//            epoch++;
//
//            error = training.getError();
//
//            if(error < minError) {
//                minError = error;
//                sameCount = 1;
//            }
//            else
//                sameCount++;
//
//            if(sameCount > MAX_SAME_COUNT)
//                break;
//
//            EncogHelper.log(epoch, error,false,false);
//
//        } while (error > TOLERANCE && epoch < MAX_EPOCHS);
//
//        training.finishTraining();
//
//        EncogHelper.log(epoch, error,sameCount > MAX_SAME_COUNT, true);
//        EncogHelper.report(trainingSet, network);
//        EncogHelper.describe(network);
//
//        Encog.getInstance().shutdown();
    }
}
