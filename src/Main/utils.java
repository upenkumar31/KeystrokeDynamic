package Main;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import static Main.NNConstant.*;
import static Main.algorithm.*;

/**
 * Created by upendra on 8/11/18.
 */
public class utils {
    static int NUM_INPUTS = NNConstant.NUM_INPUTS;
    static int NUM_PATTERNS = NNConstant.NUM_PATTERNS;
    static int NUM_HIDDEN = NNConstant.NUM_HIDDEN;
    static int NUM_EPOCHS = NNConstant.NUM_EPOCHS;
    static double LR_IH =NNConstant.LR_IH;
    static double LR_HO = NNConstant.LR_HO;
    public static double hiddenVal[];
    public static double weightsIH[][];
    public static double weightsHO[];

    private static final String FILENAME = "data.txt";

    static void displayResults()
    {
        int total=0;
        try {
            String filename = "result.txt";
            FileWriter fw = new FileWriter(filename, true); //the true will append the new data
            for (int i = 0; i < NUM_PATTERNS; i++) {
                calcNet(i);
                System.out.println("pat = " + (i + 1) + " actual = " + trainOutput[i] + " neural model = " + outPred);
                if (Math.abs(1 - outPred) < .00000001) {
                    total++;
                }
            }
            System.out.println("total correct:" + total);
            fw.write("total: "+total+"\n");//appends the string to the file
            fw.close();
        }
        catch(IOException ioe)
        {
            System.err.println("IOException: " + ioe.getMessage());
        }
        return;
    }
    static void calcNet(final int patNum)
    {
        // Calculates values for Hidden and Output nodes.
        for(int i = 0; i < NUM_HIDDEN; i++)
        {
            hiddenVal[i] = 0.0;
            for(int j = 0; j < NUM_INPUTS; j++)
            {
                hiddenVal[i] += (trainInputs[patNum][j] * weightsIH[j][i]);
            } // j
            hiddenVal[i] = Math.tanh(hiddenVal[i]);
        } // i

        outPred = 0.0;

        for(int i = 0; i < NUM_HIDDEN; i++)
        {
            outPred += hiddenVal[i] * weightsHO[i];
        }

        errThisPat = outPred - trainOutput[patNum]; // Error = "Expected" - "Actual"
        return;
    }
    static void WeightSave()
    {
        //  Initialize weights to random values.
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(FILENAME);
            bw = new BufferedWriter(fw);
            for(int j = 0; j < NUM_HIDDEN; j++)
            {
                bw.write(Double.toString(weightsHO[j])+'\n');
                for(int i = 0; i < NUM_INPUTS; i++)
                {
                    bw.write(Double.toString(weightsIH[i][j])+'\n');

                } // i
            } // j
            System.out.println("Done");

        } catch (IOException e) {

            e.printStackTrace();

        }
        finally {

            try {

                if (bw != null)
                    bw.close();

                if (fw != null)
                    fw.close();

            } catch (IOException ex) {

                ex.printStackTrace();

            }

        }

    }
    static void WeightChangesHO()
    {
        // Adjust the Hidden to Output weights.
        for(int k = 0; k < NUM_HIDDEN; k++)
        {
            double weightChange = LR_HO * errThisPat * hiddenVal[k];
            weightsHO[k] -= weightChange;

            // Regularization of the output weights.
            //if(weightsHO[k] < -20.0){
            //  weightsHO[k] = -20.0;
            //}else if(weightsHO[k] > 20.0){
            //  weightsHO[k] = 20.0;
            //}
        }
        return;
    }

    static void WeightChangesIH(final int patNum)
    {
        // Adjust the Input to Hidden weights.
        for(int i = 0; i < NUM_HIDDEN; i++)
        {
            for(int k = 0; k < NUM_INPUTS; k++)
            {
                double x = 1 - Math.pow(hiddenVal[i],2);
                x = x * weightsHO[i] * errThisPat * LR_IH;
                x = x * trainInputs[patNum][k];

                double weightChange = x;
                weightsIH[k][i] -= weightChange;
            } // k
        } // i
        return;
    }

    static double calcOverallError()
    {
        double errorValue = 0.0;

        for(int i = 0; i < NUM_PATTERNS; i++)
        {
            calcNet(i);
            errorValue += Math.pow(errThisPat, 2);
        }

        errorValue /= NUM_PATTERNS;

        return Math.sqrt(errorValue);
    }

}
