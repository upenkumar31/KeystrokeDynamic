package Main;

import java.io.*;
import java.util.Scanner;
import Main.utils;

import static Main.utils.displayResults;

/**
 * Created by upendra on 8/11/18.
 */
public class algorithm{

    static int NUM_INPUTS = NNConstant.NUM_INPUTS;
    static int NUM_PATTERNS = NNConstant.NUM_PATTERNS;
    static int NUM_HIDDEN = NNConstant.NUM_HIDDEN;
    static int NUM_EPOCHS = NNConstant.NUM_EPOCHS;
    static double LR_IH =NNConstant.LR_IH;
    static double LR_HO = NNConstant.LR_HO;
    public static double hiddenVal[];
    public static double weightsIH[][];
    public static double weightsHO[];

    public static double trainInputs[][];
    public static int trainOutput[];

    private static double errThisPat = NNConstant.errThisPat;
    private static double outPred = NNConstant.outPred;
    private static double RMSerror = NNConstant.RMSerror;

    private static final String FILENAME = "data.txt";

    private static void initWeights()
    {
        //  Initialize weights to random values.
        BufferedReader br = null;
        FileReader fr = null;
        try {
            fr = new FileReader(FILENAME);
            br = new BufferedReader(fr);
            for(int j = 0; j < NUM_HIDDEN; j++)
            {
                weightsHO[j] = Double.parseDouble(br.readLine());
                for(int i = 0; i < NUM_INPUTS; i++)
                {
                    weightsIH[i][j] = Double.parseDouble(br.readLine());
                    System.out.println("Weight = " + weightsIH[i][j]);

                }
            }

        } catch (IOException e) {

            e.printStackTrace();

        } finally {

            try {

                if (br != null)
                    br.close();

                if (fr != null)
                    fr.close();

            } catch (IOException ex) {

                ex.printStackTrace();

            }

        }

    }
    private static void initData(String file)
    {
        File text = new File(file);
        try {
            Scanner scnr = new Scanner(text);

            for (int i = 0; i < NUM_PATTERNS; i++) {
                int k;
                k=0;
                while(k<NUM_INPUTS) {
                    int bits = Float.floatToIntBits((float) scnr.nextDouble());
                    for (int ii=31; ii>=0; --ii) {
                        trainInputs[i][k]=(bits & (1 << ii)) ;
                        k++;
                    }
                }
                trainOutput[i] = 1;
            }

        }catch(FileNotFoundException fp)
        {
            fp.printStackTrace();
        }
        return;
    }
    public void algorithm()
    {
        hiddenVal = new double[NUM_HIDDEN];
        weightsIH =  new double[NUM_INPUTS][NUM_HIDDEN];
        weightsHO = new double[NUM_HIDDEN];
        trainInputs = new double[NUM_PATTERNS][NUM_INPUTS];
        trainOutput = new int[NUM_PATTERNS];
        initWeights();
        initData("train.txt");
        int patNum = 0;
        initWeights();
        initData("fulldata.txt");
        utils obj=new utils();
        while(RMSerror>0.000001){
            for (int i = 0; i < NUM_PATTERNS; i++) {
                patNum = (i);
                obj.calcNet(patNum);
                obj.WeightChangesHO();
                obj.WeightChangesIH(patNum);

            }
            RMSerror = obj.calcOverallError();
            System.out.println("epoch = " + " RMS Error = " + RMSerror);
        }
        displayResults();
        obj.WeightSave();
        System.out.println("error:"+RMSerror);
        displayResults();
    }

}
