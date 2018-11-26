package Main;

/**
 * Created by upendra on 8/11/18.
 */
public class NNConstant {
    static final int NUM_INPUTS = 31*32;
    static final int NUM_PATTERNS = 142;
    static final int NUM_HIDDEN = 30;
    static final int NUM_EPOCHS = 200;
    static final double LR_IH = 0.7;
    static final double LR_HO = 0.07;
    static double errThisPat = 0.0;
    static double outPred = 0.0;
    static double RMSerror = 0.0;
}
