import java.util.*;

public class Activation {
    static final int SIGMOID = 1; //TODO: implement other functions - tanh, ReLU, SiLu
    //also consider linear and softmax for output layer

    int funcType;

    public Activation(int t) {
        funcType = t;
    }

    public double get(double x) {
        if (funcType == SIGMOID) {
            return 1/(1 + Math.exp(-x)); //Math.exp(-x) is the same as e^-x
        }
        System.out.println("activation function error");
        return -1;
    }
    public double getDerivative(double x) {
        if (funcType == SIGMOID) {
            return get(x) * (1-get(x));
        }
        System.out.println("activation function derivative error");
        return -1;
    }
}
