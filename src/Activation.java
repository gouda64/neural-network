import java.io.Serializable;
import java.util.*;

public class Activation implements Serializable {
    static final int LINEAR = 0; //only really useful for output layer
    static final int SIGMOID = 1; //TODO: implement other functions - tanh, ReLU, SiLu
    //also consider softmax for output layer

    int funcType;

    public Activation(int t) {
        funcType = t;
    }

    public double get(double x) {
        switch(funcType) {
            case LINEAR:
                return x;
            case SIGMOID:
                return 1/(1 + Math.exp(-x)); //Math.exp(-x) is the same as e^-x
            default:
                System.out.println("activation function error");
                return -1;
        }
    }
    public double getDerivative(double x) {
        switch(funcType) {
            case LINEAR:
                return 1;
            case SIGMOID:
                return get(x) * (1-get(x));
            default:
                System.out.println("activation function derivative error");
                return -1;
        }
    }
}
