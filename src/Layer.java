import java.util.*;

public class Layer {
    int numNodesIn, numNodesOut;
    double[][] weights = new double[numNodesIn][numNodesOut];
    double[] biases = new double[numNodesOut];
    int activationType;

    public Layer(int in, int out) {
        numNodesIn = in;
        numNodesOut = out;
    }

    public double[] getOutputs(double[] inputs) {
        Activation act = new Activation(activationType);
        double[] outputs = new double[numNodesOut];
        for (int i = 0; i < outputs.length; i++) {
            double total = biases[i];
            for (int j = 0; j < inputs.length; j++) {
                total += inputs[j] * weights[j][i];
            }
            outputs[i] = act.get(total);
        }

        return outputs;
    }
}
