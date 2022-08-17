public class Layer {
    int nodesIn, nodesOut;
    double[][] weights = new double[nodesIn][nodesOut];
    double[] biases = new double[nodesOut];
    double[][] weightGradients = new double[nodesIn][nodesOut];
    double[] biasGradients = new double[nodesOut];
    double[] inputs = new double[nodesIn];
    double[] preActivationOutputs = new double[nodesOut];

    Activation activation = new Activation(Activation.SIGMOID); //default activation function in case something falls through

    public Layer(int in, int out) {
        nodesIn = in;
        nodesOut = out;
        initWeights();
    }

    public double[] getOutputs(double[] inputs) {
        double[] outputs = new double[nodesOut];
        for (int i = 0; i < outputs.length; i++) {
            double total = biases[i];
            for (int j = 0; j < inputs.length; j++) {
                this.inputs[j] = inputs[j];
                total += inputs[j] * weights[j][i];
            }
            preActivationOutputs[i] = total;
            outputs[i] = activation.get(total);
        }

        return outputs;
    }

    public void addGradients(double learnRate) { //higher learn rate means more speed but less accuracy
        for (int i = 0; i < nodesOut; i++) {
            biases[i] -= biasGradients[i] * learnRate;
            for (int j = 0; j < nodesIn; j++) {
                weights[j][i] -= weightGradients[j][i] * learnRate;
            }
        }
    }
    public void updateGradients(double[] derivConstants) {
        for (int i = 0; i < nodesOut; i++) {
            for (int j = 0; j < nodesIn; j++) {
                //partial derivative of weighted output to weight is just the input
                weightGradients[j][i] += inputs[j] * derivConstants[i];
            }
            biasGradients[i] += 1 * derivConstants[i];
            //partial derivative of the weighted output to bias is just one
        }
    }
    public double[] getDerivConstants(Layer oldLayer, double[] oldDerivConstants) { //don't use for output layer
        double[] derivConstants = new double[nodesOut];
        for (int j = 0; j < derivConstants.length; j++) {
            for (int k = 0; k < oldDerivConstants.length; k++) {
                derivConstants[j] += oldLayer.weights[j][k] * oldDerivConstants[k];
            }
            derivConstants[j] *= activation.getDerivative(preActivationOutputs[j]);
        }
        return derivConstants;
    }

    public void initWeights() { //biases are okay to start at 0
        // TODO: add different weight initializations (after adding activation funcs)
        //  https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        for (int i = 0; i < nodesIn; i++) {
            for (int j = 0; j < nodesOut; j++) {
                if (activation.funcType == Activation.SIGMOID) { //or tanh -> Xavier initialization
                    weights[i][j] = (Math.random() * 2 / Math.sqrt(nodesIn)) - 1 / Math.sqrt(nodesIn);
                }
            }
        }
    }
}
