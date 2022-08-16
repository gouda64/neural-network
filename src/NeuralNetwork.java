import java.util.*;

public class NeuralNetwork {
    Layer[] layers;
    Cost costFunc; //default cost function

    public NeuralNetwork(int[] layerSizes, int activation, int cost) {
        layers = new Layer[layerSizes.length-1]; //output "layer" is really just the outputs from the last layer
        costFunc = new Cost(cost);
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
            layers[i].act = new Activation(activation);
        }
    }

    public double[] getOutputs(double[] inputs) {
        for (Layer l : layers) {
            inputs = l.getOutputs(inputs);
        }
        return inputs;
    }

    public int maxOutputIndex(double[] inputs) {
        double[] outputs = getOutputs(inputs);
        int index = 0;
        for (int i = 1; i < outputs.length; i++) {
            if (inputs[i] > inputs[index]) {
                index = i;
            }
        }
        return index;
    }

    public double cost(DataPoint d) {
        double[] outputs = getOutputs(d.inputs);
        double c = 0;

        for (int i = 0; i < outputs.length; i++) {
            c += costFunc.cost(outputs[i], d.expectedOuts[i]);
        }
        return c;
    }

    public double averageCost(DataPoint[] ds) {
        double total = 0;
        for (DataPoint d : ds) {
            total += cost(d);
        }
        return total/ds.length;
    }
}
