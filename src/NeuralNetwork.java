import java.util.*;

public class NeuralNetwork {
    Layer[] layers;

    public NeuralNetwork(int[] layerSizes, int activation) {
        layers = new Layer[layerSizes.length-1]; //output "layer" isn't a true layer
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
            layers[i].activationType = activation;
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
}
