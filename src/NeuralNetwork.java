import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork {
    Layer[] layers;
    Cost costFunc; //default cost function
    Activation outputAct;
    double learnRate;

    public NeuralNetwork(int[] layerSizes, int activation, int outActivation, int cost, double learnRate) {
        layers = new Layer[layerSizes.length-1]; //output "layer" is really just the outputs from the last layer
        costFunc = new Cost(cost);
        outputAct = new Activation(outActivation);
        this.learnRate = learnRate;
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
            layers[i].activation = new Activation(activation);
        }
    }
    public NeuralNetwork(int[] layerSizes) {
        layers = new Layer[layerSizes.length-1]; //output "layer" is really just the outputs from the last layer
        costFunc = new Cost(Cost.QUADRATIC);
        outputAct = new Activation(Activation.LINEAR);
        this.learnRate = 1;
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
            layers[i].activation = new Activation(Activation.SIGMOID);
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

    public void updateGradients(DataPoint d) { //backpropagation!!!!! :DDDD
        getOutputs(d.inputs); //stores the inputs and pre-activation outputs in each layer
        //to get the post-activation outputs just look at next layer - to get the pre-activation inputs look at previous layer
        //obviously this is different for the output layer, which is fine since output layer could have a different activation function anyways

        //output layer deriv constants are a bit different from hidden layer
        Layer outLayer = layers[layers.length-1];
        double[] derivConstants = new double[outLayer.nodesOut]; //length of the output layer
        //basically partial derivatives of activation to weighted output times cost to activation
        //since these are the same for any weight leading to the same node it's more efficient to calculate them first
        for (int i = 0; i < derivConstants.length; i++) { //basically each value of the output layer
            double a = outLayer.preActivationOutputs[i];
            derivConstants[i] = outputAct.getDerivative(a) * costFunc.costDerivative(outputAct.get(a), d.expectedOuts[i]);
        }
        outLayer.updateGradients(derivConstants);

        //other layers
        for (int i = layers.length - 2; i >= 0; i--) {
            derivConstants = layers[i].getDerivConstants(layers[i+1], derivConstants);
            layers[i].updateGradients(derivConstants);
        }
    }
    public void learn(DataPoint[] ds) { //TODO: use multithreading to increase speed
        for (DataPoint d : ds) {
            updateGradients(d);
        }

        for (Layer l : layers) {
            l.addGradients(learnRate / ds.length); //division to make sure it's the average
            l.weightGradients = new double[l.nodesIn][l.nodesOut];
            l.biasGradients = new double[l.nodesOut];
        }
    }

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(new int[]{28*28, 100, 10});
        double[][] trainImages = IDXInterpreter.imageRead("./src/MNIST-dataset/train-images.idx");
        int[] trainLabels = IDXInterpreter.labelRead("./src/MNIST-dataset/train-labels.idx");

        double[][] testImages = IDXInterpreter.imageRead("./src/MNIST-dataset/test-images.idx");
        int[] testLabels = IDXInterpreter.labelRead("./src/MNIST-dataset/test-labels.idx");

        System.out.println("starting if");
        if (trainImages != null && trainLabels != null && testImages != null && testLabels != null) {
            DataPoint[] trainDs = new DataPoint[trainLabels.length];
            DataPoint[] testDs = new DataPoint[testLabels.length];

            for (int i = 0; i < trainLabels.length; i++) {
                double[] outs = new double[10];
                outs[trainLabels[i]] = 1;
                trainDs[i] = new DataPoint(trainImages[i], outs);
            }
            for (int i = 0; i < testLabels.length; i++) {
                double[] outs = new double[10];
                outs[testLabels[i]] = 1;
                testDs[i] = new DataPoint(testImages[i], outs);
            }

            int minibatch = 100;
            for (int k = 0; k < 10; k++) {
                System.out.println("starting learning");
                ArrayList<DataPoint> trainAvailable = new ArrayList<>(Arrays.asList(trainDs));
                ArrayList<DataPoint> testAvailable = new ArrayList<>(Arrays.asList(testDs));

                for (int i = 0; i < trainImages.length; i++) {
                    DataPoint[] trainBatch = new DataPoint[minibatch];
                    DataPoint[] testBatch = new DataPoint[minibatch];
                    if (trainAvailable.size() < minibatch) {
                        trainAvailable = new ArrayList<>(Arrays.asList(trainDs));
                    }
                    if (testAvailable.size() < minibatch) {
                        testAvailable = new ArrayList<>(Arrays.asList(testDs));
                    }
                    for (int j = 0; j < minibatch; j++) {
                        trainBatch[j] = trainAvailable.remove((int)(Math.random()*trainAvailable.size()));
                        testBatch[j] = testAvailable.remove((int)(Math.random()*testAvailable.size()));
                    }

                    nn.learn(trainBatch);
                }
                nn.averageCost(testDs);
            }
        }
    }
}
