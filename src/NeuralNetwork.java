import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork implements Serializable {
    Layer[] layers;
    Cost costFunc; //default cost function
    Activation outputAct;
    double learnRate;
    private String infoFile;

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
    public void learn(DataPoint[] ds) {
        for (DataPoint d : ds) {
            new GradientThread(d).start();
        }

        for (Layer l : layers) {
            l.addGradients(learnRate / ds.length); //division to make sure it's the average
            l.weightGradients = new double[l.nodesIn][l.nodesOut];
            l.biasGradients = new double[l.nodesOut];
        }
    }

    public void train(DataPoint[] trainDs, int batch, int epochs) {
        for (int k = 0; k < epochs; k++) {
            ArrayList<DataPoint> trainAvailable = new ArrayList<>(Arrays.asList(trainDs));

            //epoch
            for (int i = 0; i < trainDs.length/batch; i++) {
                DataPoint[] trainBatch = new DataPoint[batch];
                if (trainAvailable.size() < batch) {
                    trainAvailable = new ArrayList<>(Arrays.asList(trainDs));
                }
                for (int j = 0; j < batch; j++) {
                    trainBatch[j] = trainAvailable.remove((int)(Math.random()*trainAvailable.size()));
                }

                learn(trainBatch);
            }
        }
    }

    public double test(DataPoint[] testDs, int batch) {
        ArrayList<DataPoint> testAvailable = new ArrayList<>(Arrays.asList(testDs));
        DataPoint[] testBatch = new DataPoint[batch];
        for (int j = 0; j < batch; j++) {
            testBatch[j] = testAvailable.remove((int)(Math.random()*testAvailable.size()));
        }

        return averageCost(testBatch);
    }

    public void writeInfo() {
        if (infoFile == null) {
            System.out.println("no info file available"); //or use exception, it's just less convenient
            return;
        }
        try (FileOutputStream fos = new FileOutputStream(infoFile);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(this);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void writeInfo(String fileName) {
        infoFile = fileName;
        try (FileOutputStream fos = new FileOutputStream(infoFile);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(this);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static NeuralNetwork readInfo(String fileName) {
        try (FileInputStream fis = new FileInputStream(fileName);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            NeuralNetwork from = (NeuralNetwork) ois.readObject();
            from.infoFile = fileName;
            return from;
        }
        catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public class GradientThread extends Thread {
        DataPoint d;
        public GradientThread(DataPoint d) {
            this.d = d;
        }
        @Override
        public void run() {
            updateGradients(d);
        }
    }
}
