import java.util.ArrayList;
import java.util.Arrays;

public class Testing {
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
                System.out.println("epoch " + k);
                ArrayList<DataPoint> trainAvailable = new ArrayList<>(Arrays.asList(trainDs));

                for (int i = 0; i < trainDs.length/minibatch; i++) {
                    DataPoint[] trainBatch = new DataPoint[minibatch];
                    if (trainAvailable.size() < minibatch) {
                        trainAvailable = new ArrayList<>(Arrays.asList(trainDs));
                    }
                    for (int j = 0; j < minibatch; j++) {
                        trainBatch[j] = trainAvailable.remove((int)(Math.random()*trainAvailable.size()));
                    }

                    nn.learn(trainBatch);
                }
                System.out.println("cost: " + nn.averageCost(testDs));
            }
        }
    }
}
