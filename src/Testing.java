import java.util.ArrayList;
import java.util.Arrays;

public class Testing {
    public static void main(String[] args) {
        System.out.println("Untrained: ");
        NeuralNetwork nn = new NeuralNetwork(new int[]{28*28, 100, 10});

        double[][] testImages = IDXInterpreter.imageRead("./src/MNIST-dataset/test-images.idx");
        int[] testLabels = IDXInterpreter.labelRead("./src/MNIST-dataset/test-labels.idx");
        if (testImages != null && testLabels != null) {
            DataPoint[] testDs = toDataPoints(testImages, testLabels);

            System.out.println(nn.test(testDs, 100));
        }

        System.out.println("Trained: ");
        test("nn-info", 10);

        //train("nn-info");
    }
    public static void test(String fileName, int times) {
        NeuralNetwork nn = NeuralNetwork.readInfo(fileName);

        double[][] testImages = IDXInterpreter.imageRead("./src/MNIST-dataset/test-images.idx");
        int[] testLabels = IDXInterpreter.labelRead("./src/MNIST-dataset/test-labels.idx");
        if (testImages != null && testLabels != null) {
            DataPoint[] testDs = toDataPoints(testImages, testLabels);

            System.out.println(nn.test(testDs, 100));
        }
    }

    public static void train(String fileName) {
        NeuralNetwork nn = new NeuralNetwork(new int[]{28*28, 100, 10});

        double[][] trainImages = IDXInterpreter.imageRead("./src/MNIST-dataset/train-images.idx");
        int[] trainLabels = IDXInterpreter.labelRead("./src/MNIST-dataset/train-labels.idx");

        if (trainImages != null && trainLabels != null) {
            DataPoint[] trainDs = toDataPoints(trainImages, trainLabels);

            nn.train(trainDs, 100, 1);
            nn.writeInfo(fileName);
        }
    }

    public static DataPoint[] toDataPoints(double[][] images, int[] labels) {
        DataPoint[] data = new DataPoint[images.length];
        for (int i = 0; i < labels.length; i++) {
            double[] outs = new double[10];
            outs[labels[i]] = 1;
            data[i] = new DataPoint(images[i], outs);
        }
        return data;
    }
}
