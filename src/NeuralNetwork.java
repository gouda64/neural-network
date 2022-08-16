import java.util.*;

public class NeuralNetwork {
    ArrayList<Layer> layers;
    int activationType;

    public NeuralNetwork(ArrayList<Layer> layers, int a) {
        this.layers = layers;
        activationType = a;
        for (Layer l : this.layers) {
            l.activationType = activationType;
        }
    }
}
