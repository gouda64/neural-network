public class DataPoint {
    double[] inputs;
    double[] expectedOuts; //normally only one answer

    public DataPoint(double[] inputs, double[] outputs) {
        this.inputs = inputs;
        expectedOuts = outputs;
    }
}
