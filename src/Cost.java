public class Cost {
    static final int QUADRATIC = 0;
    //TODO: add different cost functions
    // (https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications)
    int funcType;

    public Cost(int type) {
        funcType = type;
    }

    public double cost(double get, double expected) {
        switch(funcType) {
            case QUADRATIC:
                return (get-expected) * (get-expected);
            default:
                System.out.println("cost function error");
                return 0;
        }
    }
    public double costDerivative(double get, double expected) {
        //same deal as above
        switch(funcType) {
            case QUADRATIC:
                return 2 * (get-expected);
            default:
                System.out.println("cost function derivative error");
                return 0;
        }
    }
}
