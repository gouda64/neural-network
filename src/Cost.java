public class Cost {
    static final int QUADRATIC = 1;
    int funcType;

    public Cost(int type) {
        funcType = type;
    }

    public double cost(double get, double expected) {
        //TODO: add different cost functions
        // (https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications)
        if (funcType == QUADRATIC) {
            return (get-expected)*(get-expected);
        }
        System.out.println("cost function error");
        return 0;
    }
}
