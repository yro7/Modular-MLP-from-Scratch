public class Neuron {

    private double weight;
    private double bias;
    private final ActivationFunction activationFunction;

    // Permet de stocker l'activation d'un neuronne
    // après qu'il ait été nourri
    private double activation;

    public Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public Neuron(ActivationFunction activationFunction, int numberOfInputs, int layerSize) {
        this.activationFunction = activationFunction;
        initialize(numberOfInputs, layerSize);
    }

    public double getWeight() {
        return weight;
    }

    public ActivationFunction getActivationFunction(){
        return this.activationFunction;
    }

    public double getBias() {
        return bias;
    }

    public double getActivation() {
        return activation;
    }

    public void setActivation(double activation) {
        this.activation = activation;
    }

    public double feed(double input) {
        double res = activationFunction.apply(input+getBias());
        setActivation(res);
        return res;
    }

    public double computeActivationFromLayer(Layer layer){
        double input = layer.getNeurons()
                .stream()
                .mapToDouble(n -> n.getActivation()*n.getWeight())
                .sum();
        return feed(input);

    }

    public void initialize(int numberOfInputs, int layerSize){
        double bias = getRandomBias(numberOfInputs, layerSize);
        double weight = getRandomWeight(numberOfInputs, layerSize);
        this.bias = bias;
        this.weight = weight;
    }


    public double getRandomBias(int numberOfInputs, int layerSize){
        return this.getActivationFunction().initializationFunction.getRandomBias.apply(numberOfInputs, layerSize);
    }

    public double getRandomWeight(int numberOfInputs, int layerSize){
        return this.getActivationFunction().initializationFunction.getRandomWeight.apply(numberOfInputs, layerSize);
    }

}
