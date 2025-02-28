import Function.ActivationFunction;
import Function.LossFunction;
import Matrices.ActivationMatrix;
import Matrices.GradientMatrix;
import Matrices.WeightMatrix;

import java.util.ArrayList;
import java.util.List;

public class MLP {

    private final int dimInput;
    private final List<Layer> layers;

    public MLP(List<Layer> layers, int dimInput){
        this.layers = layers;
        this.dimInput = dimInput;
    }
    /**
     * Envoie au réseau de neurones une batch ({@link ActivationMatrix}
     * et renvoie les matrices d'activations des couches du réseau après le calcul.
     * La fonction renvoie une paire ou le premier élément de la paire est la matrice des activations
     * après application de la fonction d'activation, le second avant. (utile pour {@link #gradientDescent}.
     * @param input Le vecteur d'activation initial
     * @return Une liste de paires matrices d'activation qui correspond aux activations de chaque couche
     * dans l'ordre.
     */

    public List<Pair<ActivationMatrix,ActivationMatrix>> feedForward(ActivationMatrix input){
        assert(input.getNumberOfRows() == dimInput) : "Erreur : dim d'entrée attendue = " + dimInput + " , obtenue : " + input.getNumberOfRows() + ".";

        List<Pair<ActivationMatrix,ActivationMatrix>> res = new ArrayList<>();
        ActivationMatrix preFunctionActivationMatrix = input.clone();
        ActivationMatrix newActivationMatrix = input.clone();

        for(Layer layer : layers){

            // Calcul de W*A + B
            preFunctionActivationMatrix = layer.computePreFunctionNewActivationMatrix(newActivationMatrix);
            // Calcul de f(WxA + B)
            newActivationMatrix = preFunctionActivationMatrix.applyFunction(layer.getActivationFunction());
            res.add(new Pair<>(newActivationMatrix, preFunctionActivationMatrix));

        }

        return res;
    }


    /**
     * Calcule le résultat d'une du réseau de neurones sur une certaine entrée {@link ActivationMatrix}.
     * @param input Le {@link ActivationMatrix} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction){
        ActivationMatrix networkOutput = feedForward(input).getLast().getA();
        System.out.println("Network output : " + networkOutput);
        return lossFunction.apply(networkOutput, expectedOutput);
    }

    /**
     * Calcule la fonction coût {@link LossFunction#MSE} du réseau de neurones sur une certaine entrée {@link ActivationMatrix}.
     * @param input Le {@link ActivationMatrix} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationMatrix input, ActivationMatrix expectedOutput) {
        return computeLoss(input, expectedOutput, LossFunction.MSE);
    }

    public List<GradientMatrix> gradientDescent(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction) {
        List<GradientMatrix> res = new ArrayList<>();

        List<Pair<ActivationMatrix,ActivationMatrix>> activations = this.feedForward(input);
        Pair<ActivationMatrix,ActivationMatrix> outputPair = activations.getLast();
        ActivationMatrix output = outputPair.getA();
        ActivationMatrix outputPreAF = outputPair.getB();

        ActivationMatrix derivativeOnActivation = outputPreAF.applyFunction(getLastLayer().getActivationFunction().derivativeFunction);

        GradientMatrix firstGradient = lossFunction.derivative.apply(output,expectedOutput)
                .hadamardProduct(derivativeOnActivation);

        res.add(firstGradient);

        GradientMatrix currentGradient = firstGradient;

        for(int i = layers.size() - 1; i >= 0; i--){
            Layer layer = this.layers.get(i);
            WeightMatrix weightOfLayer = layer.getWeightMatrix();
            ActivationMatrix preActivation = activations.get(i).getB();

            ActivationMatrix derivativeActivation = preActivation.applyFunction(layer.getActivationFunction().derivativeFunction);
            currentGradient = currentGradient.multiply(weightOfLayer.transpose())
                    .hadamardProduct(derivativeActivation);
            res.add(currentGradient);
        }

        return res;
    }

    // TODO
    public void backPropagate(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction) {
        List<GradientMatrix> gradients = this.gradientDescent(input, expectedOutput, lossFunction);
    }

    public static MLPBuilder builder( int dimInput){
        return new MLPBuilder(dimInput);
    }




    public void print(){

        int n = layers.size();
        System.out.println();
        System.out.println("Dimension d'entrée : " + this.dimInput);
        System.out.println();
        for(int i = 0; i < n; i ++){
            System.out.println("Layer n°" + i + " : ");
            this.layers.get(i).print();;
            System.out.println();
        }
    }

    public Layer getLastLayer(){
        return this.layers.getLast();
    }


}
