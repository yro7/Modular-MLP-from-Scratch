package MLP;

import Function.LossFunction;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.GradientMatrix;
import Matrices.WeightMatrix;

import java.util.ArrayList;
import java.util.List;

import static Function.ActivationFunction.SoftMax;
import static Function.LossFunction.CE;

public class MLP {

    private final int dimInput;
    private final List<Layer> layers;

    public MLP(List<Layer> layers, int dimInput){
        this.layers = layers;
        this.dimInput = dimInput;
    }

    /**
     * Envoie au réseau de neurones une batch ({@link ActivationMatrix}.
     * et renvoie les matrices d'activations des couches du réseau après le calcul.
     * La fonction renvoie une paire ou le premier élément de la paire est la matrice des activations
     * après application de la fonction d'activation, le second avant. (utile pour {@link #backpropagate}.
     * @param input Le vecteur d'activation initial
     * @return Une liste de paires matrices d'activation qui correspond aux activations de chaque couche
     * dans l'ordre.
     */

    public List<Pair<ActivationMatrix,ActivationMatrix>> feedForward(ActivationMatrix input){
        assert(input.getNumberOfColumns() == dimInput) : "Erreur : dim d'entrée attendue = " + dimInput + " , obtenue : " + input.getNumberOfColumns() + ".";

        List<Pair<ActivationMatrix,ActivationMatrix>> result = new ArrayList<>();
        ActivationMatrix activationMatrixOfLayer = input;


        for(Layer layer : layers){
            // Calcul de    AxW + B
            ActivationMatrix activationsBeforeAF = layer.multiplyByWeightsAndAddBias(activationMatrixOfLayer);
            // Calcul de  f(AxW + B)
            activationMatrixOfLayer = layer.getActivationFunction()
                    .apply(activationsBeforeAF.clone()); // Clone nécessaire pour éviter de modifier
                                                         // activationsBeforeAF

            result.add(new Pair<>(activationMatrixOfLayer, activationsBeforeAF));
        }


        return result;
    }


    /**
     * Evalue la prédiction du réseau de neurones sur une certaine entrée {@link ActivationMatrix}.
     * @param input Le {@link ActivationMatrix} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction){
        ActivationMatrix networkOutput = feedForward(input).getLast().getA();
        return lossFunction.apply(networkOutput, expectedOutput);
    }

    public void updateParameters(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction){
        assert lossFunction != CE || (this.getLastLayer().getActivationFunction() == SoftMax) : "La couche de sortie du réseau " +
                "devrait être Softmax si la fonction de coût utilisée est la Cross Entropie !";

        List<Pair<GradientMatrix,BiasVector>> gradients = backpropagate(input, expectedOutput, lossFunction);

        // TODO IMPLEMENT OPTIMIZERS
        double learningRate = 0.0001;

        for(int l = 0; l < this.layers.size(); l++){
            GradientMatrix weightCorrection = gradients.get(l).getA().clone().multiply(learningRate);
            BiasVector biasGradient = gradients.get(l).getB().clone().multiply(learningRate);

            Layer currentLayer = this.getLayer(l);

            currentLayer.getWeightMatrix().substract(weightCorrection);
            currentLayer.getBiasVector().substract(biasGradient);
        }


    }

    /**
     *
     * @param input
     * @param expectedOutput
     * @param lossFunction
     * @return
     */
    public List<Pair<GradientMatrix,BiasVector>> backpropagate(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction) {
        List<Pair<GradientMatrix,BiasVector>> gradients = new ArrayList<>();
        List<Pair<ActivationMatrix, ActivationMatrix>> activations = this.feedForward(input);
        int L = layers.size();
        ActivationMatrix a_L = activations.getLast().getA(); // Activations de la dernière couche APRES fonction d'activation
        ActivationMatrix z_L = activations.getLast().getB(); // Activations de la dernière couche AVANT fonction d'activation

        // Vérification des dimensions de sortie
        assert(expectedOutput.hasSameDimensions(a_L)) : String.format(
                "Dimensions de sortie incorrectes: attendu %dx%d, obtenu %dx%d",
                a_L.getNumberOfRows(), a_L.getNumberOfColumns(),
                expectedOutput.getNumberOfRows(), expectedOutput.getNumberOfColumns());


        GradientMatrix delta_L = this.computeOutputGradient(lossFunction, expectedOutput, a_L, z_L);

        // Calcul des gradients pour la couche de sortie
        ActivationMatrix a_L_minus_1 = (L-1 > 0) ? activations.get(L-2).getA() : input;
        GradientMatrix dL_dW_L = delta_L.multiplyAtRight(a_L_minus_1.transpose());

        BiasVector dL_db_L = delta_L.sumErrorTerm();
        gradients.addFirst(new Pair<>(dL_dW_L, dL_db_L));

        // Rétropropagation à travers les couches cachées
        GradientMatrix delta_l_plus_1 = delta_L;

        // Rétropropagation à travers les couches cachées

        for (int l = L-2; l >= 0; l--) {

            ActivationMatrix z_l = activations.get(l).getB(); // Activations Pre-AF
            ActivationMatrix a_l_minus_1 = (l > 0) ? activations.get(l-1).getA() : input; // Activations Post-AF

            Layer layer_l_plus_1 = this.layers.get(l+1);
            Layer layer_l = this.layers.get(l);

            WeightMatrix W_l_plus_1_T = layer_l_plus_1.getWeightMatrix().transpose();
            ActivationMatrix sigma_prime_z_l = layer_l.getDerivativeOfAF().apply(z_l);
            GradientMatrix delta_l = delta_l_plus_1
                    .multiply(W_l_plus_1_T)
                    .hadamardProduct(sigma_prime_z_l);

            // Finalement : calcul du gradient des poids et du biais
            GradientMatrix dl_dW_l = delta_l.multiplyAtRight(a_l_minus_1.transpose());
            BiasVector dL_db_l = delta_l.sumErrorTerm();

            gradients.addFirst(new Pair<>(dl_dW_l, dL_db_l));
            delta_l_plus_1 = delta_l;

        }

        return gradients;

    }

    private GradientMatrix computeOutputGradient(LossFunction lossFunction, ActivationMatrix expectedOutput,
                                                 ActivationMatrix a_L, ActivationMatrix z_L) {

        GradientMatrix delta_L;

        // Cas spécial pour Softmax + Entropie croisée
                              // TODO factorizer cette immondice
        if (getLastLayer().getActivationFunction() == SoftMax && lossFunction instanceof CE) {
            // Pour Softmax avec CE, delta_L est simplement (a_L - expected)
            delta_L = a_L.substract(expectedOutput).toGradientMatrix();
        } else {
            // Calcul standard pour les autres AF:

            // Calcul de dL/da_L [sortie attendue]
            GradientMatrix dL_da_L = lossFunction.applyDerivative(a_L, expectedOutput);
            // Calcul de σ'(z_L)
            ActivationMatrix sigma_prime_z_L = getLastLayer().getDerivativeOfAF().apply(z_L);
            //  dL/da_L ⊙ σ'(z_L)
            delta_L = dL_da_L.hadamardProduct(sigma_prime_z_L);
        }

        return delta_L;
    }

    public static MLPBuilder builder(int dimInput){
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

    public void printDimensions(){
        for(int i = 0; i < layers.size(); i++){
            System.out.println("Layer " + i + " of size : " + this.getLayer(i).size());
            this.getLayer(i).getWeightMatrix().printDimensions(String.valueOf(i));
            this.getLayer(i).getBiasVector().printDimensions(String.valueOf(i));
            System.out.println();
            System.out.println();
        }
    }

    public Layer getLastLayer(){
        return this.layers.getLast();
    }

    public Layer getLayer(int i){
        return this.layers.get(i);
    }


    public void printNorms() {
        for(int i = 0; i < layers.size(); i++){
            System.out.println("Layer " + i + " of size : " + this.getLayer(i).size());
            System.out.println(this.getLayer(i).getWeightMatrix().norm());
            System.out.println(this.getLayer(i).getBiasVector().norm());
            System.out.println();
            System.out.println();
        }
    }

    public List<Layer> getLayers(){
        return this.layers;
    }

    public int getDimInput() {
        return this.dimInput;
    }
}
