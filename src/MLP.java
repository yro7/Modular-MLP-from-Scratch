import Function.ActivationFunction;
import Function.LossFunction;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.GradientMatrix;
import Matrices.WeightMatrix;

import javax.swing.text.Position;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MLP {

    public int DEBUG_LAYER = 0;
    public int DEBUG_PASSAGE = 0;

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

            System.out.println("Layer n° " + DEBUG_LAYER);
            System.out.println("Norme " + newActivationMatrix.norm());
            System.out.println("Norme des poids " + layer.getWeightMatrix().norm());
            DEBUG_LAYER++;
        }

        DEBUG_LAYER = 0;
        System.out.println();
        System.out.println();

        return res; // TODO FIX FEED FORWARD
    }


    /**
     * Calcule le résultat d'une du réseau de neurones sur une certaine entrée {@link ActivationMatrix}.
     * @param input Le {@link ActivationMatrix} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction){
        ActivationMatrix networkOutput = feedForward(input).getLast().getA();
        System.out.println("Passage " + DEBUG_PASSAGE + " : " + networkOutput.norm());
        DEBUG_PASSAGE++;
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

    public void backpropagate(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction){
        List<Pair<GradientMatrix,BiasVector>> gradients = gradientDescent(input, expectedOutput, lossFunction);

        // TODO IMPLEMENT OPTIMIZERS
        double learningRate = 0.000001;

        for(int l = 0; l < this.layers.size(); l++){
            GradientMatrix weightCorrection = gradients.get(l).getA().multiply(learningRate);
            BiasVector biasGradient = gradients.get(l).getB().multiply(learningRate);

            Layer currentLayer = this.getLayer(l);

            // TODO use setter with assert to verify dimensions
            currentLayer.weightMatrix = currentLayer.getWeightMatrix()
                    .substract(weightCorrection);
            currentLayer.biasVector = currentLayer.getBiasVector().
                    substract(biasGradient);


        }

    }

    public List<Pair<GradientMatrix,BiasVector>> gradientDescent(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction) {

        List<Pair<GradientMatrix,BiasVector>> result = new ArrayList<>();

        List<Pair<ActivationMatrix, ActivationMatrix>> layersActivations = this.feedForward(input);
        Pair<ActivationMatrix, ActivationMatrix> outputPair = layersActivations.getLast();

        ActivationMatrix a_L = outputPair.getA(); // Activations de la dernière couche APRES AF
        ActivationMatrix z_L = outputPair.getB(); // Activations de la dernière couche AVANT AF
        ActivationMatrix a_L_minus_1 = layersActivations.get(layersActivations.size()-2).getA();

        // On vérifie que la sortie théorique a les mêmes dimensions que la sortie réelle
        assert(expectedOutput.hasSameDimensions(a_L)) : "La matrice d'activation de sortie attendue doit être de même dimension \n" +
                "que la matrice d'activation de sortie du réseau. \n" +
                " Taille du batch attendue : " + input.getNumberOfColumns() +
                " Obtenue : " + expectedOutput.getNumberOfColumns() + "\n" +
                " Hauteur de la matrice d'activation attendue : " + a_L.getNumberOfRows() +
                " Obtenue : " + expectedOutput.getNumberOfRows() + "\n";


        Layer lastLayer = this.getLastLayer();

        // L est l'indice de la dernière couche
        // l est l'indice de la couche courante dans la boucle for


        // On calcul sigma'(z^L) où sigma est l'AF de la couche L
        // et z^L les activations pre-AF de la couche L (ici, la dernière)
        // sigma' de z^L où z^L = W^L * a^L-1 + B^L
        ActivationMatrix sigmaDerivative_z_L = z_L.applyFunction(lastLayer.getActivationFunction().derivativeFunction);
        // TODO cas particulier pour Soft Max + Cross Entropy
        GradientMatrix lossGradient_a_L = lossFunction.derivative.apply(a_L, expectedOutput);
        // Terme d'erreur de la couche de sortie
        GradientMatrix delta_L = lossGradient_a_L.hadamardProduct(sigmaDerivative_z_L);

        GradientMatrix weightGradient_L = delta_L.multiply(a_L_minus_1.transpose());
        BiasVector biasesGradient_L = delta_L.sumErrorTerm();
        result.add(new Pair<>(weightGradient_L, biasesGradient_L));

        GradientMatrix previousErrorTerm = delta_L;

        // - 1 car on ne traite pas la première,
        // et - 1 parce que les indices commencent à 0 (d'où -2)
        for (int l = this.layers.size() - 2; l >= 0; l--) {

            ActivationMatrix z_l = layersActivations.get(l).getB(); // Activations Pre-AF
            ActivationMatrix a_l_minus_1 = (l >= 1) ? layersActivations.get(l-1).getA() : input;

            Layer previousLayer = this.layers.get(l+1);
            Layer layer_l = this.layers.get(l);

            WeightMatrix weightsOfPreviousLayer = previousLayer.getWeightMatrix().transpose();
            ActivationMatrix sigmaDerivative_z_l = z_l.applyFunction(layer_l.getDerivativeOfAF());

            // Calcul du terme d'erreur
            GradientMatrix delta_l = previousErrorTerm
                    .multiplyAtRight(weightsOfPreviousLayer)
                    .hadamardProduct(sigmaDerivative_z_l);

            // Finalement : calcul du gradient des poids et du biais

            GradientMatrix weightGradient = delta_l.multiply(a_l_minus_1.transpose());
            BiasVector biasGradient = delta_l.sumErrorTerm();
            Pair<GradientMatrix, BiasVector> layerGradient = new Pair<>(weightGradient, biasGradient);
            result.add(layerGradient);

            previousErrorTerm = delta_l;
        }

        // On retourne la liste dans l'ordre des layers
        Collections.reverse(result);
        return result;
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
}
