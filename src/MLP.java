import Function.LossFunction;
import Matrices.ActivationVector;

import java.util.List;

public class MLP {

    private final int dimInput;
    private final List<Layer> layers;

    public MLP(List<Layer> layers, int dimInput){
        this.layers = layers;
        this.dimInput = dimInput;
    }

    // TODO TODO implémenter plutôt le calcul en multiplications
    // TODO TODO matricielles plutôt qu'en séquentiel

    /**
     * Envoie au réseau de neurones un vecteur d'activations initial (la donnée d'entrée mise sous forme de vecteur
     * aux dimensions de la première couche du réseau) et renvoie le vecteur d'activations de la dernière couche du réseau
     * (la réponse du réseau de neurones).
     * @param input Le vecteur d'activation initial
     * @return
     */

    public ActivationVector feedForward(ActivationVector input){
        // TODO Custom Annotation pour forcer l'usage de la bonne taille à la compilation (et pas juste au runtime)
        assert(input.size() == dimInput) : "Erreur : dim d'entrée attendue = " + dimInput + " , obtenue : " + input.size() + ".";
        ActivationVector activationsOfPreviousLayer = input;
        // Pour chaque couche, on calcule un nouveau vecteur d'activations à partir du précédent
        // Et on l'envoie à la prochaine couche.
        int i = 0;

        for(Layer layer : layers) {

            System.out.println("Activation de la couche n°" + i + " : " + activationsOfPreviousLayer);
            activationsOfPreviousLayer = layer.computeActivationVectorOfPreviousLayer(activationsOfPreviousLayer);
            i++;
        }
        return activationsOfPreviousLayer;
    }


    /**
     * Calcule le résultat d'une du réseau de neurones sur une certaine entrée {@link ActivationVector}.
     * @param input Le {@link ActivationVector} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationVector input, ActivationVector expectedOutput, LossFunction lossFunction){
        ActivationVector networkOutput = feedForward(input);
        System.out.println("Network output : " + networkOutput);
        return lossFunction.apply(networkOutput, expectedOutput);
    }

    /**
     * Calcule la fonction coût {@link LossFunction#MSE} du réseau de neurones sur une certaine entrée {@link ActivationVector}.
     * @param input Le {@link ActivationVector} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationVector input, ActivationVector expectedOutput) {
        return computeLoss(input, expectedOutput, LossFunction.MSE);
    }

    public void backPropagate(ActivationVector input, ActivationVector vector) {

    }

    public static MLPBuilder builder(int dimInput){
        return new MLPBuilder(dimInput);
    }




    public void print(){

        int n = layers.size();
        System.out.println("Dimension d'entrée : " + this.dimInput);
        System.out.println();
        for(int i = 0; i < n; i ++){
            System.out.println("Layer n°" + i + " : ");
            layers.get(i).print();
            System.out.println();
        }
    }


}
