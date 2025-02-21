import Function.LossFunction;
import Matrices.ActivationMatrix;

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

    public ActivationMatrix feedForward(ActivationMatrix input){
        // TODO Custom Annotation pour forcer l'usage de la bonne taille à la compilation (et pas juste au runtime)
        assert(input.getNumberOfRows() == dimInput) : "Erreur : dim d'entrée attendue = " + dimInput + " , obtenue : " + input.getNumberOfRows() + ".";

        // Pour chaque couche, on calcule un nouveau vecteur d'activations à partir du précédent
        // Et on l'envoie à la prochaine couche.
        layers.forEach(layerMatrix -> layerMatrix.computeNewActivationMatrix(input));
        return input;
    }


    /**
     * Calcule le résultat d'une du réseau de neurones sur une certaine entrée {@link ActivationMatrix}.
     * @param input Le {@link ActivationMatrix} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction){
        ActivationMatrix networkOutput = feedForward(input);
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

    public void backPropagate(ActivationMatrix input, ActivationMatrix vector) {

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
