import java.util.List;
import java.util.Vector;

public class MLP {

    private final int dimInput;
    private final List<Layer> layers;

    public MLP(List<Layer> layers, int dimInput){
        this.layers = layers;
        this.dimInput = dimInput;
    }

    /**
     * Envoie au réseau de neurones un vecteur d'activations initial (la donnée d'entrée mise sous forme de vecteur
     * aux dimensions de la première couche du réseau) et renvoie le vecteur d'activations de la dernière couche du réseau
     * (la réponse du réseau de neurones).
     * @param input Le vecteur d'activation initial
     * @return
     */

    public ActivationVector feedForward(ActivationVector input){
        // TODO Custom Annotation pour forcer l'usage de la bonne taille à la compilation (et pas juste runtime)
        assert(input.size() == dimInput);
        ActivationVector activationsOfPreviousLayer = input;
        // Pour chaque couche, on calcule un nouveau vecteur d'activations à partir du précédent
        // Et on l'envoie à la prochaine couche.
        for(Layer layer : layers) {
            activationsOfPreviousLayer = layer.computeActivationVectorOfPreviousLayer(activationsOfPreviousLayer);
        }
        return activationsOfPreviousLayer;
    }

    /**
     * Calcule la fonction coût du réseau de neurone sur une certaine entrée.
     * @param input Le {@link ActivationVector} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationVector input){

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
