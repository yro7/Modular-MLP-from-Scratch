package MLP;

/**
 * Représente la sortie d'un réseau de neurones de type Classifier (ex MNIST).
 * Une classification de n classes est définie comme un tableau de taille n
 * dont tous les élémentes sont à 0, sauf la classe attendue qui est à 1.
 */
public class Classification {

    int[] classes;

    public Classification(int outputDimension, int labelNumber) {
        this.classes = new int[outputDimension];
        this.classes[labelNumber] = 1;
    }


}
