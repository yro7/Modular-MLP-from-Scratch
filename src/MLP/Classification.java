package MLP;

/**
 * Représente la sortie d'un réseau de neurones de type Classifier (ex MNIST).
 * Une classification de n classes est définie comme un tableau de taille n
 * dont tous les élémentes sont à 0, sauf la classe attendue qui est à 1.
 */
public class Classification {

    double[] classes;

    public Classification(int outputDimension, int labelNumber) {
        this.classes = new double[outputDimension];
        this.classes[labelNumber] = 1.0;
    }


    public double[] getHotEncoding() {
        return this.classes;
    }
}
