package MLP.Data;

import Matrices.ActivationMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Représente un ensemble ordonnée de données qui peut être chargée dans la RAM
 * via la méthode {@link #load};.
 * Le Dataset se charge de gérer l'import d'une donnée en dur dans la mémoire du programme
 */
public abstract class LabeledDataset<InputType, OutputType> {

    /**
     * Renvoie la taille de l'ensemble de données.
     * Par exemple pour du MNIST, le nombre d'images de l'ensemble (60 000).
     * @return
     */
    int size;
    int inputDimension;
    public int outputDimension;

    /**
     * Représente le chemin d'accès vers les données d'entrée du modèle
     */
    public String path;

    /**
     * Représente le chemin d'accès vers les labels du modèle
     */
    public String labelPath;

    public LabeledDataset(int size, int inputDimension, int outputDimension,
                          String path, String labelPath) {
        this.size = size;
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.path = path;
        this.labelPath = labelPath;
    }

    /**
     * Récupère de la data et la vectorize pour qu'elle puisse
     * être transformée en matrice d'activation par la suite.
     * @return
     */
    public abstract double[] vectorizeInput(InputType input);

    public abstract double[] vectorizeOutput(OutputType input);


    /**
     * Charge en mémoire et retourne le i-ème item labélisé de l'ensemble d'entraînement.
     * @return
     */
    public abstract LabeledDataSample<InputType, OutputType> load(int i);

    /**
     * Charge en mémoire et récupère sous forme de liste
     * l'ensemble des données du set de a à b
     * @return
     */
    public List<LabeledDataSample<InputType, OutputType>> loadList(int a, int b) {
        List<LabeledDataSample<InputType, OutputType>> res = new ArrayList<>();
        for(int i = a; i < b; i++) {
            res.add(load(i));
        }

        return res;
    }

    public LabeledBatch DataSampleToDataPair(LabeledDataSample<InputType, OutputType> dataSample) {
        return new LabeledBatch(this.vectorizeInput(dataSample.getInput()),
                vectorizeOutput(dataSample.getOutput()));
    }

    // TODO Trouver des meilleurs noms prck la terminologie est pas ouf

    /**
     * Représente un échantillon labélisé du Dataset.
     * Il contient la paire (entrée, sortie) de l'ensemble.
     * Doit ensuite être transformée en {@link ActivationMatrix} via {@link #vectorizeInput}.
     *
     * pr permettre pour chaque classe de dataset de def de nvx objets entrée, sortie qui gèreront l'import etc
     */
    public static class LabeledDataSample<InputType,OutputType> {

        InputType input;
        OutputType output;

        public LabeledDataSample(InputType input, OutputType output) {
            this.input = input;
            this.output = output;
        }

        public void setInput(InputType input) {
            this.input = input;
        }

        public void setOutput(OutputType output) {
            this.output = output;
        }

        public InputType getInput() {
            return input;
        }

        public OutputType getOutput() {
            return output;
        }

    }

    /**
     * Représente un couple entrée-sortie du réseau de neurones,
     * sous forme de paire ({@link ActivationMatrix},{@link ActivationMatrix}).
     */
    public static class LabeledBatch {

        ActivationMatrix input;
        ActivationMatrix output;

        public LabeledBatch(ActivationMatrix input, ActivationMatrix output) {
            this.input = input;
            this.output = output;
        }

        public LabeledBatch(double[] input, double[] output) {
            ActivationMatrix inputMatrix = new ActivationMatrix(input);
            ActivationMatrix outputMatrix = new ActivationMatrix(output);
            this(inputMatrix, outputMatrix);
        }

        public ActivationMatrix getInput() {
            return input;
        }

        public ActivationMatrix getOutput() {
            return output;
        }
    }

}
