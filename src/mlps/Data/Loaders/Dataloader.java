package mlps.Data.Loaders;

import matrices.ActivationMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Représente extracteur de données qui permet de charger dans la RAM
 * de la donnée labélisée stockée en dur via la méthode {@link #load};.
 */
public abstract class Dataloader<FeaturesType, LabelsType> {

    /**
     * Renvoie la taille de l'ensemble de données.
     * Par exemple pour du MNIST, le nombre d'images de l'ensemble (60 000).
     * @return
     */
    public int size;
    public int inputDimension;
    public int outputDimension;

    /**
     * Représente le chemin d'accès vers les données d'entrée du modèle
     */
    public String featuresPath;

    /**
     * Représente le chemin d'accès vers les labels du modèle
     */
    public String labelPath;

    public int batchSize;

    public Dataloader(int size, int inputDimension, int outputDimension,
                      String featuresPath, String labelPath, int batchSize) {
        this.size = size;
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.featuresPath = featuresPath;
        this.labelPath = labelPath;
        this.batchSize = batchSize;
    }


    /**
     * Récupère de la data et la vectorize pour qu'elle puisse
     * être transformée en matrice d'activation par la suite.
     * @return
     */
    public abstract double[] vectorizeInput(FeaturesType input);

    public abstract double[] vectorizeOutput(LabelsType input);


    /**
     * Charge en mémoire et retourne le i-ème item labélisé de l'ensemble de données.
     * @return
     */
    public abstract LabeledDataSample<FeaturesType, LabelsType> load(int i);


    /**
     * Charge en mémoire et récupère sous forme de liste
     * l'ensemble des données du set de a à b
     * @return
     */
    public List<LabeledDataSample<FeaturesType, LabelsType>> loadList(int a, int b) {
        List<LabeledDataSample<FeaturesType, LabelsType>> res = new ArrayList<>();
        for(int i = a; i < b; i++) {
            res.add(load(i));
        }

        return res;
    }

    public LabeledBatch DataSampleToDataPair(LabeledDataSample<FeaturesType, LabelsType> dataSample) {
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
            this.input = new ActivationMatrix(input);
            this.output = new ActivationMatrix(output);
        }

        public ActivationMatrix getInput() {
            return input;
        }

        public ActivationMatrix getOutput() {
            return output;
        }
    }


    /**
     * Récupère un tableau d'entrées et le met sous forme de batch.
     * Chaque entrée sera vectorisée et formera une ligne du batch résultant.
     * @param objects
     * @return
     */
    public ActivationMatrix objects_To_Batch(List<FeaturesType> objects) {
        double[] firstItem = this.vectorizeInput(objects.getFirst());
        int itemSizes = firstItem.length;
        int batchSize = objects.size();
        double[][] data = new double[batchSize][itemSizes];
        for(int i = 0; i < batchSize; i++) {
            data[i] = this.vectorizeInput(objects.get(i));
        }

        return new ActivationMatrix(data);
    }

    /**
     * Charge en mémoire et renvoie la batch numéro i du {@link Dataloader},
     * sous forme de couple (Entrée, Sortie Attendue)
     * @param batchNumber
     * @return
     */
    public LabeledBatch getBatch(int batchNumber) {
        assert(batchNumber < this.getNumberOfBatches()) : "Le numéro de la batch doit être inférieur au nombre de batch du dataset !";

        int begin = batchNumber * batchSize;
        int end = (batchNumber + 1) * batchSize;

        List<LabeledDataSample<FeaturesType, LabelsType>> objects = loadList(begin, end);
        double[] firstItem = this.vectorizeInput(objects.getFirst().getInput());

        ActivationMatrix batch = new ActivationMatrix(batchSize, firstItem.length);
        ActivationMatrix labels = new ActivationMatrix(batchSize, outputDimension);

        for(int i = 0; i < batchSize; i++) {
            LabeledDataSample<FeaturesType, LabelsType> object = objects.get(i);

            batch.getData()[i] = this.vectorizeInput(object.getInput());
            labels.getData()[i] = this.vectorizeOutput(object.getOutput());
        }

        return new LabeledBatch(batch, labels);
    }


    /**
     * Renvoie le nombre de batch dans l'ensemble d'entraînement,
     * c'est à dire la taille de l'ensemble divisé par la taille de chaque batch.
     * @return
     */
    public int getNumberOfBatches() {
        return this.size / batchSize;
    }




}
