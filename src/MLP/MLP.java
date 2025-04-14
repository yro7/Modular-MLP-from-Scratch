package MLP;

import Function.LossFunction;
import MLP.Optimizers.Adam;
import MLP.Optimizers.Optimizer;
import MLP.Regularizations.ParameterRegularization;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.GradientMatrix;
import Matrices.WeightMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static Function.ActivationFunction.SoftMax;
import static Function.LossFunction.CE;

public class MLP implements Serializable {

    private int dimInput;
    private List<Layer> layers;

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

    public FeedForwardResult feedForward(ActivationMatrix input){
        assert(input.getNumberOfColumns() == dimInput) : "Erreur : dim d'entrée attendue = " + dimInput + " , obtenue : " + input.getNumberOfColumns() + ".";

        FeedForwardResult result = new FeedForwardResult();
        ActivationMatrix activationMatrixOfLayer = input;


        for(Layer layer : layers){
            // Calcul de    AxW + B
            ActivationMatrix activationsBeforeAF = layer.multiplyByWeightsAndAddBias(activationMatrixOfLayer);
            // Calcul de  f(AxW + B)
            activationMatrixOfLayer = layer.getActivationFunction()
                    .apply(activationsBeforeAF.clone()); // Clone nécessaire pour éviter de modifier
                                                                     // activationsBeforeAF
            result.add(activationMatrixOfLayer, activationsBeforeAF);
        }


        return result;
    }


    /**
     * Evalue la prédiction du réseau de neurones sur une certaine entrée {@link ActivationMatrix}.
     * @param input Le {@link ActivationMatrix} dont on calcule le coût
     * @return le coût associé
     */
    public double computeLoss(ActivationMatrix input, ActivationMatrix expectedOutput, LossFunction lossFunction){
        ActivationMatrix networkOutput = feedForward(input).getNetworkOutput();
        return lossFunction.apply(networkOutput, expectedOutput);
    }


    /**
     * Entraîne le modèle en fonction du {@link Trainer} donné,
     * et renvoie le modèle entraîné.
     **/
    public MLP train(Trainer trainer){
        // On délègue simplement à trainer
        // pour éviter de trop bloat la classe MLP qui est déjà bien remplie
        trainer.train(this);
        return this;
    }

    /**
     * Mets à jour les poids & biais du réseau en fonction pour un couple (batch, sortie théorique),
     * en fonction de l'{@link Optimizer} choisi.
     * La méthode délègue simplement au Optimizer concerné la gestion de l'update.
     * @param input La matrice
     * @param expectedOutput
     * @param lossFunction
     * @param optimizer
     */
    public void updateParameters(ActivationMatrix input, ActivationMatrix expectedOutput,
                                 LossFunction lossFunction, Optimizer optimizer, ParameterRegularization regularization){

        optimizer.updateParametersBody(input, expectedOutput, lossFunction, this, regularization);

    }

    /**
     *
     * @param input
     * @param expectedOutput
     * @param lossFunction
     * @return
     */
    public BackProResult backpropagate(ActivationMatrix input, ActivationMatrix expectedOutput,
                                       LossFunction lossFunction, ParameterRegularization parameterRegularization) {
        BackProResult gradients = new BackProResult();
        FeedForwardResult activations = this.feedForward(input);
        int L = layers.size();
        ActivationMatrix a_L = activations.getNetworkOutput(); // Activations de la dernière couche APRES fonction d'activation
        ActivationMatrix z_L = activations.getNetworkOutput_BeforeAF(); // Activations de la dernière couche AVANT fonction d'activation

        // Vérification des dimensions de sortie
        assert(expectedOutput.hasSameDimensions(a_L)) : String.format(
                "Dimensions de sortie incorrectes: attendu %dx%d, obtenu %dx%d",
                a_L.getNumberOfRows(), a_L.getNumberOfColumns(),
                expectedOutput.getNumberOfRows(), expectedOutput.getNumberOfColumns());


        GradientMatrix delta_L = this.computeOutputGradient(lossFunction, expectedOutput, a_L, z_L, parameterRegularization);

        // Calcul des gradients pour la couche de sortie
        ActivationMatrix a_L_minus_1 = (L-1 > 0) ? activations.getResult_PostAF(L-2) : input;
        GradientMatrix dL_dW_L = delta_L.multiplyAtRight(a_L_minus_1.transpose());

        BiasVector dL_db_L = delta_L.sumErrorTerm();
        gradients.add(dL_dW_L, dL_db_L);

        // Rétropropagation à travers les couches cachées
        GradientMatrix delta_l_plus_1 = delta_L;

        // Rétropropagation à travers les couches cachées

        for (int l = L-2; l >= 0; l--) {

            ActivationMatrix z_l = activations.getResult_PreAF(l); // Activations Pre-AF
            ActivationMatrix a_l_minus_1 = (l > 0) ? activations.getResult_PostAF(l-1) : input; // Activations Post-AF

            Layer layer_l_plus_1 = this.layers.get(l+1);
            Layer layer_l = this.layers.get(l);

            WeightMatrix W_l_plus_1_T = layer_l_plus_1.getWeightMatrix().transpose();
            ActivationMatrix sigma_prime_z_l = layer_l.getDerivativeOfAF().apply(z_l);
            GradientMatrix delta_l = delta_l_plus_1
                    .multiply(W_l_plus_1_T)
                    .hadamardProduct(sigma_prime_z_l);

            // Finalement : calcul du gradient des poids et du biais
            GradientMatrix dL_dW_l = delta_l.multiplyAtRight(a_l_minus_1.transpose());
            BiasVector dL_db_l = delta_l.sumErrorTerm();

            // Application de la régularization en fonction des paramètres, sur les gradients calculés
            if(parameterRegularization != null) parameterRegularization.accept(dL_dW_l, layer_l.getWeightMatrix());

            gradients.add(dL_dW_l, dL_db_l);
            delta_l_plus_1 = delta_l;

        }

        return gradients;

    }

    private GradientMatrix computeOutputGradient(LossFunction lossFunction, ActivationMatrix expectedOutput,
                                                 ActivationMatrix a_L, ActivationMatrix z_L, ParameterRegularization regularization) {

        GradientMatrix delta_L;

        // Cas spécial pour Softmax + Entropie croisée
        // Pour Softmax avec CE, delta_L est simplement (a_L - expected)

        if (getLastLayer().getActivationFunction() == SoftMax && lossFunction instanceof CE) {
            delta_L = a_L.substract(expectedOutput).toGradientMatrix();

            // TODO verifier que ça marche
            if(regularization != null) delta_L.add(regularization.computePenalty(this));
            // Calcul standard pour les autres AF:
        } else {

            // Calcul de dL/da_L [sortie attendue]
            GradientMatrix dL_da_L = lossFunction.applyDerivative(a_L, expectedOutput);
            if(regularization != null) dL_da_L.add(regularization.computePenalty(this));
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


    public void serialize(String modelName) {
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(modelName + ".mlp");
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
            this.writeObject(objectOutputStream);
            objectOutputStream.flush();
            objectOutputStream.close();
        } catch (Exception e) {
            throw new RuntimeException();
        }

    }

    // TODO plus d'insight pour les exceptions.
    public static MLP importModel(String modelName) {
        MLP res;
        try {
            FileInputStream fileInputStream
                    = new FileInputStream(modelName + ".mlp");
            ObjectInputStream objectInputStream
                    = new ObjectInputStream(fileInputStream);
            res = MLP.readObject(objectInputStream);
            objectInputStream.close();
        } catch (FileNotFoundException e) {
            System.out.println("Erreur : Modèle '" + modelName + "' non trouvé. Vérifiez le nom du fichier");
            System.out.println("Pour importer le modèle 'mnist_resolver.mlp', " +
                    "l'argument doit être 'mnist_resolver' (sans l'extension).");
            throw new RuntimeException(e);
        } catch (ClassNotFoundException | IOException e) {
            throw new RuntimeException(e);
        }

        return res;
    }



    public void writeObject(java.io.ObjectOutputStream out) throws IOException {
         out.writeObject(this.dimInput); // Dimension de l'entrée
         out.writeObject(this.getLayers().size()); // Nombre de layers
        for(Layer layer : this.layers) { // Sauvegarde des layers avec leur contenu (AF / Poids & Biais
            layer.writeObject(out);
        }
    }
    public static MLP readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        MLP nullMLP = MLP.builder(1).addIdentityLayer(1).addIdentityLayer(1).build();
        int dimInput = (Integer) in.readObject();
        int layersSize = (Integer) in.readObject();
        nullMLP.dimInput = dimInput;
        nullMLP.layers = new ArrayList<>(layersSize);
        for(int i = 0; i < layersSize; i++){
            Layer layer_i = new Layer(null, null, null);
            layer_i.readObject(in);
            nullMLP.getLayers().add(layer_i);
        }
        return nullMLP;
    }

    public void readObjectNoData() throws ObjectStreamException {

    };

    /**
     * Encapsule simplement les résultats de {@link #feedForward} du réseau
     * pour des notations plus compactes
     */
    public static class FeedForwardResult {

        List<Pair<ActivationMatrix, ActivationMatrix>> results;

        /**
         * Renvoie la {@link ActivationMatrix} de la couche n°i du MLP,
         * avant application de la fonction d'activation de la couche.
         * @param i
         * @return
         */
        public ActivationMatrix getResult_PreAF(int i){
            return this.results.get(i).getB();
        }

        public FeedForwardResult() {
            this.results = new ArrayList<>();
        }

        /**
         * Renvoie la {@link ActivationMatrix} de la couche n°i du MLP,
         * après application de la fonction d'activation de la couche.
         * @param i
         * @return
         */
        public ActivationMatrix getResult_PostAF(int i) {
            return this.results.get(i).getA();
        }

        public void add(ActivationMatrix activationMatrixOfLayer, ActivationMatrix activationsBeforeAF) {
            this.results.add(new Pair<>(activationMatrixOfLayer, activationsBeforeAF));
        }

        /**
         * Renvoie la matrice d'activation de la dernière couche du réseau,
         * après application de la fonction d'activation.
         * @return
         */
        public ActivationMatrix getNetworkOutput() {
            return this.results.getLast().getA();
        }

        /**
         * Renvoie la matrice d'activation de la dernière couche du réseau,
         * avant application de la fonction d'activation.
         * @return
         */
        public ActivationMatrix getNetworkOutput_BeforeAF() {
            return this.results.getLast().getB();
        }
    }

    /**
     * Encapsule simplement les résultats de {@link #backpropagate} du réseau
     * pour des notations plus compactes
     * Permet de récupérer une {@link GradientMatrix} qui correspond au gradient de poids
     * et un {@link BiasVector} qui correspond au gradient de biais
     */
    public static class BackProResult {

        List<Pair<GradientMatrix, BiasVector>> results;

        public BackProResult() {
            this.results = new ArrayList<>();
        }

        /**
         * Renvoie la {@link GradientMatrix} de la couche n°i du MLP.
         * @param i
         * @return
         */
        public GradientMatrix getWeightGradient(int i) {
            return this.results.get(i).getA();
        }

        public BackProResult(List<Pair<GradientMatrix, BiasVector>> results) {
            this.results = results;
        }

        /**
         * Renvoie le {@link BiasVector} de la couche n°i du MLP.
         * @param i
         * @return
         */
        public BiasVector getBiasGradient(int i) {
            return this.results.get(i).getB();
        }

        public void add(GradientMatrix weightGradientOfLayer, BiasVector biasGradientOfLayer) {
            this.results.addFirst(new Pair<>(weightGradientOfLayer, biasGradientOfLayer));
        }

        /**
         * Renvoie la matrice de gradient des poids de la dernière couche du réseau.
         * @return
         */
        public GradientMatrix getLastWeightGradient() {
            return this.results.getLast().getA();
        }

        /**
         * Renvoie le vecteur de gradient des biais de la dernière couche du réseau.
         * @return
         */
        public BiasVector getLastBiasGradient() {
            return this.results.getLast().getB();
        }

        public int size() {
            return this.results.size();
        }

        public List<GradientMatrix> getGradients() {
            List<GradientMatrix> res = new ArrayList<>();
            for(Pair<GradientMatrix,BiasVector> bpr : this.results) {
                res.add(bpr.getA());

            }

            return res;
        }
    }


    public double getWeightsNorm(){
        return mapWeightsToDouble(Math::abs);
    }

    public double getWeightsSignum(){
        return mapWeightsToDouble(Math::signum);
    }

    public double mapWeightsToDouble(Function<Double,Double> map){
        double[] res = {0};
        for(Layer l : getLayers()) {
            l.getWeightMatrix().forEach(d -> res[0] += Math.signum(d));
        }

        return res[0];
    }


}
