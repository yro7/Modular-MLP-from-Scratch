package gui;

import java.util.ArrayList;

import mlps.MLP;
import mlps.MLPBuilder;
import matrices.*;
import mlps.optimizers.Optimizer;
import mlps.optimizers.SGD;

import static functions.ActivationFunction.*;
import static functions.LossFunction.*;

/**
 * Contient le modèle et les méthodes nécessaires pour
 * les intéractions sur l'interface.
 */
public class Modele extends java.util.Observable {
    private ArrayList<Integer> liste;
    private double learningRate = 0.001;
    private String activationFunction = "ReLU";
    private String regularization = "L2"; // ou ElasticNet, L1
    private double regularizationRate = 0.0;
    private String problemType = "classification";
    private MLP trainedMLP;
    private double[][] predictions;

    public Modele() {
        this.liste = new ArrayList<Integer>();
        this.liste.add(2);
        this.trainedMLP = null;
        this.predictions = null;
    } 

    public int taille() {
        return this.liste.size();
    } 

    public int taille(int index) {
        if (0 <= index && index < this.liste.size()) { 
            return this.liste.get(index);
        }
        return 0;
    } 

    public void ajouterCouche() {
        if (this.liste.size() < 6) { 
            this.liste.add(2);
            this.avertir();
        } 
    } 

    public void retirerCouche() {
        if (this.liste.size() > 1) { 
            this.liste.remove(this.liste.size() - 1);
            this.avertir();
        } 
    } 

    public void ajouterNeurone(int index) {
        int taille = this.taille(index);
        if (0 < taille && taille < 8) {
            this.liste.set(index, taille + 1);
            this.avertir();
        } 
    } 

    public void retirerNeurone(int index) {
        int taille = this.taille(index);
        if (taille > 1) {
            this.liste.set(index, taille - 1);
            this.avertir();
        } 
    }    

    public void setLearningRate(double rate) {
        this.learningRate = rate;
    }
    
    public void setActivation(String activation) {
        this.activationFunction = activation;
    }
    
    public void setRegularization(String reg) {
        this.regularization = reg;
    }
    
    public void setRegularizationRate(double rate) {
        this.regularizationRate = rate;
    }
    
    public void setProblemType(String type) {
        this.problemType = type;
    }

    private functions.ActivationFunction getActivationFunction() {
        switch (activationFunction) {
            case "ReLU": return ReLU;
            case "Tanh": return TanH;
            case "Sigmoid": return Sigmoid;
            case "Identity": return Identity;
            case "SoftMax": return SoftMax;
            default: return ReLU;
        }
    }

    private MLP construireMLP(int dimInput) {
        MLPBuilder mlpBuilder = MLP.builder(dimInput).setRandomSeed(3);
        for (int i = 0; i < this.taille(); i++) {
            mlpBuilder.addLayer(this.taille(i),this.getActivationFunction());
        }
        if (problemType.equals("classification")) {
            mlpBuilder.addLayer(1, Sigmoid); // binaire, sinon SoftMax à adapter
        } else {
            mlpBuilder.addLayer(1, Identity); // sortie linéaire pour régression
        }
        return mlpBuilder.build();
    } 

    // permet de retourner la training loss, utile pour l'affichage des résultats.
    public double trainingloss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique) {
        return mlp.computeLoss(batchInput, batchTheorique, CE);
    }

    public void commencer(Data data) {
        double[][] input = data.getData();
        double[][] output = data.getResult();

        ActivationMatrix batchInput = new ActivationMatrix(input);
        ActivationMatrix batchTheorique = new ActivationMatrix(output);

        this.trainedMLP = construireMLP(input[0].length);

        Optimizer sgd = new SGD(this.learningRate);
        for (int i = 0; i < 10_000; i++) {
            this.trainedMLP.updateParameters(batchInput, batchTheorique, MSE, sgd, null);
        }

        ActivationMatrix outputMatrix = this.trainedMLP.feedForward(batchInput).getNetworkOutput();
        double[][] rawOutput = outputMatrix.toArray();
        this.predictions = new double[rawOutput.length][1];
        for (int i = 0; i < rawOutput.length; i++) {
            this.predictions[i][0] = rawOutput[i][0] >= 0.5 ? 1 : 0;
        }

        this.avertir();
    } 

    public double[][] getPredictions() {
        return this.predictions;
    }

    public MLP getTrainedMLP() {
        return this.trainedMLP;
    }

    private void avertir() {
        this.setChanged();
        this.notifyObservers();
    } 
}
