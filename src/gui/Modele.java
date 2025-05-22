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
public class Modele extends java.util.Observable{
    private ArrayList<Integer> liste;

    public Modele(){
        this.liste = new ArrayList<Integer>();
        this.liste.add(2);
    } 

    public int taille(){
        return this.liste.size();
    } 

    public int taille(int index){
        if (0 <= index && index < this.liste.size()){ 
            return this.liste.get(index);
        }
        return 0;
    } 

    public void ajouterCouche(){
        if (this.liste.size() < 6){ 
            this.liste.add(2);
            this.avertir();
        } 
    } 

    public void retirerCouche(){
        if (this.liste.size() > 1){ 
            this.liste.remove(this.liste.size() - 1);
            this.avertir();
        } 
    } 

    public void ajouterNeurone(int index){
        int taille = this.taille(index);
        if (0 < taille && taille < 8){
            this.liste.set(index, taille+1);
            this.avertir();
        } 
    } 

    public void retirerNeurone(int index){
        int taille = this.taille(index);
        if (taille > 1){
            this.liste.set(index, taille-1);
            this.avertir();
        } 
    }    

    private MLP construireMLP(int dimInput){
        MLPBuilder mlpBuilder = MLP.builder(dimInput).setRandomSeed(3);
        for (int i = 0; i < this.taille(); i++){
            mlpBuilder.addLayer(this.taille(i), ReLU);
        } 
        mlpBuilder.addLayer(1, Sigmoid);
        return mlpBuilder.build();
    } 

    public void commencer(Data data){
        double[][] input = data.getData();
        double[][] output = data.getResult();

        ActivationMatrix batchInput = new ActivationMatrix(input);
        ActivationMatrix batchTheorique = new ActivationMatrix(output);

        MLP mlp = construireMLP(input[0].length);

        //mlp.feedForward(batchInput).getNetworkOutput().print();

        Optimizer sgd = new SGD(0.001);
        for(int i = 0; i < 10_000; i++){
            mlp.updateParameters(batchInput, batchTheorique, MSE, sgd, null);
        };

        //mlp.feedForward(batchInput).getNetworkOutput().print();
    } 

    private void avertir(){
        this.setChanged();
        this.notifyObservers();
    } 
}
