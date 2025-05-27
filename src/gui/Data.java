package gui;

import java.util.Observable;
import java.util.Random;

public class Data extends Observable {

    final static int NBPOINTS = 1000;
    

    private double[][] data;
    private double[][] result;

    public Data(){
        double[][] xorData = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        double[][] xorResult = {
            {0},
            {1},
            {1},
            {0},
        };
        this.data = xorData;
        this.result = xorResult;
    } 

    private void addElement(double x, double y, double label) {
        double[][] nvData = new double[this.data.length + 1][2];
        System.arraycopy(this.data, 0, nvData, 0, this.data.length);
        nvData[this.data.length][0] = x;
        nvData[this.data.length][1] = y;
        this.data = nvData;

        double[][] nvResult = new double[this.result.length + 1][1];
        System.arraycopy(this.result, 0, nvResult, 0, this.result.length);
        nvResult[this.result.length][0] = label;
        this.result = nvResult;

    }

    public void generer(String forme){
        // Vider data et result
        this.data = new double[0][0];
        this.result = new double[0][0];
        
        Random rd = new Random();

        if (forme.equals("disques")) {
            // 1er disque de taille 3 / label=0
            for (int i=0; i < NBPOINTS/2; i++) {
                double rayon = rd.nextDouble() * 3.0;
                double angle = rd.nextDouble() * 2 * Math.PI;
                double x = rayon * Math.cos(angle);
                double y = rayon * Math.sin(angle);
                this.addElement(x, y, 0);
            }

            // 2e disque de taille 6
            for (int i=0; i < NBPOINTS/2; i++) {
                double rayon = (rd.nextDouble() + 1.0) * 3.0;
                double angle = rd.nextDouble() * 2 * Math.PI;
                double x = rayon * Math.cos(angle);
                double y = rayon * Math.sin(angle);
                this.addElement(x, y, 1);
            }
        } else if (forme.equals("clusters")) {
            // 1er cluster en (-3,-3) label=0
            for (int i=0; i < NBPOINTS/2; i++) {
                double rayon = rd.nextDouble() * 3.0;
                double angle = rd.nextDouble() * 2 * Math.PI;
                double x = rayon * Math.cos(angle) - 3;
                double y = rayon * Math.sin(angle) - 3;
                this.addElement(x, y, 0);
            }
            // 2e cluster en (3,3) label=1
            for (int i=0; i < NBPOINTS/2; i++) {
                double rayon = rd.nextDouble() * 3.0;
                double angle = rd.nextDouble() * 2 * Math.PI;
                double x = rayon * Math.cos(angle) + 3;
                double y = rayon * Math.sin(angle) + 3;
                this.addElement(x, y, 1);
            }
        } else if (forme.equals("carres")) {
            // carre en (3,3) label=0
            for (int i=0; i < NBPOINTS/4; i++) {
                double x = rd.nextDouble() * 6;
                double y = rd.nextDouble() * 6;
                this.addElement(x, y, 0);
            }
            // carre en (-3,-3) label=0
            for (int i=0; i < NBPOINTS/4; i++) {
                double x = rd.nextDouble() * 6 - 6;
                double y = rd.nextDouble() * 6 - 6;
                this.addElement(x, y, 0);
            }
            // carre en (-3,3) label=1
            for (int i=0; i < NBPOINTS/4; i++) {
                double x = rd.nextDouble() * 6 - 6;
                double y = rd.nextDouble() * 6 ;
                this.addElement(x, y, 1);
            }
            // carre en (3,-3) label=1
            for (int i=0; i < NBPOINTS/4; i++) {
                double x = rd.nextDouble() * 6;
                double y = rd.nextDouble() * 6 - 6;
                this.addElement(x, y, 1);
            }
        } else {
            System.out.println("forme non reconnue : " + forme);
        }
        
        // Notify observers of data change // ADDED
        setChanged(); // ADDED
        notifyObservers(); // ADDED
    }

    public double[][] getData(){
        return this.data;
    } 

    public double[][] getResult(){
        return this.result;
    } 

    public void setData(double[][] data){
        this.data = data;
        setChanged(); // ADDED
        notifyObservers(); // ADDED
    } 

    public void setResult(double[][] result){
        this.result = result;
        setChanged(); // ADDED
        notifyObservers(); // ADDED
    } 
}
