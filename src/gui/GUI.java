package gui;

import javax.swing.*;
import java.awt.*;

/**
 * Représente la fenêtre de l'interface graphique.
 * Permet de lancer le GUI.
 */
public class GUI extends JFrame {
    public GUI() {
        JTabbedPane onglet = new JTabbedPane();

        OngletClassification ongletclassification = new OngletClassification();
        onglet.addTab("Onglet 1", ongletclassification);
        
        Predicteurchiffre onglet2 = new Predicteurchiffre();
        onglet.addTab("Onglet 2", onglet2.getFrame());

        this.getContentPane().add(onglet);
        this.setSize(new Dimension(1280,720));
        this.setLocationRelativeTo(null);
        this.setVisible(true);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    } 

    public static void main(String[] args) {
        new GUI();
    } 
}
