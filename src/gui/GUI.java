package gui;

import javax.swing.*;
import java.awt.*;

/**
 * Représente la fenêtre de l'interface graphique.
 * Permet de lancer le GUI.
 */
public class GUI extends JFrame {
    public GUI() {
        Data data = new Data();
        Modele modele = new Modele();

        ControleurReseau controle = new ControleurReseau(modele, data);
        AffichageReseau vueReseau = new AffichageReseau(modele);

        this.getContentPane().setLayout(new BorderLayout());
        this.getContentPane().add(vueReseau, BorderLayout.CENTER);
        this.getContentPane().add(controle, BorderLayout.NORTH);
        
        AffichageData vueData = new AffichageData(data);

        this.getContentPane().add(vueData, BorderLayout.WEST);

        this.setSize(new Dimension(1280,720));
        this.setLocationRelativeTo(null);
        this.setVisible(true);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    } 

    public static void main(String[] args) {
        new GUI();
    } 
}
