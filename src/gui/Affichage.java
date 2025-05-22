package gui;

import java.awt.*;
import javax.swing.*;

/**
 * Représente la partie affichage du réseau de neuronne
 * et les boutons de contôle avec.
 */
public class Affichage extends JPanel {
    private Modele modele;
    private AffichageReseau centre;

    public Affichage(Modele modele){
        super(new BorderLayout());
        this.modele = modele;
        this.centre = new AffichageReseau();
        this.maj();

        modele.addObserver(new java.util.Observer() {
            public void update(java.util.Observable o, Object arg) {
                maj();
            } 
        });
    } 

    private void maj() {
        // Partie centrale avec le réseau
        this.centre.maj(modele);

        this.revalidate();
        this.repaint();
    } 
}
