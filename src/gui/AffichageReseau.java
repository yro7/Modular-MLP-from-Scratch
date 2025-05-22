package gui;

import java.awt.*;
import javax.swing.*;

/**
 * Représente la partie affichage du réseau de neuronne
 * et les boutons de contôle avec.
 */
public class AffichageReseau extends JPanel {
    private Modele modele;

    public AffichageReseau(Modele modele){
        super();
        this.setLayout(new FlowLayout());
        this.modele = modele;
        this.maj();

        modele.addObserver(new java.util.Observer() {
            public void update(java.util.Observable o, Object arg) {
                maj();
            } 
        });
    } 

    private void maj() {
        // Vider
        this.removeAll();

        // Remplir affichage
        for (int i = 0; i < modele.taille(); i++){
            final int indexCouche = i;

            JPanel couche = new JPanel();
            couche.setLayout(new BorderLayout());

            // boutons de controle
            JPanel controles = new JPanel();
            controles.setLayout(new FlowLayout());

            JButton DEC = new JButton("-");
            DEC.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            DEC.addActionListener(ev -> {
                modele.retirerNeurone(indexCouche);
            });
            controles.add(DEC);

            JButton INC = new JButton("+");
            INC.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            INC.addActionListener(ev -> {
                modele.ajouterNeurone(indexCouche);
            });
            controles.add(INC);

            couche.add(controles, BorderLayout.NORTH);

            // dessin des neuronnes
            JPanel dessin = new JPanel();
            dessin.setLayout(new BoxLayout(dessin, BoxLayout.Y_AXIS));
            
            for (int j = 0; j < modele.taille(i); j++){
                JLabel neurone = new JLabel(new ImageIcon(getClass().getResource("neurone.png")));
                dessin.add(neurone);
            } 

            couche.add(dessin, BorderLayout.CENTER);

            this.add(couche);
        }

        this.revalidate();
        this.repaint();
    } 
}
