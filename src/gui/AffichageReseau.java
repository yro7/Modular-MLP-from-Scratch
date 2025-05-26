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
        // Vider le conteneur principal
        this.removeAll();
        
        // Disposer les couches horizontalement
        this.setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
        
        // Pour chaque couche
        for (int i = 0; i < modele.taille(); i++){
            final int indexCouche = i;
            
            // Création de la couche principale avec BorderLayout
            JPanel couche = new JPanel(new BorderLayout());
            couche.setAlignmentY(Component.TOP_ALIGNMENT);
            
            // -------------------------
            // Panel de contrôle (boutons)
            // -------------------------
            // Utilisation d'un FlowLayout pour placer les boutons côte à côte
            JPanel controles = new JPanel(new FlowLayout(FlowLayout.CENTER, 5, 5));
            // Fixer la taille du panel pour que la paire de boutons n'occupe pas plus d'espace
            // Ici, on l'augmente un peu pour tenir compte des boutons plus larges : par exemple 180 pixels de large
            Dimension controlSize = new Dimension(180, 30);
            controles.setPreferredSize(controlSize);
            controles.setMinimumSize(controlSize);
            controles.setMaximumSize(controlSize);
            
            // Bouton "–"
            JButton boutonMoins = new JButton("-");
            boutonMoins.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            // Augmenter la largeur du bouton : maintenant 80 px de large, 30 px de haut
            Dimension btnSize = new Dimension(80, 30);
            boutonMoins.setPreferredSize(btnSize);
            boutonMoins.setMinimumSize(btnSize);
            boutonMoins.setMaximumSize(btnSize);
            boutonMoins.addActionListener(ev -> {
                modele.retirerNeurone(indexCouche);
            });
            controles.add(boutonMoins);
            
            // Bouton "+"
            JButton boutonPlus = new JButton("+");
            boutonPlus.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            boutonPlus.setPreferredSize(btnSize);
            boutonPlus.setMinimumSize(btnSize);
            boutonPlus.setMaximumSize(btnSize);
            boutonPlus.addActionListener(ev -> {
                modele.ajouterNeurone(indexCouche);
            });
            controles.add(boutonPlus);
            
            // Placer le panel de contrôle en haut de la couche (zone NORTH)
            couche.add(controles, BorderLayout.NORTH);
            
            // -------------------------
            // Panel pour afficher les neurones
            // -------------------------
            // On utilise un BoxLayout vertical pour empiler les neurones
            JPanel panelNeurones = new JPanel();
            panelNeurones.setLayout(new BoxLayout(panelNeurones, BoxLayout.Y_AXIS));
            
            // Ajouter une glue en haut pour permettre le centrage vertical du contenu
            panelNeurones.add(Box.createVerticalGlue());
            
            // Pour chaque neurone, on l'ajoute et on insère un espace fixe après
            for (int j = 0; j < modele.taille(i); j++){
                JLabel neurone = new JLabel(new ImageIcon(getClass().getResource("assets/neurone.png")));
                neurone.setAlignmentX(Component.CENTER_ALIGNMENT);
                panelNeurones.add(neurone);
                panelNeurones.add(Box.createVerticalStrut(10));
            }
            
            // Ajouter une glue en bas pour compléter le centrage
            panelNeurones.add(Box.createVerticalGlue());
            
            // Fixer une hauteur minimale pour le panel des neurones afin que la glue ait de l'espace pour centrer.
            Dimension neuronesDim = panelNeurones.getPreferredSize();
            int minimumHeight = 200; // Vous pouvez ajuster cette valeur selon vos besoins
            if(neuronesDim.height < minimumHeight) {
                neuronesDim.height = minimumHeight;
                panelNeurones.setPreferredSize(neuronesDim);
            }
            
            // Ajouter le panel des neurones à la zone CENTER de la couche
            couche.add(panelNeurones, BorderLayout.CENTER);
            
            // Optionnel : ajouter une bordure autour de chaque couche pour les espacer visuellement
            couche.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
            
            // Ajouter la couche dans le conteneur principal (les couches s'afficheront côte à côte)
            this.add(couche);
        }
        
        // Mise à jour de l'affichage
        this.revalidate();
        this.repaint();
    }
    
    
    
}
