package gui;

import javax.swing.*;
import java.awt.*;

public class ControleurReseau extends JPanel {
    public ControleurReseau(Modele modele, Data data) {
        super(new FlowLayout());
        final JLabel couche = new JLabel("Nombre de couches");

        final JButton START = new JButton("Commencer");
        final JButton DEC = new JButton("-");
        final JButton INC = new JButton("+");

        START.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        INC.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        DEC.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));

        START.addActionListener(ev -> {
            modele.commencer(data);
        });
        DEC.addActionListener(ev -> {
            modele.retirerCouche();
        });
        INC.addActionListener(ev -> {
            modele.ajouterCouche();
        });

        this.add(START);
        this.add(DEC);
        this.add(INC);
        this.add(couche);
    } 
}
