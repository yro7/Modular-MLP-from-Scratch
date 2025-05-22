package gui;

import java.awt.*;
import javax.swing.*;

public class AffichageData extends JPanel{

    private Data data;
    private JPanel dessin;
    private JPanel controle;

    public AffichageData(Data data){
        super();
        this.setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        this.data = data;

        this.dessin = new JPanel();
        this.dessin.setPreferredSize(new Dimension(400, 400));
        //this.add(this.dessin);

        this.controle = new JPanel();
        this.controle.setLayout(new FlowLayout());
        this.add(this.controle);

        this.maj();
    } 

    public void maj(){
        this.dessin.removeAll();
        this.controle.removeAll();
        // TODO dessin

        // controle
        JButton generer = new JButton("Générer");
        generer.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        this.controle.add(generer);

        this.revalidate();
        this.repaint();
    } 
}
