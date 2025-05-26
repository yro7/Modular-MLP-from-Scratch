package gui;

import java.awt.*;
import javax.swing.*;
import javax.swing.event.*;

public class AffichageData extends JPanel{

    private final static String[] listeModes =  {
        "disques",
        "clusters",
        "carres",
        "spirales"
    };

    private Data data;
    private JLabel dessin;
    private JPanel controle;
    private String modeData;

    public AffichageData(Data data){
        super();

        this.modeData = "disques";

        this.setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        this.data = data;

        this.dessin = new JLabel(new ImageIcon(getClass().getResource("assets/"+modeData+".png")));

        this.controle = new JPanel();
        this.controle.setLayout(new FlowLayout());
        this.add(this.controle);
        this.add(dessin);

        this.maj();
    } 

    public void maj(){
        // dessin
        this.majDessin();

        // controle
        this.controle.removeAll();
        JList<String> modes = new JList<String>(listeModes);
        modes.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        modes.setSelectedIndex(0);
        modes.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        modes.addListSelectionListener(new ListSelectionListener() {
            @Override
            public void valueChanged(ListSelectionEvent e) {
                modeData = (String) modes.getSelectedValue();
                majDessin();
            }
        });
        this.controle.add(modes);

        JButton generer = new JButton("Générer");
        generer.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        generer.addActionListener(ev -> {
            this.data.generer(this.modeData);
        });
        this.controle.add(generer);

        this.revalidate();
        this.repaint();
    } 

    public void majDessin() {
        this.remove(dessin);
        this.dessin = new JLabel(new ImageIcon(getClass().getResource("assets/"+modeData+".png")));
        this.add(dessin);
        this.revalidate();
        this.repaint();
    }
}
