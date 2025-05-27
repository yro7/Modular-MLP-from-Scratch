package gui;

import javax.swing.*;
import java.awt.*;

public class OngletClassification extends JPanel {
    public OngletClassification() {
        Data data = new Data();
        Modele modele = new Modele();
        
        ControleurReseau controle = new ControleurReseau(modele, data);
        AffichageReseau vueReseau = new AffichageReseau(modele);

        this.setLayout(new BorderLayout());
        this.add(vueReseau, BorderLayout.CENTER);
        this.add(controle, BorderLayout.NORTH);
        
        AffichageData vueData = new AffichageData(data);
        this.add(vueData, BorderLayout.WEST);

        AffichageResultat vueResultat = new AffichageResultat(data, modele);
        this.add(vueResultat, BorderLayout.EAST);

        
    }
}
