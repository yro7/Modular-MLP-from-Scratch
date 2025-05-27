package gui;

import javax.swing.*;

import static functions.ActivationFunction.valueOf;

import java.awt.*;

public class ControleurReseau extends JPanel {
    public ControleurReseau(Modele modele, Data data) {
        super(new FlowLayout());
        final JLabel couche = new JLabel("Nombre de couches");
        final JLabel label1 = new JLabel("Learning rate");
        final JLabel label2 = new JLabel("Activation");
        final JLabel label3 = new JLabel("Regularization");
        final JLabel label4 = new JLabel("Regularization rate");
        final JLabel label5= new JLabel("Problem type");
        
        final JButton START = new JButton("Commencer");
        final JButton DEC = new JButton("-");
        final JButton INC = new JButton("+");
        JPanel Panel1 = new JPanel();
        JPanel Panel2 = new JPanel();
        JPanel learningPanel = new JPanel();
        JPanel activationPanel = new JPanel();
        JPanel regularizationPanel = new JPanel();
        JPanel regulratePanel = new JPanel();
        JPanel problemPanel = new JPanel();
        Panel1.setLayout(new FlowLayout());
        Panel2.setLayout(new BoxLayout(Panel2,BoxLayout.Y_AXIS));
        learningPanel.setLayout(new BoxLayout(learningPanel,BoxLayout.Y_AXIS));
        activationPanel.setLayout(new BoxLayout(activationPanel,BoxLayout.Y_AXIS));
        regularizationPanel.setLayout(new BoxLayout(regularizationPanel,BoxLayout.Y_AXIS));
        regulratePanel.setLayout(new BoxLayout(regulratePanel,BoxLayout.Y_AXIS));
        problemPanel.setLayout(new BoxLayout(problemPanel,BoxLayout.Y_AXIS));

        String[] learningrates = {"0.00001","0.0001","0.001","0.003","0.01","0.03","0.1","0.3","1","3","10"};

        final JComboBox<String> learnbox = new JComboBox<>(learningrates);

        String[] activation = {"ReLU","Tanh","Sigmoid","Identity","SoftMax"};

        final JComboBox<String> activebox = new JComboBox<>(activation);

        String[] regularization = {"ElasticNet","L1","L2"};

        final JComboBox<String> regulatebox = new JComboBox<>(regularization);
        String[] regularizationrates = {"0","0.001","0.003","0.01","0.03","0.1","0.3","1","3","10"};

        final JComboBox<String> regulratebox = new JComboBox<>(regularizationrates);
        String[] problemtype = {"classification","regression"};

        final JComboBox<String> problemtypebox = new JComboBox<>(problemtype);

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
        learnbox.addActionListener(ev -> {
            double rate = Double.valueOf((String)learnbox.getSelectedItem());
            modele.setLearningRate(rate);
        });
        activebox.addActionListener(ev -> {
            String activate = (String)activebox.getSelectedItem();
            modele.setActivation(activate);
        });
        regulatebox.addActionListener(ev -> {
            String Regularization = (String)regulatebox.getSelectedItem();
            modele.setRegularization(Regularization);
        });
        regulratebox.addActionListener(ev -> {
            double Regularizationrate = Double.valueOf((String)regulratebox.getSelectedItem());
            modele.setRegularizationRate(Regularizationrate);
        });
        learnbox.addActionListener(ev -> {
            String type = (String)learnbox.getSelectedItem();
            modele.setProblemType(type);
        });
        this.add(START);
        Panel1.add(DEC);
        Panel1.add(INC);
        Panel2.add(couche);
        Panel2.add(Panel1);
        this.add(Panel2);
        learningPanel.add(label1);
        learningPanel.add(learnbox);
        activationPanel.add(label2);
        activationPanel.add(activebox);
        regularizationPanel.add(label3);
        regularizationPanel.add(regulatebox);
        regulratePanel.add(label4);
        regulratePanel.add(regulratebox);
        problemPanel.add(label5);
        problemPanel.add(problemtypebox);
        

        this.add(learningPanel);
        this.add(activationPanel);
        this.add(regularizationPanel);
        this.add(regulratePanel);
        this.add(problemPanel);
        
    } 
}
