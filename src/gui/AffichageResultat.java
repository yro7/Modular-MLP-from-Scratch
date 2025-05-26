package gui;

import javax.swing.*;
import java.awt.*;
import matrices.ActivationMatrix;
import mlps.MLP;

public class AffichageResultat extends JPanel {
    private Data data;
    private Modele modele;
    private double[][] donnee;
    private double[][] predictedLabels;

    public AffichageResultat(Data data, Modele modele) {
        super();
        this.setBackground(Color.WHITE);
        this.setPreferredSize(new Dimension(400, 400));
        this.setLayout(new FlowLayout());
        
        this.data = data;
        this.modele = modele;
        this.donnee = data.getData();
        updatePredictions();

        // Observe model changes to update predictions
        modele.addObserver((o, arg) -> {
            updatePredictions();
            repaint();
        });
    }

    private void updatePredictions() {
        if (donnee == null) return;

        // Run data through the neural network to get predictions
        ActivationMatrix batchInput = new ActivationMatrix(donnee);
        MLP mlp = modele.construireMLP(donnee[0].length);
        ActivationMatrix output = mlp.feedForward(batchInput).getNetworkOutput();
        
        // Convert output to binary predictions (threshold at 0.5 for sigmoid output)
        double[][] rawOutput = output.toArray();
        predictedLabels = new double[rawOutput.length][1];
        for (int i = 0; i < rawOutput.length; i++) {
            predictedLabels[i][0] = rawOutput[i][0] >= 0.5 ? 1 : 0;
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g); // Clear the background
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        if (donnee == null || predictedLabels == null) return;

        // Define margins and scale for XOR data (range [0,1])
        int margin = 20;
        int plotSize = Math.min(getWidth(), getHeight()) - 2 * margin;
        float pointSize = 8f; // Size of each point

        for (int i = 0; i < donnee.length; i++) {
            double abscisse = donnee[i][0]; // x-coordinate (0 to 1)
            double ordonnee = donnee[i][1]; // y-coordinate (0 to 1)

            // Set color based on predicted class
            if (predictedLabels[i][0] == 1) {
                g2.setColor(Color.RED);
            } else {
                g2.setColor(Color.BLUE); // Use blue instead of black for better contrast
            }

            // Scale coordinates to panel size
            int x = margin + (int) (abscisse * plotSize);
            int y = margin + (int) ((1 - ordonnee) * plotSize); // Flip y-axis for intuitive display

            // Draw filled circle for each point
            g2.fillOval(x - (int)(pointSize/2), y - (int)(pointSize/2), (int)pointSize, (int)pointSize);
        }

        // Draw axes
        g2.setColor(Color.BLACK);
        g2.drawLine(margin, margin + plotSize, margin + plotSize, margin + plotSize); // x-axis
        g2.drawLine(margin, margin, margin, margin + plotSize); // y-axis
    }
}   
