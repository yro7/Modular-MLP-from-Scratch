package gui;

import javax.swing.*;
import java.awt.*;

public class AffichageResultat extends JPanel {
    private Data data;
    private double[][] label;
    private Modele modele;
    private double[][] donnee;
    private double[][] predictedLabels;

    public AffichageResultat(Data data, Modele modele) {
        super();
        this.setBackground(Color.WHITE); // Couleur de fond
        this.setPreferredSize(new Dimension(335, 335));
        this.setLayout(new FlowLayout());
        
        this.data = data;
        this.modele = modele;
        this.donnee = data.getData();
        this.label = data.getResult();
        updatePredictions();

        // Observer les changements de modèle
        modele.addObserver((o, arg) -> {
            updatePredictions();
            repaint();
        });

        // Observer les changements de data
        data.addObserver((o, arg) -> {
            this.donnee = data.getData();
            this.label = data.getResult();
            updatePredictions();
            repaint();
        });
    }

    private void updatePredictions() {
        this.donnee = data.getData();
        this.predictedLabels = modele.getPredictions();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g); // Supprimer le fond
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        if (donnee == null || donnee.length == 0 || predictedLabels == null) {
            g2.setColor(Color.BLACK);
            g2.drawString("No data or predictions to display", 50, 200);
            return;
        }

        // Trouver les dimensions pour les axes
        double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE;
        double minY = Double.MAX_VALUE, maxY = -Double.MAX_VALUE;
        for (double[] point : donnee) {
            minX = Math.min(minX, point[0]);
            maxX = Math.max(maxX, point[0]);
            minY = Math.min(minY, point[1]);
            maxY = Math.max(maxY, point[1]);
        }
        double rangeX = maxX - minX;
        double rangeY = maxY - minY;
        if (rangeX == 0) rangeX = 1; // éviter la division par 0
        if (rangeY == 0) rangeY = 1;

        // marges et taille
        int margin = 30;
        int plotSize = Math.min(getWidth(), getHeight()) - 2 * margin;

        // Calcul de la densité pour les prédictions
        int gridSize = 20;
        int[][] class0Count = new int[gridSize][gridSize];
        int[][] class1Count = new int[gridSize][gridSize];
        
        // Compter les points dans chaque case
        for (int i = 0; i < donnee.length; i++) {
            double abscisse = donnee[i][0];
            double ordonnee = donnee[i][1];

            int gridX = (int) (((abscisse - minX) / rangeX) * gridSize);
            int gridY = (int) (((maxY - ordonnee) / rangeY) * gridSize);
            gridX = Math.min(Math.max(gridX, 0), gridSize - 1);
            gridY = Math.min(Math.max(gridY, 0), gridSize - 1);
            
            if (predictedLabels[i][0] == 1) {
                class1Count[gridX][gridY]++;
            } else {
                class0Count[gridX][gridY]++;
            }
        }

        // Dessiner le fond sur les décisions
        int cellWidth = plotSize / gridSize;
        int cellHeight = plotSize / gridSize;
        for (int gx = 0; gx < gridSize; gx++) {
            for (int gy = 0; gy < gridSize; gy++) {
                int total = class0Count[gx][gy] + class1Count[gx][gy];
                float ratio = total > 0 ? (float) class1Count[gx][gy] / total : 0.5f;
                Color color = interpolateColor(Color.CYAN, Color.PINK, ratio);
                g2.setColor(color);
                int x = margin + gx * cellWidth;
                int y = margin + gy * cellHeight;
                g2.fillRect(x, y, cellWidth, cellHeight);
            }
        }

        // Dessiner les points
        float pointSize = 6f; // taille d'un point
        for (int i = 0; i < donnee.length; i++) {
            double abscisse = donnee[i][0];
            double ordonnee = donnee[i][1];
            
            g2.setColor(label[i][0] == 1 ? Color.RED : Color.BLUE);

            int x = margin + (int) ((abscisse - minX) / rangeX * plotSize);
            int y = margin + (int) ((maxY - ordonnee) / rangeY * plotSize);

            g2.fillOval(x - (int)(pointSize/2), y - (int)(pointSize/2), (int)pointSize, (int)pointSize);
        }

        // Dessiner les axes
        g2.setColor(Color.BLACK);
        g2.drawLine(margin, margin + plotSize, margin + plotSize, margin + plotSize);
        g2.drawLine(margin, margin, margin, margin + plotSize);

        g2.setFont(new Font("Arial", Font.PLAIN, 12));
        g2.drawString(String.format("%.1f", minX), margin - 10, margin + plotSize + 15);
        g2.drawString(String.format("%.1f", maxX), margin + plotSize - 20, margin + plotSize + 15);
        g2.drawString(String.format("%.1f", minY), margin - 30, margin + plotSize);
        g2.drawString(String.format("%.1f", maxY), margin - 30, margin + 5);
    }

    // Methode pour faire l'interpolation de 2 couleurs
    private Color interpolateColor(Color start, Color end, float ratio) {
        float r = start.getRed() + ratio * (end.getRed() - start.getRed());
        float g = start.getGreen() + ratio * (end.getGreen() - start.getGreen());
        float b = start.getBlue() + ratio * (end.getBlue() - start.getBlue());
        return new Color(r / 255f, g / 255f, b / 255f);
    }
}