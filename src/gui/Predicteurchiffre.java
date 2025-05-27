package gui;
import javax.swing.*;

import matrices.ActivationMatrix;
import mlps.MLP;
import mlps.MLP.FeedForwardResult;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

public class Predicteurchiffre extends JPanel {
 private static final int SIZE = 28;
 private static final int SCALE = 10; // Pour agrandir l'affichage
 private BufferedImage image;
 private JPanel frame;

 public Predicteurchiffre() {
 this.setPreferredSize(new Dimension(SIZE * SCALE, SIZE * SCALE));
 image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_BYTE_GRAY);

 this.frame = new JPanel();

 JLabel resultLabel = new JLabel("Dessinez et appuyer sur \"Afficher matrice\"");
 
 MLP mnist = MLP.importModel("src/gui/mnistTest2_ADAM");

 JButton btnClear = new JButton("Effacer");
 btnClear.addActionListener(e -> this.clear());

 JButton btnGetData = new JButton("Afficher matrice");
 btnGetData.addActionListener(e -> {
 double[][] data = this.getImageData();
 double[] vectorized = vectorize(data);

 
 ActivationMatrix matrix = new ActivationMatrix(vectorized);
 
 FeedForwardResult result = mnist.feedForward(matrix);
    int resultEncoded = indiceMax(result.getNetworkOutput().getData()[0]);
    System.out.println(resultEncoded);
    resultLabel.setText("" + resultEncoded);

 });

 JPanel controls = new JPanel();
 controls.add(btnClear);
 controls.add(btnGetData);

 frame.setLayout(new BorderLayout());
 frame.add(this, BorderLayout.CENTER);
 frame.add(controls, BorderLayout.SOUTH);
 frame.add(resultLabel, BorderLayout.EAST);

 this.clear(); // Initialiser l'image en blanc

 // Dessin avec la souris
 this.addMouseMotionListener(new MouseMotionAdapter() {
 public void mouseDragged(MouseEvent e) {
 int x = e.getX() / SCALE;
 int y = e.getY() / SCALE;
 if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
 Graphics2D g2d = image.createGraphics();
 g2d.setColor(Color.BLACK);
 g2d.fillOval(x - 1, y - 1, 3, 3); // Épaisseur du trait
 g2d.dispose();
 repaint();
 }
 }
 });
 }

 @Override
 protected void paintComponent(Graphics g) {
 super.paintComponent(g);
 // Agrandir l'image 28x28 pour l'affichage
 g.drawImage(image, 0, 0, SIZE * SCALE, SIZE * SCALE, null);
 }

 // Récupérer l'image comme matrice normalisée [0.0 - 1.0]
 public double[][] getImageData() {
 double[][] data = new double[SIZE][SIZE];
 for (int y = 0; y < SIZE; y++) {
 for (int x = 0; x < SIZE; x++) {
 int color = image.getRGB(x, y) & 0xFF; // Niveau de gris (0=black, 255=white)
 data[y][x] = 1.0 - (color / 255.0); // Inversé pour MNIST (noir = 1.0)
 }
 }
 return data;
 }

 public void clear() {
 Graphics2D g2d = image.createGraphics();
 g2d.setColor(Color.WHITE);
 g2d.fillRect(0, 0, SIZE, SIZE);
 g2d.dispose();
 repaint();
 }

 public static double[] vectorize(double[][] matrix) {
 int height = matrix.length;
 int width = matrix[0].length;
 double[] vector = new double[height * width];
 
 for (int y = 0; y < height; y++) {
 for (int x = 0; x < width; x++) {
 vector[y * width + x] = matrix[y][x];
 }
 }
 
 return vector;
 }

 
 public static int indiceMax(double[] tableau) {
    if (tableau == null || tableau.length == 0) {
        throw new IllegalArgumentException("Le tableau ne peut pas être nul ou vide.");
    }

    int indiceMax = 0;
    double max = tableau[0];

    for (int i = 1; i < tableau.length; i++) {
        if (tableau[i] > max) {
            max = tableau[i];
            indiceMax = i;
        }
    }

    return indiceMax;
}
 public JPanel getFrame(){
    return this.frame;
 } 
}

