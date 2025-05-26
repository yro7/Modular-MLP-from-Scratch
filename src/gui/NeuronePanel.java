package gui;

import java.awt.*;
import java.util.List;
import java.util.ArrayList;
import java.awt.Point;
import javax.swing.*;

public class NeuronePanel extends JPanel {
    private List<Point> connections = new ArrayList<>();
    
    public void addConnection(Point target) {
        connections.add(target);
    }
    
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        
        // Dessiner les connexions
        g2d.setColor(Color.GRAY);
        for (Point target : connections) {
            g2d.drawLine(getWidth()/2, getHeight()/2, target.x, target.y);
        }
        
        // Dessiner le neurone
        g2d.setColor(Color.BLUE);
        g2d.fillOval(getWidth()/2 - 15, getHeight()/2 - 15, 30, 30);
    }
}