package Matrices;

import java.util.Vector;
import java.util.function.Function;

/**
 * Représente un vecteur d'activations des neurones de la couche précédente, qui sera envoyé aux neurones de la couche suivante.
 * Implémente (essaie plus ou moins) un design d'<a href="https://en.wikipedia.org/wiki/Fluent_interface#Immutability">Immutable fluent interface</a>.
 */

// TODO remplacer l'héritage Vector<Double> par List<Double, car Vector pas opti
public class ActivationVector extends Vector<Double> {

    public static ActivationVector of(Double ... doubles){
        ActivationVector res = new ActivationVector();
        for(Double d : doubles) res.add(d);
        return res;
    }
    /**
     * Soustrait un autre {@link ActivationVector} terme à terme au vecteur.
     * C'est une opération intermédiaire.
     * @param vector le vecteur de même dimension que this, qu'on soustrait
     * @return un nouveau {@link ActivationVector} qui correspond à la différence terme à terme.
     */
    public ActivationVector substract(ActivationVector vector){
        int size = this.size();
        assert(size == vector.size());
        ActivationVector res = new ActivationVector();
        for(int i = 0; i < size; i++){
            res.add(this.get(i) - vector.get(i));
        }
        return res;
    }

    /**
     * Calcule la somme des éléments du vecteur.
     * C'est une opération terminale.
     * @return La somme calculée.
     */
    public double sum(){
        return this.stream().mapToDouble(d -> d).sum();
    }

    /**
     * Renvoie un nouveau vecteur dont chaque composante est le résultat
     * de la fonction appliqué au composant du vecteur initial.
     * C'est une opération intermédiaire.
     * @param function la fonction à appliquer
     * @return le nouveau vecteur d'activation.
     */
    public ActivationVector applyFunction(Function<Double,Double> function){
        ActivationVector res = new ActivationVector();
        this.forEach(d -> res.add(function.apply(d)));
        return res;
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le logarithme népérien (log base e) de l'ancien;
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public ActivationVector log(){
        return this.applyFunction(Math::log);
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le cosinus hyperbolique de l'ancien;
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public ActivationVector cosh(){
        return this.applyFunction(Math::cosh);
    }

    /** Renvoie un nouveau vecteur dont chaque composante est
     * le carré de l'ancien;
     * C'est une opération intermédiaire.
     * @return Un nouveau vecteur d'activation dont les élements correspondent au cosh de l'ancien.
     */
    public ActivationVector square(){
        return this.applyFunction(d -> Math.pow(d,2));
    }
}
