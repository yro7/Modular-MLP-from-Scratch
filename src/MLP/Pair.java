package MLP;

public class Pair<A,B> {

    A objectA;
    B objectB;

    public A getA(){
        return objectA;
    }

    public B getB(){
        return objectB;
    }

    public Pair(A objectA, B objectB) {
        this.objectA = objectA;
        this.objectB = objectB;
    }
}
