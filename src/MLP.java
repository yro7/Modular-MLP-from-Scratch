import java.util.List;

public class MLP {

    private final int dimInput;
    private final List<Layer> layers;

    public MLP(List<Layer> layers, int dimInput){
        this.layers = layers;
        this.dimInput = dimInput;
    }

    public static MLPBuilder builder(int dimInput){
        return new MLPBuilder(dimInput);
    }


    public void print(){

        int n = layers.size();
        for(int i = 0; i < n; i ++){
            System.out.println("Layer n°" + i + " : ");
            layers.get(i).print();
            System.out.println();
        }
    }


}
