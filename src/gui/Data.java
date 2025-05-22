package gui;

public class Data {
    
    private double[][] data;
    private double[][] result;

    public Data(){
        double[][] xorData = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        double[][] xorResult = {
            {0},
            {1},
            {1},
            {0},
        };
        this.data = xorData;
        this.result = xorResult;
    } 

    public double[][] getData(){
        return this.data;
    } 

    public double[][] getResult(){
        return this.result;
    } 

    public void setData(double[][] data){
        this.data = data;
    } 

    public void setResult(double[][] result){
        this.result = result;
    } 
}
