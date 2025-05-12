package mlps.Trainer;


public class Evaluation {

    double TP;
    double FP;
    double TN;
    double FN;

    int classificationSize;

    public Evaluation(double TP, double FP, double TN, double FN, int classificationSize) {
        this.TP = TP;
        this.FP = FP;
        this.TN = TN;
        this.FN = FN;
    }

    public void print(){

        double size = TP + TN + FP + FN;

        double accuracy = (TP + TN) /(size);
        double precision = TP /(TP + FP);
        double recall = TP /(TP + FN);
        double f1Score = 2*(precision*recall)/(precision+recall);


        System.out.println("Accuracy : " + accuracy);
        System.out.println("Precision : " + precision);
        System.out.println("Recall : " + recall);
        System.out.println("f1Score : " + f1Score);
        System.out.println();

        ;
        System.out.println("Nombre de prédictions correctes : " + this.TP + "/" + size/classificationSize);
    }
}