package mlps.MNIST;


import mlps.Data.Loaders.Dataloader;
import mlps.MLP;
import mlps.Optimizers.SGD;
import mlps.Trainer.Evaluation;
import matrices.ActivationMatrix;
import mlps.Trainer.Trainer;
import mlps.Classification;
import matrices.Utils;

import java.util.List;

import static functions.LossFunction.*;
import static mlps.MLP.FeedForwardResult;

public class Instancier {

    public static void main(String[] args)  {
        // Chemins vers les fichiers MNIST
        // Génère un ensemble de données d'entraînement Mnist
        MnistDataset mnistDataset = new MnistDataset();



        // Construction du trainer
        Trainer mnistTrainer = Trainer.builder()
                .setLossFunction(CE)
                .setOptimizer(new SGD(0.001))
                .setDataset(mnistDataset)
                .setEpoch(1)
                .setParameterRegularization(null)
             //   .setParameterRegularization(new ElasticNet(1e-4, 1e-3))
                .setBatchSize(1)
                .build();

      //  mnistDataset.trainDataLoader.getBatch(0).getInput().print();
       // mnistDataset.trainDataLoader.getBatch(0).getOutput().print();
/**
        MLP mnistMLP = MLP.builder(784)
                .setRandomSeed(420)
                .addLayer(256, RefLU)
                .addLayer(128, ReLU)
                addLayer(10, SoftMax)
                .build()
                .train(mnistTrainer);**/

        MLP mnistMLP = MLP.importModel("mnist_resolver");

       // mnistMLP.serialize("mnistTest_1epoch_ADAM");


      //  MLP mnistMLP = MLP.importModel("mnistTest");
        Evaluation eval = mnistTrainer.evaluate(mnistMLP);
        eval.print();

        // récupère les 100 premiers exemples

        List<Dataloader.LabeledDataSample<MnistImage, Classification>> t = mnistDataset.trainDataLoader.loadList(1,100);

        for(int i = 0; i < 99; i++){
            ActivationMatrix batchCaca = mnistDataset.trainDataLoader.objects_To_Batch(List.of(t.get(i).getInput()));
            batchCaca.printDimensions("dzdz");
            FeedForwardResult result = mnistMLP.feedForward(batchCaca);

            int pred = Utils.argmax(result.getNetworkOutput().getData()[0]);
            int trueLabel = Utils.argmax(t.get(i).getOutput().getHotEncoding());

            System.out.println(pred + " / " + trueLabel + " (pred, truelabel");

        }

        /**
         * COMPARAISON ADAM ET SGD :
         *
         * Adam :
         *
         * Accuracy : 87.878 %
         * Precision : 0.3939
         * Recall : 0.3939
         * f1Score : 0.39390000000000003
         *
         * Temps d'entraînement : 59 secondes.
         *
         * ******************************
         *
         * SGD :
         *
         * Accuracy : 82.056 %
         * 
         * Precision : 0.1028
         * Recall : 0.1028
         * f1Score : 0.1028
         * Temps d'entraînement : 56 secondes.
         */

    }


    public static void printLoss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique){
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, CE));
    }

    public static int maxIndiceOfArray(double[] array){
        int res = 0;
        double max = array[0];
        for(int i = 0; i < array.length; i ++){
            if(max < array[i]) {
                res = i;
                max = array[i];
            }
        }

        return res;
    }

}