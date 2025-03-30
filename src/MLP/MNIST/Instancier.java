package MLP.MNIST;


import MLP.Data.LabeledDataset;
import MLP.Data.Loaders.Dataloader;
import MLP.MLP;
import Matrices.ActivationMatrix;
import MLP.Trainer;
import MLP.Optimizer;
import MLP.Classification;
import Matrices.Utils;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static Function.ActivationFunction.*;
import static Function.LossFunction.*;
import static MLP.MNIST.MnistDataset.*;
import static MLP.Trainer.Evaluation;
import static MLP.MLP.FeedForwardResult;

public class Instancier {

    public static void main(String[] args)  {
        // Chemins vers les fichiers MNIST
        // Génère un ensemble de données d'entraînement Mnist
        MnistDataset mnistDataset = new MnistDataset();


        // Construction du trainer
        Trainer mnistTrainer = Trainer.builder()
                .setLossFunction(CE)
                .setOptimizer(new Optimizer(1))
                .setDataset(mnistDataset)
                .setEpoch(3)
                .setBatchSize(10_000)
                .build();

        MLP mnistMLP = MLP.builder(784)
                .setRandomSeed(420)
                .addLayer(256, ReLU)
                .addLayer(128, ReLU)
                .addLayer(10, SoftMax)
                .build()
                .train(mnistTrainer);

        mnistMLP.serialize("mnistTest_3epoch");

      //  MLP mnistMLP = MLP.importModel("mnistTest");
        Evaluation eval = mnistTrainer.evaluate(mnistMLP);
        eval.print();

        // récupère les 100 premiers exemples
        List<Dataloader.LabeledDataSample<MnistImage, Classification>> t = mnistDataset.trainDataLoader.loadList(1,100);

        for(int i = 0; i < 99; i++){
            ActivationMatrix batchCaca = mnistDataset.trainDataLoader.objects_To_Batch(List.of(t.get(i).getInput()));
            batchCaca.printDimensions("caca");
            FeedForwardResult result = mnistMLP.feedForward(batchCaca);

            int pred = Utils.argmax(result.getNetworkOutput().getData()[0]);
            int trueLabel = Utils.argmax(t.get(i).getOutput().getHotEncoding());

            System.out.println(pred + " / " + trueLabel + " (pred, truelabel");

        }


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