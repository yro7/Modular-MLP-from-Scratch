import mlps.Data.LabeledDataset;
import mlps.Data.Loaders.Dataloader;
import mlps.MLP;
import mlps.Optimizers.Adam;
import matrices.*;
import mlps.Optimizers.Optimizer;
import mlps.Optimizers.SGD;
import mlps.Trainer.Trainer;

import java.util.IntSummaryStatistics;
import java.util.stream.IntStream;

import static functions.ActivationFunction.*;
import static functions.LossFunction.*;


// TODO ROADMAP :
/**
 * Implémenter ADAM & autres optimizers
 * Implémenter régularization
 * Implémenter dropout
 */


public class Main {

    public static void main(String[] args) {


        MLP mlp = MLP.builder(2)
                .setRandomSeed(3)
                .addLayer(4, ReLU)
                .addLayer(1, Sigmoid)
                .build();


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


        ActivationMatrix batchInput = new ActivationMatrix(xorData);
        ActivationMatrix batchTheorique = new ActivationMatrix(xorResult);
        Adam optimizer = new Adam();

        mlp.feedForward(batchInput).getNetworkOutput().print();

        Optimizer adam = new Adam(0.4, 0.9, 0.999);
        //0.001, 0.9, 0.999
        Optimizer sgd = new SGD(0.001);
        for(int i = 0; i < 10_000; i++){
            mlp.updateParameters(batchInput, batchTheorique, MSE, adam, null);
           // printLoss(mlp, batchInput, batchTheorique);
        };


        mlp.feedForward(batchInput).getNetworkOutput().print();


    }

        public static void printLoss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique){
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, MSE));
    }

    public static void save(){

        MLP mlp = MLP.builder(2)
                .setRandomSeed(3)
                .addLayer(4, ReLU)
                .addLayer(1, Sigmoid)
                .addLayer(1, Sigmoid)
                .build();


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


        ActivationMatrix batchInput = new ActivationMatrix(xorData);
        ActivationMatrix batchTheorique = new ActivationMatrix(xorResult);
        Adam optimizer = new Adam();

        Dataloader loader = new Dataloader(4, 2, 1, null, null, 4) {
            @Override
            public double[] vectorizeInput(Object input) {
                return (double[]) input;
            }

            @Override
            public double[] vectorizeOutput(Object input) {
                return (double[]) input;
            }

            @Override
            public LabeledDataSample load(int i) {
                return new LabeledDataSample(xorData[i], xorResult[i]);
            }
        };


        LabeledDataset dataset = new LabeledDataset(loader, loader);

        // Construction du trainer
        Trainer trainer = Trainer.builder()
                .setLossFunction(BCE)
                .setOptimizer(new Adam(0.1,0.9,0.999))
                .setDataset(dataset)
                .setEpoch(10)
                .setParameterRegularization(null)
                //   .setParameterRegularization(new ElasticNet(1e-4, 1e-3))
                .setBatchSize(4)
                .build();

        /**
         *
         * EN 10_000 EPOCHS :
         *
         * Avec Adam (0.1, 0.9, 0.999) :
         *
         * 0.0012854337670645363,
         * 0.9930651281264739,
         * 0.9988654605782511,
         * 0.00202564798675737,
         *
         * Avec SGD(0.1) :
         *
         * 6.419657276914641E-4,
         * 0.9945661655430321,
         * 0.9995142178198154,
         * 8.508416840000762E-4,
         *
         * EN 10 EPOCHS :
         *
         * ADAM :
         *
         * 0.49691084052990014,
         * 0.49348979694177214,
         * 0.5552906321798,
         * 0.48333864357268713,
         *
         * SGD :
         *
         * 0.4936722930702566,
         * 0.4925392124255475,
         * 0.5665708058080177,
         * 0.4822245652481131,
         */

        trainer.train(mlp);

        mlp.feedForward(batchInput).getNetworkOutput().print();


    }

}