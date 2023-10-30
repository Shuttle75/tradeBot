package com.trading.bot.configuration;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.ExchangeFactory;
import org.knowm.xchange.ExchangeSpecification;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.kucoin.KucoinExchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.util.Calendar;
import java.util.List;

import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min1;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());

    private final int trainCycles = 1440;
    private final int dataSize = 40;
    private final int outputSize = 8;

    @Bean
    public Exchange getXChangeExchange() {
        ExchangeSpecification exchangeSpecification = new ExchangeSpecification(KucoinExchange.class);

        exchangeSpecification.setUserName("alexey.osadchiy@gmail.com");
        exchangeSpecification.setApiKey("65380094ab6f110001a5a383");
        exchangeSpecification.setSecretKey("58da9817-38b5-4f89-85cd-718e54edd8fe");
        exchangeSpecification.setExchangeSpecificParametersItem("Use_Sandbox", false);
        exchangeSpecification.setExchangeSpecificParametersItem("passphrase", "cassandre");

        return ExchangeFactory.INSTANCE.createExchange(exchangeSpecification);
    }

    @Bean
    public MultiLayerConfiguration getMultiLayerConfiguration() {
        logger.info("Build model....");
        return new NeuralNetConfiguration.Builder()
                .seed(6)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(dataSize).nOut(320)
                        .build())
                .layer(new DenseLayer.Builder().nIn(320).nOut(160)
                        .build())
                .layer(new DenseLayer.Builder().nIn(160).nOut(40)
                        .build())
                .layer(new DenseLayer.Builder().nIn(40).nOut(16)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                        .nIn(16).nOut(outputSize).build())
                .build();
    }

    @Bean
    public MultiLayerNetwork getModel(Exchange exchange, MultiLayerConfiguration multiLayerConfiguration) throws IOException {
        Calendar startCalendar = Calendar.getInstance();
        startCalendar.add(Calendar.DATE, -3);
        final Long startDate = startCalendar.getTimeInMillis() / 1000L;
        Calendar endCalendar = Calendar.getInstance();
        endCalendar.add(Calendar.DATE, -1);
        final Long endDate = endCalendar.getTimeInMillis() / 1000L;
        final List<KucoinKline> kucoinKlines;
        final CurrencyPair currencyPair = new CurrencyPair("BTC", "USDT");

        kucoinKlines = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(currencyPair, startDate, endDate, min1);

        float[][] floatData = new float[trainCycles][dataSize];
        float[][] floatLabels = new float[trainCycles][outputSize];
        for (int i = 0; i < trainCycles; i++) {
            for (int y = 0; y < 10; y++) {
                floatData[i][y * 4] = kucoinKlines.get(y + i + 1).getClose().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 4 + 1] = kucoinKlines.get(y + i + 1).getHigh().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 4 + 2] = kucoinKlines.get(y + i + 1).getLow().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 4 + 3] = kucoinKlines.get(y + i + 1).getVolume().floatValue();
            }

            int delta = (kucoinKlines.get(i).getClose().subtract(kucoinKlines.get(i + 1).getClose()).intValue() + 40) / 10;
            delta = Math.max(delta, 0);
            delta = Math.min(delta, 7);
            floatLabels[i][delta] = 1.0F;
        }
        INDArray trainingData = Nd4j.create(floatData);
        INDArray trainingLabels = Nd4j.create(floatLabels);

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(multiLayerConfiguration);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        for(int i = 0; i < 2000; i++ ) {
            model.fit(trainingData, trainingLabels);
        }
        return model;
    }
}
