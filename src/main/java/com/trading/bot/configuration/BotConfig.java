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

import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min15;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min5;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());

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
    public MultiLayerNetwork getModel(Exchange exchange) throws IOException {
        Calendar startCalendar = Calendar.getInstance();
        startCalendar.add(Calendar.DATE, -3);
        final Long startDate = startCalendar.getTimeInMillis() / 1000L;
        final Long endDate = Calendar.getInstance().getTimeInMillis() / 1000L;
        final List<KucoinKline> kucoinKlines;
        final CurrencyPair currencyPair = new CurrencyPair("BTC", "USDT");

        kucoinKlines = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(currencyPair, startDate, endDate, min5);

        float[][] floatData = new float[200][100];
        float[][] floatLabels = new float[200][9];
        for (int i = 0; i < 200; i++) {
            for (int y = 0; y < 20; y++) {
                floatData[i][y * 5] = kucoinKlines.get(y + i + 1).getOpen().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 5 + 1] = kucoinKlines.get(y + i + 1).getClose().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 5 + 2] = kucoinKlines.get(y + i + 1).getHigh().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 5 + 3] = kucoinKlines.get(y + i + 1).getLow().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 5 + 4] = kucoinKlines.get(y + i + 1).getVolume().floatValue();
            }

            int delta = (kucoinKlines.get(i).getClose().subtract(kucoinKlines.get(i + 1).getClose()).intValue() + 80) / 20;
            delta = delta < 0 ? 0 : delta;
            delta = delta > 8 ? 8 : delta;
            floatLabels[i][delta] = 1.0F;
        }
        INDArray trainingData = Nd4j.create(floatData);
        INDArray trainingLabels = Nd4j.create(floatLabels);

        logger.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(6)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(100).nOut(100)
                        .build())
                .layer(new DenseLayer.Builder().nIn(100).nOut(20)
                        .build())
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                        .nIn(20).nOut(9).build())
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        for(int i = 0; i < 2000; i++ ) {
            model.fit(trainingData, trainingLabels);
        }
        return model;
    }
}
