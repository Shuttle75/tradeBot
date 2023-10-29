package com.trading.bot.controllers;

import com.trading.bot.dto.market.TickerDTO;
import com.trading.bot.dto.util.CurrencyPairDTO;
import com.trading.bot.util.base.Base;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.knowm.xchange.service.marketdata.params.CurrencyPairsParam;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min15;


@RestController
public class KlinesController {
    /** Logger. */
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());

    private final Exchange exchange;

    public KlinesController(Exchange exchange) {
        this.exchange = exchange;
    }

    @GetMapping(path = "klines")
    public List<KucoinKline> getTicker() {
        Calendar startCalendar = Calendar.getInstance();
        startCalendar.add(Calendar.DATE, -3);
        final Long startDate = startCalendar.getTimeInMillis() / 1000L;
        final Long endDate = Calendar.getInstance().getTimeInMillis() / 1000L;
        final List<KucoinKline> kucoinKlines;

        try {
            kucoinKlines = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(
                    new CurrencyPair("BTC", "USDT"),
                    startDate,
                    endDate, min15);
        } catch (IOException e) {
            logger.error("Error retrieving tickers: {}", e.getMessage());
            return Collections.emptyList();
        }

        float[][] floats = new float[60][300];
        for (int i = 0; i < 60; i++) {
            for (int y = 0; y < 60; y++) {
                floats[i][y * 5] = kucoinKlines.get(y + i).getOpen().floatValue();
                floats[i][y * 5 + 1] = kucoinKlines.get(y + i).getClose().floatValue();
                floats[i][y * 5 + 2] = kucoinKlines.get(y + i).getHigh().floatValue();
                floats[i][y * 5 + 3] = kucoinKlines.get(y + i).getLow().floatValue();
                floats[i][y * 5 + 4] = kucoinKlines.get(y + i).getVolume().floatValue();
            }
        }
        INDArray trainingData = Nd4j.create(floats);


        logger.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(6)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(300).nOut(3)
                       .build())
            .layer(new DenseLayer.Builder().nIn(3).nOut(3)
                       .build())
            .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                        .nIn(3).nOut(5).build())
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        for(int i=0; i<1000; i++ ) {
            model.fit(trainingData, trainingData);
        }


        return kucoinKlines;
    }
}
