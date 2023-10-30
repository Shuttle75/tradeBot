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
import java.math.BigDecimal;
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
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min5;


@RestController
public class KlinesController {
    /** Logger. */
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());

    private final Exchange exchange;
    private final MultiLayerNetwork model;

    public KlinesController(Exchange exchange, MultiLayerNetwork model) {
        this.exchange = exchange;
        this.model = model;
    }

    @GetMapping(path = "predict")
    public List<String> getPredict() throws IOException {
        Calendar startCalendar = Calendar.getInstance();
        startCalendar.add(Calendar.HOUR, -6);
        final Long startDate = startCalendar.getTimeInMillis() / 1000L;
        final Long endDate = Calendar.getInstance().getTimeInMillis() / 1000L;
        final List<KucoinKline> kucoinKlines;
        final CurrencyPair currencyPair = new CurrencyPair("BTC", "USDT");

        kucoinKlines = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(currencyPair, startDate, endDate, min5);

        float[][] floatData = new float[1][100];
        for (int y = 0; y < 20; y++) {
            floatData[0][y * 5] = kucoinKlines.get(y + 1).getOpen().floatValue() - kucoinKlines.get(0).getClose().floatValue();
            floatData[0][y * 5 + 1] = kucoinKlines.get(y + 1).getClose().floatValue() - kucoinKlines.get(0).getClose().floatValue();
            floatData[0][y * 5 + 2] = kucoinKlines.get(y + 1).getHigh().floatValue() - kucoinKlines.get(0).getClose().floatValue();
            floatData[0][y * 5 + 3] = kucoinKlines.get(y + 1).getLow().floatValue() - kucoinKlines.get(0).getClose().floatValue();
            floatData[0][y * 5 + 4] = kucoinKlines.get(y + 1).getVolume().floatValue();
        }

        float[] floatVector = model.output(Nd4j.create(floatData)).toFloatVector();
        List<String> listResult = new ArrayList<>();
        for (int i = 0; i < floatVector.length; i++) {
            listResult.add(String.format("%.2f", floatVector[i]));
        }

        listResult.add("Test - " + String.format("%.2f", kucoinKlines.get(0).getClose().floatValue() - kucoinKlines.get(1).getClose().floatValue()));
        listResult.add("0 - " + String.format("%.2f", kucoinKlines.get(0).getClose().floatValue()));
        listResult.add("2 - " + String.format("%.2f", kucoinKlines.get(1).getClose().floatValue()));
        return listResult;
    }


    @GetMapping(path = "klines")
    public List<KucoinKline> getTicker() {
        Calendar startCalendar = Calendar.getInstance();
        startCalendar.add(Calendar.HOUR, -1);
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

        float[][] floatData = new float[60][100];
        float[][] floatLabels = new float[60][9];
        for (int i = 0; i < 60; i++) {
            for (int y = 0; y < 20; y++) {
                floatData[i][y * 5] = kucoinKlines.get(y + i + 1).getOpen().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 5 + 1] = kucoinKlines.get(y + i + 1).getClose().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 5 + 2] = kucoinKlines.get(y + i + 1).getHigh().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 5 + 3] = kucoinKlines.get(y + i + 1).getLow().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 5 + 4] = kucoinKlines.get(y + i + 1).getVolume().floatValue();
            }

            int delta = (kucoinKlines.get(i).getClose().subtract(kucoinKlines.get(i + 1).getClose()).intValue() + 60) / 40;
            delta = delta < 0 ? 0 : delta;
            delta = delta > 8 ? 8 : delta;
            for (int j = 0; j < 9; j++) {
                floatLabels[i][j] = 0.0F;
            }
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
            .layer(new DenseLayer.Builder().nIn(100).nOut(20)
                       .build())
            .layer(new DenseLayer.Builder().nIn(20).nOut(3)
                       .build())
            .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                        .nIn(3).nOut(9).build())
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        for(int i = 0; i < 5000; i++ ) {
            model.fit(trainingData, trainingLabels);
        }


        return kucoinKlines;
    }
}


