package com.trading.bot.controllers;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

import static com.trading.bot.configuration.BotConfig.*;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min1;


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
        startCalendar.add(Calendar.DATE, -8);
        final Long startDate = startCalendar.getTimeInMillis() / 1000L;
        final Long endDate = Calendar.getInstance().getTimeInMillis() / 1000L;
        final CurrencyPair currencyPair = new CurrencyPair("BTC", "USDT");

        final List<KucoinKline> kucoinKlines = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(currencyPair, startDate, endDate, min1);

        float[][] floatData = new float[TRAIN_CYCLES][TRAIN_DEEP * 4];
        float[][] floatLabels = new float[TRAIN_CYCLES][OUTPUT_SIZE];
        for (int i = 0; i < TRAIN_CYCLES; i++) {
            for (int y = 0; y < TRAIN_DEEP; y++) {
                floatData[i][y * 4] = kucoinKlines.get(y + i + PREDICT_DEEP).getClose().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 4 + 1] = kucoinKlines.get(y + i + PREDICT_DEEP).getHigh().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 4 + 2] = kucoinKlines.get(y + i + PREDICT_DEEP).getLow().floatValue() - kucoinKlines.get(i).getClose().floatValue();
                floatData[i][y * 4 + 3] = kucoinKlines.get(y + i + PREDICT_DEEP).getVolume().floatValue();
            }

            int delta = (kucoinKlines.get(i).getClose().subtract(kucoinKlines.get(i + PREDICT_DEEP).getClose()).intValue() + 80) / 20;
            delta = Math.max(delta, 0);
            delta = Math.min(delta, 7);
            floatLabels[i][delta] = 1.0F;
        }

        INDArray trainingData = Nd4j.create(floatData);
        INDArray trainingLabels = Nd4j.create(floatLabels);

        float[][] floatResult = model.output(trainingData).toFloatMatrix();

        List<String> listResult = new ArrayList<>();
        for (int i = 0; i < TRAIN_CYCLES; i++) {
            if ((floatResult[i][0] + floatResult[i][1]) > 0.8 ||  (floatResult[i][6] + floatResult[i][7]) > 0.8) {
                listResult.add(String.format("%.2f", floatLabels[i][0]) + ", " +
                        String.format("%.2f", floatLabels[i][1]) + ", " +
                        String.format("%.2f", floatLabels[i][2]) + ", " +
                        String.format("%.2f", floatLabels[i][3]) + ", " +
                        String.format("%.2f", floatLabels[i][4]) + ", " +
                        String.format("%.2f", floatLabels[i][5]) + " ," +
                        String.format("%.2f", floatLabels[i][6]) + ", " +
                        String.format("%.2f", floatLabels[i][7]));
                listResult.add(String.format("%.2f", floatResult[i][0]) + ", " +
                        String.format("%.2f", floatResult[i][1]) + ", " +
                        String.format("%.2f", floatResult[i][2]) + ", " +
                        String.format("%.2f", floatResult[i][3]) + ", " +
                        String.format("%.2f", floatResult[i][4]) + ", " +
                        String.format("%.2f", floatResult[i][5]) + " ," +
                        String.format("%.2f", floatResult[i][6]) + ", " +
                        String.format("%.2f", floatResult[i][7]));
                listResult.add("---------------------------------------");
            }
        }
        return listResult;
    }
}


