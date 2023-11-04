package com.trading.bot.controllers;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;

import static com.trading.bot.configuration.BotConfig.*;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min1;


@RestController
public class KlinesController {
    /** Logger. */
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    private final Exchange exchange;
    private final MultiLayerNetwork model;

    private int goodMark, wrongMark;

    public KlinesController(Exchange exchange, MultiLayerNetwork model) {
        this.exchange = exchange;
        this.model = model;
    }

    @GetMapping(path = "predict")
    public List<String> getPredict() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.MINUTES)
                .minusDays(3)
                .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.MINUTES)
                .minusDays(1)
                .toEpochSecond(ZoneOffset.UTC);

        final List<KucoinKline> kucoinKlines = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, min1);

        float[][] floatData = new float[TRAIN_CYCLES][TRAIN_DEEP];
        float[][] floatLabels = new float[TRAIN_CYCLES][OUTPUT_SIZE];
        for (int i = 0; i < TRAIN_CYCLES; i++) {
            for (int y = 0; y < TRAIN_DEEP; y++) {
                floatData[i][y] = calcNumbers(kucoinKlines, i, y, 0);
            }

            int delta = (kucoinKlines.get(i).getClose()
                    .subtract(kucoinKlines.get(i + PREDICT_DEEP).getClose())
                    .intValue() + OUTPUT_SIZE * CURRENCY_DELTA / 2) / CURRENCY_DELTA;

            delta = Math.max(delta, 0);
            delta = Math.min(delta, 7);
            floatLabels[i][delta] = 1;
        }

        INDArray trainingData = Nd4j.create(floatData);
        INDArray trainingLabels = Nd4j.create(floatLabels);

        float[][] floatResult = model.output(trainingData).toFloatMatrix();

        List<String> listResult = new ArrayList<>();
        for (int i = 0; i < TRAIN_CYCLES; i++) {
            if ((floatResult[i][0] + floatResult[i][1]) > 0.8 ||  floatLabels[i][0] == 1.0) {
                goodMark++;
            } else {
                wrongMark++;
            }
            if ((floatResult[i][6] + floatResult[i][7]) > 0.8 ||  floatLabels[i][7] == 1.0) {
                goodMark++;
            } else {
                wrongMark++;
            }

            if ((floatResult[i][0] + floatResult[i][1]) > 0.8 ||  (floatResult[i][6] + floatResult[i][7]) > 0.8) {
                listResult.add(String.format("%.2f", floatLabels[i][0]) + " " +
                        String.format("%.2f", floatLabels[i][1]) + " " +
                        String.format("%.2f", floatLabels[i][2]) + " " +
                        String.format("%.2f", floatLabels[i][3]) + " | " +
                        String.format("%.2f", floatLabels[i][4]) + " " +
                        String.format("%.2f", floatLabels[i][5]) + " " +
                        String.format("%.2f", floatLabels[i][6]) + " " +
                        String.format("%.2f", floatLabels[i][7]));
                listResult.add(String.format("%.2f", floatResult[i][0]) + " " +
                        String.format("%.2f", floatResult[i][1]) + " " +
                        String.format("%.2f", floatResult[i][2]) + " " +
                        String.format("%.2f", floatResult[i][3]) + " | " +
                        String.format("%.2f", floatResult[i][4]) + " " +
                        String.format("%.2f", floatResult[i][5]) + " " +
                        String.format("%.2f", floatResult[i][6]) + " " +
                        String.format("%.2f", floatResult[i][7]));
                listResult.add("-----------------------------------------");
            }
        }

        listResult.add(" GoodMark = " + goodMark + ", WrongMark = " + wrongMark);

        return listResult;
    }
}


