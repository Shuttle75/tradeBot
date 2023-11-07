package com.trading.bot.controllers;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
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
import static com.trading.bot.util.TradeUtil.*;

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
        int goodMark = 0, wrongMark = 0;
        int goodSign = 0, wrongSign = 0;
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.DAYS)
                .minusDays(MINUS_DAYS + 2)
                .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.DAYS)
                .minusDays(MINUS_DAYS + 1)
                .toEpochSecond(ZoneOffset.UTC);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate);

        float[][] floatData = new float[TRAIN_CYCLES][TRAIN_DEEP];
        int[][] intLabels = new int[TRAIN_CYCLES][OUTPUT_SIZE];
        for (int i = 0; i < TRAIN_CYCLES; i++) {
            for (int y = 0; y < TRAIN_DEEP; y++) {
                floatData[i][y] = calcData(kucoinKlines, i, y, PREDICT_DEEP);
            }

            intLabels[i][getDelta(kucoinKlines, i)] = 1;
        }

        INDArray indData = Nd4j.create(floatData);

        float[][] floatResult = model.output(indData).toFloatMatrix();

        List<String> listResult = new ArrayList<>();
        for (int i = 0; i < TRAIN_CYCLES; i++) {
            if (floatResult[i][0] + floatResult[i][1] + floatResult[i][2] + floatResult[i][3] > 0.5) {
                if (intLabels[i][0] + intLabels[i][1] + intLabels[i][2] + intLabels[i][3] > 0.5) {
                    goodSign++;
                } else {
                    wrongSign++;
                }
            }
            if (floatResult[i][4] + floatResult[i][5] + floatResult[i][6] + floatResult[i][7] > 0.5) {
                if (intLabels[i][4] + intLabels[i][5] + intLabels[i][6] + intLabels[i][7] > 0.5) {
                    goodSign++;
                } else {
                    wrongSign++;
                }
            }

            if (floatResult[i][0] + floatResult[i][1] > 0.8) {
                if (intLabels[i][0] + intLabels[i][1] > 0.8) {
                    goodMark++;
                } else {
                    wrongMark++;
                }
            }
            if (floatResult[i][6] + floatResult[i][7] > 0.8) {
                if (intLabels[i][6] + intLabels[i][7] > 0.8) {
                    goodMark++;
                } else {
                    wrongMark++;
                }
            }

            if (floatResult[i][0] > 0.8 || floatResult[i][7] > 0.8) {
                listResult.add(intLabels[i][0] + "    " +
                        intLabels[i][1] + "    " +
                        intLabels[i][2] + "    " +
                        intLabels[i][3] + "  |   " +
                        intLabels[i][4] + "    " +
                        intLabels[i][5] + "    " +
                        intLabels[i][6] + "    " +
                        intLabels[i][7] + "   ");
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

        listResult.add(" GoodSign = " + goodSign + ", WrongSign = " + wrongSign);
        listResult.add(" GoodMark = " + goodMark + ", WrongMark = " + wrongMark + ", K = " + 100.0 / (goodMark + wrongMark) * goodMark);

        return listResult;
    }
}


