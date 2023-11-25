package com.trading.bot.controllers;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.configuration.BotConfig.*;
import static com.trading.bot.util.TradeUtil.*;

@RestController
public class KlinesController {
    /** Logger. */
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    private final Exchange exchange;
    private final MultiLayerNetwork net;

    public KlinesController(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

    @GetMapping(path = "predict")
    public List<String> getPredict() throws IOException {
        int goodMark = 0;
        int wrongMark = 0;
        int goodSign = 0;
        int wrongSign = 0;
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.HOURS)
                .minusDays(1)
                .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.HOURS)
                .toEpochSecond(ZoneOffset.UTC);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
        Collections.reverse(kucoinKlines);

        net.rnnClearPreviousState();

        List<String> listResult = new ArrayList<>();
        for (int i = 0; i < kucoinKlines.size() - PREDICT_DEEP; i++) {
            float[] floatResult = getOneMinutePredict(kucoinKlines.get(i), net);
            int[] intLabels = new int[OUTPUT_SIZE];
            intLabels[getDelta(kucoinKlines, i)] = 1;
            if (floatResult[0] > 0.4) {
                if (intLabels[0] > 0.4) {
                    goodSign++;
                } else {
                    wrongSign++;
                }
            }
            if (floatResult[2] > 0.4) {
                if (intLabels[2] > 0.4) {
                    goodSign++;
                } else {
                    wrongSign++;
                }
            }

            if (floatResult[0] > 0.8) {
                if (intLabels[0] > 0.8) {
                    goodMark++;
                } else {
                    wrongMark++;
                }
            }
            if (floatResult[2] > 0.8) {
                if (intLabels[2] > 0.8) {
                    goodMark++;
                } else {
                    wrongMark++;
                }
            }

            if (intLabels[0] == 1 || intLabels[2] == 1) {
                listResult.add(intLabels[0] + "    " +
                        intLabels[1] + "    " +
                        intLabels[2] + "    ");
                listResult.add(String.format("%.2f", floatResult[0]) + " " +
                        String.format("%.2f", floatResult[1]) + " " +
                        String.format("%.2f", floatResult[2]));
                listResult.add("-----------------------------------------");
            }
        }

        listResult.add(" GoodSign = " + goodSign + ", WrongSign = " + wrongSign + ", K = " + goodSign / (float)(goodSign + wrongSign));
        listResult.add(" GoodMark = " + goodMark + ", WrongMark = " + wrongMark + ", K = " + goodMark / (float)(goodMark + wrongMark));

        return listResult;
    }
}


