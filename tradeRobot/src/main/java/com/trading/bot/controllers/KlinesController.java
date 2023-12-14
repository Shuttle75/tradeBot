package com.trading.bot.controllers;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.configuration.BotConfig.*;
import static com.trading.bot.util.TradeUtil.*;

@RestController
public class KlinesController {
    /** Logger. */
    private final Exchange exchange;
//    private final MultiLayerNetwork net;

    public KlinesController(Exchange exchange) {
        this.exchange = exchange;
//        this.net = net;
    }

    @GetMapping(path = "predict")
    public List<String> checkPredict() throws IOException {
        int goodMark = 0;
        int wrongMark = 0;
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
            .truncatedTo(ChronoUnit.DAYS)
            .minusDays(6)
            .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
            .truncatedTo(ChronoUnit.DAYS)
            .minusDays(1)
            .toEpochSecond(ZoneOffset.UTC);
        final double avgCandle = getAvgCandle(exchange);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
        Collections.reverse(kucoinKlines);

//        net.rnnClearPreviousState();
//        for (int i = 0; i < 96; i++) {
//            getPredict(kucoinKlines.get(i), net);
//        }

        List<String> listResult = new ArrayList<>();
        for (int i = 96; i < kucoinKlines.size() - PREDICT_DEEP; i++) {
//            float[] floatResult = getPredict(kucoinKlines.get(i), net);
            int[] intLabels = new int[OUTPUT_SIZE];
            intLabels[getDelta(kucoinKlines, i)] = 1;

            if (intLabels[0] == 1) {
//                if (floatResult[0] > 0.7) {
//                    goodMark++;
//                } else {
//                    wrongMark++;
//                }
            }
            if (intLabels[2] == 1) {
//                if (floatResult[2] > 0.7) {
//                    goodMark++;
//                } else {
//                    wrongMark++;
//                }
            }

//            listResult.add(intLabels[0] + "    " +
//                           intLabels[1] + "    " +
//                           intLabels[2] + "    ");
//            listResult.add(String.format("%.2f", floatResult[0]) + " " +
//                           String.format("%.2f", floatResult[1]) + " " +
//                           String.format("%.2f", floatResult[2]) + " " +
//                           ZonedDateTime.ofInstant(Instant.ofEpochSecond(kucoinKlines.get(i).getTime()), ZoneOffset.UTC) + " " +
//                           (floatResult[2] > 0.7 ? " +++++ " : "") + " " +
//                           (floatResult[0] > 0.7 ? " ----- " : ""));
//            listResult.add("--------------------------");
        }

        listResult.add(" GoodMark = " + goodMark + ", WrongMark = " + wrongMark + ", K = " + goodMark / (float)(goodMark + wrongMark));

        return listResult;
    }
}


