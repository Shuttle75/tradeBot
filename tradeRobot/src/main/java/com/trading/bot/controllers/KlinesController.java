package com.trading.bot.controllers;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeries;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.configuration.BotConfig.*;
import static com.trading.bot.util.TradeUtil.*;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min5;

@RestController
public class KlinesController {
    /** Logger. */
    private final Exchange exchange;
    private final MultiLayerNetwork net;

    public KlinesController(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

    @GetMapping(path = "predict")
    public List<String> checkPredict() throws IOException {
        int goodMark = 0;
        int wrongMark = 0;
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
            .truncatedTo(ChronoUnit.DAYS)
            .minusDays(5)
            .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
            .truncatedTo(ChronoUnit.DAYS)
            .minusDays(1)
            .toEpochSecond(ZoneOffset.UTC);

        final BarSeries barSeries = new BaseBarSeries();
        final RSIIndicator rsiIndicator = new RSIIndicator(new ClosePriceIndicator(barSeries), RSI_INDICATOR);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate, min5);
        Collections.reverse(kucoinKlines);
        kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));

        net.rnnClearPreviousState();

        List<String> listResult = new ArrayList<>();
        for (int i = 0; i < kucoinKlines.size() - FUTURE_PREDICT; i++) {
            float[] floatResult = getPredict(kucoinKlines.get(i), net, rsiIndicator.getValue(i));
            int[] intLabels = new int[OUTPUT_SIZE];
            intLabels[getDelta(rsiIndicator, i)] = 1;

            if (intLabels[0] == 1) {
                if (floatResult[0] > 0.7) {
                    goodMark++;
                } else {
                    wrongMark++;
                }
            }
            if (intLabels[2] == 1) {
                if (floatResult[2] > 0.7) {
                    goodMark++;
                } else {
                    wrongMark++;
                }
            }

            listResult.add(intLabels[0] + "    " +
                           intLabels[1] + "    " +
                           intLabels[2] + "    ");
            listResult.add(String.format("%.2f", floatResult[0]) + " " +
                           String.format("%.2f", floatResult[1]) + " " +
                           String.format("%.2f", floatResult[2]));
            listResult.add("--------------------------");
        }

        listResult.add(" GoodMark = " + goodMark + ", WrongMark = " + wrongMark + ", K = " + goodMark / (float)(goodMark + wrongMark));

        return listResult;
    }
}


