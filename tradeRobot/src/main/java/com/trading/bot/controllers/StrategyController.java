package com.trading.bot.controllers;

import com.trading.bot.configuration.MovingMomentumStrategy;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeries;
import org.ta4j.core.Strategy;

import java.io.IOException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.configuration.BotConfig.OUTPUT_SIZE;
import static com.trading.bot.configuration.BotConfig.PREDICT_DEEP;
import static com.trading.bot.util.TradeUtil.getAvgCandle;
import static com.trading.bot.util.TradeUtil.getDelta;
import static com.trading.bot.util.TradeUtil.getKucoinKlines;
import static com.trading.bot.util.TradeUtil.getPredict;
import static com.trading.bot.util.TradeUtil.loadBarSeries;

@RestController
public class StrategyController {
    /** Logger. */
    private final Exchange exchange;
    private final MultiLayerNetwork net;

    public StrategyController(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

    @GetMapping(path = "predictStrategy")
    public List<String> checkPredict() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
            .truncatedTo(ChronoUnit.DAYS)
            .minusDays(8)
            .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
            .truncatedTo(ChronoUnit.DAYS)
            .minusDays(1)
            .toEpochSecond(ZoneOffset.UTC);
        final double avgCandle = getAvgCandle(exchange);
        final BarSeries barSeries = new BaseBarSeries();
        final Strategy movingMomentumStrategy = MovingMomentumStrategy.buildStrategy(barSeries);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
        Collections.reverse(kucoinKlines);
        kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));

        List<String> listResult = new ArrayList<>();
        for (int i = 300; i < kucoinKlines.size() - PREDICT_DEEP; i++) {

            listResult.add(ZonedDateTime.ofInstant(Instant.ofEpochSecond(kucoinKlines.get(i).getTime()), ZoneOffset.UTC) + " " +
                           "  " + (movingMomentumStrategy.shouldEnter(i) ? " shouldEnter " : " ") +
                           "  " + (movingMomentumStrategy.shouldExit(i) ? " shouldExit " : " "));
            listResult.add("--------------------------");
        }

        return listResult;
    }
}


