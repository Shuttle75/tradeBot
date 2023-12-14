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

    public StrategyController(Exchange exchange) {
        this.exchange = exchange;
    }

    @GetMapping(path = "predictStrategy")
    public List<String> checkPredict() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
            .truncatedTo(ChronoUnit.DAYS)
            .minusDays(7)
            .toEpochSecond(ZoneOffset.UTC);
        final BarSeries barSeries = new BaseBarSeries();
        final Strategy movingMomentumStrategy = MovingMomentumStrategy.buildStrategy(barSeries);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L);
        Collections.reverse(kucoinKlines);
        kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));

        List<String> listResult = new ArrayList<>();
        for (int i = 288; i < kucoinKlines.size() - PREDICT_DEEP; i++) {

            listResult.add(ZonedDateTime.ofInstant(Instant.ofEpochSecond(kucoinKlines.get(i).getTime()), ZoneOffset.UTC) + " " +
                           "  " + (movingMomentumStrategy.shouldEnter(i) ? " shouldEnter " : " ") +
                           "  " + (movingMomentumStrategy.shouldExit(i) ? " shouldExit " : " ") + " " + kucoinKlines.get(i).getClose());
            listResult.add("--------------------------");
        }

        return listResult;
    }
}


