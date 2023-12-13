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
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.util.TradeUtil.getKucoinKlines;
import static com.trading.bot.util.TradeUtil.loadBarSeries;

@RestController
public class PurchaseController {
    /** Logger. */
    private final Exchange exchange;
    private final MultiLayerNetwork net;


    public PurchaseController(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

    @GetMapping(path = "purchase")
    public List<String> checkPredict() throws IOException {
        final BarSeries barSeries = new BaseBarSeries();
        final Strategy movingMomentumStrategy = MovingMomentumStrategy.buildStrategy(barSeries);
        final long trainDate = LocalDateTime.of(2023, 12, 1, 0 ,0).toEpochSecond(ZoneOffset.UTC);
        long startDate = LocalDateTime.of(2023, 12, 2, 0 ,0).toEpochSecond(ZoneOffset.UTC);
        long endDate;

        boolean purchased = false;
        BigDecimal purchasePrice = new BigDecimal(0);
        BigDecimal walletTrade = new BigDecimal(1800);
        long purchaseDate = 0;
        List<String> listResult = new ArrayList<>();
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, trainDate, startDate);
        Collections.reverse(kucoinKlines);
        kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));

        for (int day = 2; day < 10; day++) {
            startDate = LocalDateTime.of(2023, 12, day, 0 ,0).toEpochSecond(ZoneOffset.UTC);
            endDate = LocalDateTime.of(2023, 12, day + 1, 0 ,0).toEpochSecond(ZoneOffset.UTC);

            kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
            Collections.reverse(kucoinKlines);
            kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));

            BigDecimal startWalletTrade = walletTrade;
            for (int i = 0; i < kucoinKlines.size(); i++) {

                if (!purchased && movingMomentumStrategy.shouldEnter(1440 * (day - 1) + i)) {
                    purchasePrice = kucoinKlines.get(i).getClose();
                    purchaseDate = kucoinKlines.get(i).getTime();
                    purchased = true;
                }
                if (purchased && movingMomentumStrategy.shouldExit(1440 * (day - 1) + i)) {
                    walletTrade = walletTrade.multiply(kucoinKlines.get(i).getClose()).divide(purchasePrice, RoundingMode.HALF_UP);
                    purchased = false;

                    listResult.add(ZonedDateTime.ofInstant(Instant.ofEpochSecond(purchaseDate), ZoneOffset.UTC) + " " +
                                   ZonedDateTime.ofInstant(Instant.ofEpochSecond(kucoinKlines.get(i).getTime()), ZoneOffset.UTC) + " " +
                                   new DecimalFormat("#0.000").format(purchasePrice) + " " +
                                   new DecimalFormat("#0.000").format(kucoinKlines.get(i).getClose()) + " - " +
                                   new DecimalFormat("#0.00").format(walletTrade));
                }
            }
            listResult.add("Day " + day + " startWalletTrade " + startWalletTrade.intValue() + " endWalletTrade " + walletTrade.intValue());
        }

        return listResult;
    }
}


