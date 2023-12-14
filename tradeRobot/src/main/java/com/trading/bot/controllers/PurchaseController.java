package com.trading.bot.controllers;

import com.trading.bot.configuration.MovingMomentumStrategy;
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


    public PurchaseController(Exchange exchange) {
        this.exchange = exchange;
    }

    @GetMapping(path = "purchase")
    public List<String> checkPredict() throws IOException {
        final BarSeries barSeries = new BaseBarSeries();
        final Strategy movingMomentumStrategy = MovingMomentumStrategy.buildStrategy(barSeries);
        final long trainDate = LocalDateTime.of(2023, 11, 1, 0 ,0).toEpochSecond(ZoneOffset.UTC);
        long startDate = LocalDateTime.of(2023, 11, 2, 0 ,0).toEpochSecond(ZoneOffset.UTC);
        long endDate;

        boolean purchased = false;
        BigDecimal purchasePrice = new BigDecimal(0);
        BigDecimal walletTrade = new BigDecimal(1800);
        BigDecimal volume = new BigDecimal(1800);
        long purchaseDate = 0;
        List<String> listResult = new ArrayList<>();
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, trainDate, startDate);
        Collections.reverse(kucoinKlines);
        kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));

        for (int day = 2; day < 30; day++) {
            startDate = LocalDateTime.of(2023, 11, day, 0 ,0).toEpochSecond(ZoneOffset.UTC);
            endDate = LocalDateTime.of(2023, 11, day + 1, 0 ,0).toEpochSecond(ZoneOffset.UTC);

            kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
            Collections.reverse(kucoinKlines);
            kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));


            for (int i = 0; i < kucoinKlines.size(); i++) {
                int seriesIndex = 288 * (day - 1) + i;
                BigDecimal startWalletTrade = walletTrade;

                if (!purchased
                        && movingMomentumStrategy.shouldEnter(seriesIndex)) {
                    purchasePrice = kucoinKlines.get(i).getClose();
                    purchaseDate = kucoinKlines.get(i).getTime();
                    volume = kucoinKlines.get(i).getVolume();
                    purchased = true;
                }

                if (purchased
                        && movingMomentumStrategy.shouldExit(seriesIndex)) {
                    walletTrade = walletTrade.multiply(kucoinKlines.get(i).getClose()).divide(purchasePrice, RoundingMode.HALF_UP);
                    purchased = false;

                    listResult.add(ZonedDateTime.ofInstant(Instant.ofEpochSecond(purchaseDate), ZoneOffset.UTC) + " " +
                                   ZonedDateTime.ofInstant(Instant.ofEpochSecond(kucoinKlines.get(i).getTime()), ZoneOffset.UTC) + "   " +
                                   new DecimalFormat("#0.000").format(purchasePrice) + " " +
                                   new DecimalFormat("#0.000").format(kucoinKlines.get(i).getClose()) + "   " +
                                   new DecimalFormat("#0.00").format(startWalletTrade) + " " +
                                   new DecimalFormat("#0.00").format(walletTrade) + " " +
                                   new DecimalFormat("#0.00").format(walletTrade.subtract(startWalletTrade)) + "   " +
                            ((kucoinKlines.get(i).getTime() - purchaseDate) / 60) + " ----- " +
                            new DecimalFormat("#0.000").format(volume));
                }
            }
            listResult.add("Day " + day);
        }

        return listResult;
    }
}


