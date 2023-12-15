package com.trading.bot.controllers;

import com.trading.bot.configuration.MovingMomentumStrategy;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.ta4j.core.*;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.num.Num;

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

        Num walletTrade = DecimalNum.valueOf(1800);
        long purchaseDate = 0;
        List<String> listResult = new ArrayList<>();
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, trainDate, startDate);
        Collections.reverse(kucoinKlines);
        kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));

        TradingRecord tradingRecord = new BaseTradingRecord();
        tradingRecord.enter(0, DecimalNum.valueOf(100), DecimalNum.valueOf(40));
        tradingRecord.exit(1, DecimalNum.valueOf(100), DecimalNum.valueOf(40));

        for (int day = 2; day < 30; day++) {
            startDate = LocalDateTime.of(2023, 11, day, 0 ,0).toEpochSecond(ZoneOffset.UTC);
            endDate = LocalDateTime.of(2023, 11, day + 1, 0 ,0).toEpochSecond(ZoneOffset.UTC);

            kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
            Collections.reverse(kucoinKlines);
            kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));


            for (int i = 0; i < kucoinKlines.size(); i++) {
                int index = 96 * (day - 1) + i;
                Num startWalletTrade = walletTrade;

                if (tradingRecord.isClosed()
                        && movingMomentumStrategy.shouldEnter(index, tradingRecord)) {
                    purchaseDate = kucoinKlines.get(i).getTime();
                    tradingRecord.enter(index, DecimalNum.valueOf(kucoinKlines.get(i).getClose()), DecimalNum.valueOf(40));
                }

                if (!tradingRecord.isClosed()
                        && movingMomentumStrategy.shouldExit(index, tradingRecord)) {
                    tradingRecord.exit(index, DecimalNum.valueOf(kucoinKlines.get(i).getClose()), DecimalNum.valueOf(40));
                    walletTrade = walletTrade.plus(tradingRecord.getLastPosition().getProfit());

                    listResult.add(ZonedDateTime.ofInstant(Instant.ofEpochSecond(purchaseDate), ZoneOffset.UTC) + " " +
                                   ZonedDateTime.ofInstant(Instant.ofEpochSecond(kucoinKlines.get(i).getTime()), ZoneOffset.UTC) + "   " +
                                   new DecimalFormat("#0.000").format(tradingRecord.getLastPosition().getEntry().getPricePerAsset().doubleValue()) + " " +
                                   new DecimalFormat("#0.000").format(tradingRecord.getLastPosition().getExit().getPricePerAsset().doubleValue()) + "   " +
                                   new DecimalFormat("#0.00").format(startWalletTrade.doubleValue()) + " " +
                                   new DecimalFormat("#0.00").format(walletTrade.doubleValue()) + " " +
                                   new DecimalFormat("#0.00").format(tradingRecord.getLastPosition().getProfit().doubleValue()));
                }
            }
            listResult.add("Day " + day);
        }

        return listResult;
    }
}


