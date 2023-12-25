package com.trading.bot.controllers;

import com.trading.bot.configuration.MovingMomentumStrategy;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeries;
import org.ta4j.core.BaseTradingRecord;
import org.ta4j.core.Strategy;
import org.ta4j.core.TradingRecord;
import org.ta4j.core.num.DecimalNum;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


import static com.trading.bot.util.TradeUtil.getPredict;
import static com.trading.bot.util.TradeUtil.loadBarSeries;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min15;

@RestController
public class PurchaseController {
    /** Logger. */
    private final Exchange exchange;
    private final MultiLayerNetwork net;


    public PurchaseController(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

/*
    GET http://localhost:8080/purchase?baseSymbol=SOL&counterSymbol=USDT&startDate=2023-11-01T00:00:00&endDate=2023-12-01T00:00:00&walletUSDT=1800&stopLoss=95
*/
    @GetMapping(path = "purchase")
    public List<String> checkPredict(
            @RequestParam String baseSymbol,
            @RequestParam String counterSymbol,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @RequestParam BigDecimal walletUSDT) throws IOException {

        final BarSeries barSeries = new BaseBarSeries();
        final Strategy strategy = MovingMomentumStrategy.buildStrategy(barSeries);

        long purchaseDate = 0;
        BigDecimal walletUSDTBefore = BigDecimal.valueOf(0);
        BigDecimal exitPrice = BigDecimal.ZERO;
        BigDecimal walletBase = BigDecimal.ZERO;
        List<String> listResult = new ArrayList<>();
        List<KucoinKline> klines = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(
                        new CurrencyPair(baseSymbol, counterSymbol),
                        startDate.minusDays(1).toEpochSecond(ZoneOffset.UTC),
                        startDate.toEpochSecond(ZoneOffset.UTC),
                        min15);
        Collections.reverse(klines);
        klines.forEach(kline -> {
            getPredict(kline, net);
            loadBarSeries(barSeries, kline);
        });

        TradingRecord tradingRecord = new BaseTradingRecord();
        float purchasePredict = 0F;
        for (int day = 0; day < ChronoUnit.DAYS.between(startDate, endDate); day++) {

            klines = ((KucoinMarketDataService) exchange.getMarketDataService())
                    .getKucoinKlines(
                            new CurrencyPair(baseSymbol, counterSymbol),
                            startDate.plusDays(day).toEpochSecond(ZoneOffset.UTC),
                            startDate.plusDays(day + 1L).toEpochSecond(ZoneOffset.UTC),
                            min15);
            Collections.reverse(klines);


            for (int i = 0; i < klines.size(); i++) {
                float[] floatResult = getPredict(klines.get(i), net);

                loadBarSeries(barSeries, klines.get(i));

                final int index = 96 + 96 * day + i;
                final BigDecimal closePrice = klines.get(i).getClose();

                if (tradingRecord.isClosed()
                    && floatResult[2] > 0.7
//                    && strategy.shouldEnter(index, tradingRecord)
                ) {
                    purchaseDate = klines.get(i).getTime();
                    walletUSDTBefore = walletUSDT;
                    walletBase = walletUSDT.divide(closePrice, 0, RoundingMode.DOWN);
                    walletUSDT = walletUSDT.subtract(walletBase.multiply(closePrice));

                    tradingRecord.enter(index, DecimalNum.valueOf(closePrice), DecimalNum.valueOf(walletBase));

                    purchasePredict = floatResult[2];
                }

                if (!tradingRecord.isClosed()
                    && floatResult[0] > 0.7
//                    && strategy.shouldExit(index, tradingRecord)
                ) {

                    if (walletBase.compareTo(BigDecimal.valueOf(0)) > 0) {
                        tradingRecord.exit(index, DecimalNum.valueOf(closePrice), tradingRecord.getCurrentPosition().getEntry().getAmount());
                        walletUSDT = walletUSDT.add(walletBase.multiply(closePrice));
                        walletBase = BigDecimal.valueOf(0);
                        exitPrice = closePrice;
                    } else {
                        tradingRecord.exit(index, DecimalNum.valueOf(exitPrice), tradingRecord.getCurrentPosition().getEntry().getAmount());
                    }

                    listResult.add(ZonedDateTime.ofInstant(Instant.ofEpochSecond(purchaseDate), ZoneOffset.UTC) + " " +
                                   ZonedDateTime.ofInstant(Instant.ofEpochSecond(klines.get(i).getTime()), ZoneOffset.UTC) + "   " +
                                   new DecimalFormat("#0.000").format(tradingRecord.getLastPosition().getEntry().getPricePerAsset().doubleValue()) + " " +
                                   new DecimalFormat("#0.000").format(tradingRecord.getLastPosition().getExit().getPricePerAsset().doubleValue()) + "   " +
                                   new DecimalFormat("#0.00").format(walletUSDTBefore.doubleValue()) + " " +
                                   new DecimalFormat("#0.00").format(walletUSDT.doubleValue()) + " " +
                                   new DecimalFormat("#0.00").format(floatResult[0])  + " -- " + new DecimalFormat("#0.00").format(purchasePredict));
                }
            }
            listResult.add("Day " + day);
        }

        return listResult;
    }
}


