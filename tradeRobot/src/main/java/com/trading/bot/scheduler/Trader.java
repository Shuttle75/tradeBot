package com.trading.bot.scheduler;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeries;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.MACDIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.util.TradeUtil.getKucoinKlines;
import static com.trading.bot.util.TradeUtil.getPredict;
import static com.trading.bot.util.TradeUtil.loadBarSeries;
import static com.trading.bot.util.TradeUtil.printRates;
import static java.util.Objects.isNull;

@Service
public class Trader {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    private final Exchange exchange;
    private final MultiLayerNetwork net;
    private boolean purchased;
    private BigDecimal firstPrice;
    private KucoinKline prevKline;
    private float[] predict;
    private final BarSeries barSeries;
    private final MACDIndicator macdLine;
    private final EMAIndicator emaF;
    private float macdHistogramValue;

    @Value("${trader.buylimit}")
    public float tradeLimit;


    // Only for test !!!!!!!!!!!!!
    private BigDecimal curAccount = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
        barSeries = new BaseBarSeries();
        macdLine = new MACDIndicator(new ClosePriceIndicator(barSeries));
        emaF = new EMAIndicator(new ClosePriceIndicator(barSeries), 9);
    }

    @PostConstruct
    public void postConstruct() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusDays(4).toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC).truncatedTo(ChronoUnit.MINUTES)
            .minusMinutes(5).toEpochSecond(ZoneOffset.UTC);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
        Collections.reverse(kucoinKlines);
        kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));
    }

//    @Scheduled(cron = "30 */5 * * * *")
    public void predict() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L);
        prevKline = kucoinKlines.get(1);

        loadBarSeries(barSeries, prevKline);
        macdHistogramValue = macdLine.getValue(barSeries.getEndIndex())
            .minus(emaF.getValue(barSeries.getEndIndex())).floatValue();

        predict = getPredict(prevKline, net);
        String rates = printRates(predict);
        logger.info("{}", rates);
    }

//    @Scheduled(cron = "5/10 * * * * *")
    public void sell() throws IOException {
        if (isNull(prevKline)) {
            return;                 // When prediction not run before
        }

        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L);
        KucoinKline lastKline = kucoinKlines.get(0);

        if (!purchased && predict[2] > tradeLimit && macdHistogramValue < 0) {
            firstPrice = lastKline.getClose();
            logger.info("BUY {} Price {}", curAccount, lastKline.getClose());
            purchased = true;
            return;
        }

        if (purchased && predict[0] > tradeLimit) {
            curAccount = curAccount.multiply(lastKline.getClose()).divide(firstPrice, 2, RoundingMode.HALF_UP);
            logger.info("SELL {} firstPrice {} newPrice {}", curAccount, firstPrice, lastKline.getClose());
            predict = new float[] {0F, 0F, 0F};
            purchased = false;
        }
    }
}
