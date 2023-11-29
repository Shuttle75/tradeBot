package com.trading.bot.scheduler;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.function.ToDoubleFunction;

import static com.trading.bot.configuration.BotConfig.RSI_INDICATOR;
import static com.trading.bot.configuration.BotConfig.TREND_QUEUE;
import static com.trading.bot.util.TradeUtil.*;
import static java.util.Objects.isNull;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min5;

@Service
public class Trader {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    private final Exchange exchange;
    private final MultiLayerNetwork net;
    private boolean purchased;
    private BigDecimal firstPrice;
    private float lastRsiIndicator = 100F;
    private KucoinKline prevKline;
    private float[] predict;
    private final LimitedQueue<BigDecimal> trendQueue = new LimitedQueue<>(TREND_QUEUE);
    private final BarSeries barSeries;
    private final RSIIndicator rsiIndicator;

    @Value("${trader.buylimit}")
    public float tradeLimit;


    // Only for test !!!!!!!!!!!!!
    private BigDecimal curAccount = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork net, BarSeries barSeries) {
        this.exchange = exchange;
        this.net = net;
        this.barSeries = barSeries;
        rsiIndicator = new RSIIndicator(new ClosePriceIndicator(barSeries), RSI_INDICATOR);
    }

    @Scheduled(cron = "30 */5 * * * *")
    public void predict() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L, min5);
        prevKline = kucoinKlines.get(1);
        loadBarSeries(barSeries, prevKline);
        lastRsiIndicator = rsiIndicator.getValue(barSeries.getEndIndex()).floatValue();
        predict = getPredict(prevKline, net, rsiIndicator.getValue(barSeries.getEndIndex()));
        String rates = printRates(predict);
        logger.info("{}", rates);
        trendQueue.clear();
    }

    @Scheduled(cron = "5/10 * * * * *")
    public void sell() throws IOException {
        if (isNull(prevKline)) {
            return;                 // When prediction not run before
        }

        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L, min5);
        KucoinKline lastKline = kucoinKlines.get(0);

        double lastDelta = lastKline.getClose().subtract(lastKline.getOpen()).doubleValue();
        trendQueue.add(lastKline.getClose().subtract(lastKline.getOpen()));

        if (!purchased
                && predict[2] > tradeLimit
                && lastRsiIndicator < 70F) {
            firstPrice = lastKline.getClose();
            logger.info("BUY {} Price {}", curAccount, lastKline.getClose());
            purchased = true;
            return;
        }

        if (purchased
                && lastRsiIndicator > 30F
                && (trendQueue.trendDown(BigDecimal::doubleValue, lastDelta) || (predict[0] > tradeLimit))) {
            curAccount = curAccount.multiply(lastKline.getClose()).divide(firstPrice, 2, RoundingMode.HALF_UP);
            logger.info("SELL {} firstPrice {} newPrice {}", curAccount, firstPrice, lastKline.getClose());
            predict = new float[] {0F, 0F, 0F};
            purchased = false;
        }
    }

    private static class LimitedQueue<E> extends LinkedList<E> {

        private final int limit;

        public LimitedQueue(int limit) {
            this.limit = limit;
        }

        @Override
        public boolean add(E o) {
            boolean added = super.add(o);
            while (added && size() > limit) {
                super.remove();
            }
            return added;
        }

        public boolean trendDown(ToDoubleFunction<E> toDoubleFunction, double lastDelta) {
            if (size() < limit) {
                return false;
            }
            return (this.stream().mapToDouble(toDoubleFunction).sum() + lastDelta) < 0;
        }
    }
}
