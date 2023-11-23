package com.trading.bot.scheduler;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import static com.trading.bot.util.TradeUtil.*;
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
    private final LimitedQueue<BigDecimal> trendQueue = new LimitedQueue<>(4);

    @Value("${trader.buylimit}")
    public float traderBuyLimit;


    // Only for test !!!!!!!!!!!!!
    private BigDecimal curAccount = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

    @Scheduled(cron = "5 */5 * * * *")
    public void predict() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L);
        prevKline = kucoinKlines.get(1);
        predict = getOneMinutePredict(prevKline, net);
        String rates = printRates(predict);
        logger.info("{}", rates);
        trendQueue.clear();
    }

    @Scheduled(cron = "10/10 * * * * *")
    public void sell() throws IOException {
        if (isNull(prevKline)) {
            return;                 // When prediction not run before
        }

        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L);
        KucoinKline lastKline = kucoinKlines.get(0);

        boolean lessThenPrev = lastKline.getClose().compareTo(prevKline.getLow()) < 0;
        boolean lessThenDelta = lastKline.getOpen()
                .subtract(lastKline.getClose())
                .compareTo(lastKline.getClose().movePointLeft(3)) > 0;
        boolean isRed = lastKline.getOpen().compareTo(lastKline.getClose()) > 0;

        trendQueue.add(lastKline.getClose());

        if (!purchased && predict[2] > 0.8 && trendQueue.trendIsUp()) {
            firstPrice = lastKline.getClose();
            logger.info("BUY {} Price {}", curAccount, prevKline.getClose());
            purchased = true;
            return;
        }

        if (purchased && ((predict[2] < 0.8 && lessThenPrev) || (predict[0] > 0.8 && isRed) || lessThenDelta)) {
            curAccount = curAccount.multiply(lastKline.getClose()).divide(firstPrice, 2, RoundingMode.HALF_UP);
            logger.info("SELL {} firstPrice {} newPrice {}", curAccount, firstPrice, lastKline.getClose());
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

        public boolean trendIsUp() {
            return size() == limit
                    && this.stream().sorted().collect(Collectors.toList()).equals(new ArrayList<>(this));
        }
    }
}
