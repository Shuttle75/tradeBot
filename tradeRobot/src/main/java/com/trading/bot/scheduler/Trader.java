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
import java.util.List;

import static com.trading.bot.configuration.BotConfig.*;
import static com.trading.bot.util.TradeUtil.*;
import static java.util.Objects.nonNull;

@Service
public class Trader {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    private final Exchange exchange;
    private final MultiLayerNetwork net;
    private boolean active;
    private BigDecimal maxPrice;
    private BigDecimal firstPrice;
    private String rates;
    private int activeCounter = 20;

    private KucoinKline prevKline;
    private KucoinKline lastKline;

    @Value("${trader.buylimit}")
    public float traderBuyLimit;


    // Only for test !!!!!!!!!!!!!
    private BigDecimal curAccount = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

    @Scheduled(cron = "55 * * * * *")
    public void buy() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getAggregatedKucoinKlinesByMin(exchange, startDate, 0L);
        lastKline = kucoinKlines.get(0);

        float[] predict = getOneMinutePredict(kucoinKlines.get(0), net);
        rates = printRates(predict);
        boolean isGoodTrend =
                kucoinKlines.get(kucoinKlines.size() - 1).getOpen()
                        .subtract(kucoinKlines.get(0).getClose()).floatValue() < CURRENCY_DELTA * 10F;

        if (!active) {
            if (predict[OUTPUT_SIZE - 1] > 0.7 && isGoodTrend) {
                active = true;
                firstPrice = lastKline.getClose();
                maxPrice = lastKline.getClose();
                activeCounter = predict[OUTPUT_SIZE - 1] > 0.7 ? 12 : 0;
                logger.info("{} BUY {} Price {}", rates, curAccount, lastKline.getClose());
            } else {
                logger.info("{} {}", rates, isGoodTrend);
            }
        }

        prevKline = lastKline;
    }

    @Scheduled(cron = "5/15 * * * * *")
    public void sell() throws IOException {
        if (active && nonNull(prevKline)) {
            lastKline = getLastKucoinKlinesByMin(exchange);

            boolean downNoLessThenPrev = lastKline.getClose().compareTo(prevKline.getOpen()) < 0;
            boolean downNoLessThenDelta = lastKline.getOpen().subtract(lastKline.getClose()).floatValue() > CURRENCY_DELTA;
            boolean notGreen = lastKline.getClose().compareTo(lastKline.getOpen()) < 0;

            if ((downNoLessThenPrev || downNoLessThenDelta) && activeCounter <= 0 && notGreen) {
                curAccount = curAccount.multiply(lastKline.getClose()).divide(firstPrice, 2, RoundingMode.HALF_UP);
                logger.info("{} SELL {} firstPrice {} newPrice {}", rates, curAccount, firstPrice, lastKline.getClose());
                active = false;
            } else {
                if (maxPrice.compareTo(lastKline.getClose()) < 0) {
                    maxPrice = lastKline.getClose();
                }
                logger.info("{} HOLD {} {} {}", rates, downNoLessThenPrev, downNoLessThenDelta, notGreen);
            }
            activeCounter--;
        }
    }
}
