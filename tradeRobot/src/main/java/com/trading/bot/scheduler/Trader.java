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
    private KucoinKline prevKline;
    private float[] predict;

    @Value("${trader.buylimit}")
    public float traderBuyLimit;


    // Only for test !!!!!!!!!!!!!
    private BigDecimal curAccount = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

    @Scheduled(cron = "5 */5 * * * *")
    public void buy() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L);
        prevKline = kucoinKlines.get(1);
        predict = getOneMinutePredict(prevKline, net);
        rates = printRates(predict);

        if (!active) {
            if (predict[OUTPUT_SIZE - 1] > 0.7) {
                active = true;
                firstPrice = prevKline.getClose();
                maxPrice = prevKline.getClose();
                logger.info("{} BUY {} Price {}", rates, curAccount, prevKline.getClose());
            } else {
                logger.info("{}", rates);
            }
        }
    }

    @Scheduled(cron = "5/15 * * * * *")
    public void sell() throws IOException {
        if (active && nonNull(prevKline) && predict[OUTPUT_SIZE - 1] < 0.7) {
            if (predict[OUTPUT_SIZE - 1] < 0.7) {
                logger.info("{} HOLD by predict", rates);
                return;
            }

            final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
            List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L);
            KucoinKline lastKline = kucoinKlines.get(0);

            boolean downNoLessThenPrev = lastKline.getClose().compareTo(prevKline.getOpen()) < 0;
            boolean downNoLessThenDelta = lastKline.getOpen().subtract(lastKline.getClose()).floatValue() > CURRENCY_DELTA;

            if ((downNoLessThenPrev || downNoLessThenDelta || predict[0] > 0.7)) {
                curAccount = curAccount.multiply(lastKline.getClose()).divide(firstPrice, 2, RoundingMode.HALF_UP);
                logger.info("{} SELL {} firstPrice {} newPrice {}", rates, curAccount, firstPrice, lastKline.getClose());
                active = false;
            } else {
                if (maxPrice.compareTo(lastKline.getClose()) < 0) {
                    maxPrice = lastKline.getClose();
                }
                logger.info("{} HOLD", rates);
            }
        }
    }
}
