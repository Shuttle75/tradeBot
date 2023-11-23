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

    @Value("${trader.buylimit}")
    public float traderBuyLimit;


    // Only for test !!!!!!!!!!!!!
    private BigDecimal curAccount = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork net) {
        this.exchange = exchange;
        this.net = net;
    }

    @Scheduled(cron = "5 * * * * *")
    public void buy() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);
        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, 0L);
        prevKline = kucoinKlines.get(1);
        predict = getOneMinutePredict(prevKline, net);
        String rates = printRates(predict);
        logger.info("{}", rates);
    }

    @Scheduled(cron = "10/5 * * * * *")
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
        boolean isGreen = lastKline.getClose().compareTo(lastKline.getOpen()) > 0;
        boolean isRed = !isGreen;
        boolean notUp = predict[2] < 0.8;
        boolean goDown = predict[0] > 0.8;

        if (!purchased && predict[2] > 0.8 && isGreen) {
            firstPrice = lastKline.getClose();
            logger.info("BUY {} Price {}", curAccount, prevKline.getClose());
            purchased = true;
            return;
        }

        if (purchased && ((notUp && lessThenPrev) || (goDown && isRed) || lessThenDelta)) {
            curAccount = curAccount.multiply(lastKline.getClose()).divide(firstPrice, 2, RoundingMode.HALF_UP);
            logger.info("SELL {} firstPrice {} newPrice {}", curAccount, firstPrice, lastKline.getClose());
            purchased = false;
        }
    }
}
