package com.trading.bot.scheduler;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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

@Service
public class Trader {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    private final Exchange exchange;
    private final MultiLayerNetwork model;
    private boolean active;
    private BigDecimal maxPrice;
    private BigDecimal firstPrice;
    private float[] predict;
    private String rates;
    private int activeCounter = 20;

    private Ticker prevTicker;
    private Ticker lastTicker;

    @Value("${trader.buylimit}")
    public float traderBuyLimit;


    // Only for test !!!!!!!!!!!!!
    private BigDecimal USDT = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork model) {
        this.exchange = exchange;
        this.model = model;
    }

    @Scheduled(cron = "55 * * * * *")
    public void buy() throws IOException {
        List<KucoinKline> kucoinKlines = getKlines();
        KucoinKline lastKline0 = kucoinKlines.get(0);

        if (lastKline0.getVolume().floatValue() > 0.3F) {
            predict = getOneMinutePredict(kucoinKlines.get(0));
            rates = printRates(predict);
        }
        boolean isGoodTrend =
                kucoinKlines.get(kucoinKlines.size() - 1).getOpen()
                        .subtract(kucoinKlines.get(0).getClose()).floatValue() > CURRENCY_DELTA * 10F;

        if (!active) {
            if (predict[OUTPUT_SIZE - 1] > 0.7 && isGoodTrend) {
                active = true;
                firstPrice = lastKline0.getClose();
                maxPrice = lastKline0.getClose();
                activeCounter = predict[OUTPUT_SIZE - 1] > 0.7 ? 12 : 0;
                logger.info("{} BUY {} Price {}", rates, USDT, lastKline0.getClose());
            } else {
                logger.info("{} {}", rates, isGoodTrend);
            }
        }

        prevTicker = lastTicker;
    }

    @Scheduled(cron = "5/15 * * * * *")
    public void sell() throws IOException {
         lastTicker = getTicker(exchange);

        boolean downNoLessThenPrev = lastTicker.getLast().compareTo(prevTicker.getOpen()) < 0;
        boolean downNoLessThenDelta = lastTicker.getOpen().subtract(lastTicker.getLast()).floatValue() > CURRENCY_DELTA;
        boolean notGreen = lastTicker.getLast().compareTo(lastTicker.getOpen()) < 0;

        if (active) {
            if ((downNoLessThenPrev || downNoLessThenDelta) && activeCounter <= 0 && notGreen) {
                USDT = USDT.multiply(lastTicker.getLast()).divide(firstPrice, 2, RoundingMode.HALF_UP);
                logger.info("{} SELL {} firstPrice {} newPrice {}", rates, USDT, firstPrice, lastTicker.getLast());
                active = false;
            } else {
                if (maxPrice.compareTo(lastTicker.getLast()) < 0) {
                    maxPrice = lastTicker.getLast();
                }
                logger.info("{} HOLD {} {} {}", rates, downNoLessThenPrev, downNoLessThenDelta, notGreen);
            }
            activeCounter--;
        }
    }

    private List<KucoinKline> getKlines() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(30).toEpochSecond(ZoneOffset.UTC);

        return getKucoinKlines(exchange, startDate, 0L);
    }

    public float[] getOneMinutePredict(KucoinKline kucoinKline) {
        try (INDArray nextInput = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            nextInput.putScalar(new int[]{0, 0, 0},
                    kucoinKline.getClose()
                            .subtract(kucoinKline.getOpen()).movePointLeft(NORMAL).floatValue());
            nextInput.putScalar(new int[]{0, 1, 0},
                    kucoinKline.getVolume().movePointLeft(NORMAL).floatValue());

            return model.rnnTimeStep(nextInput).ravel().toFloatVector();
        }
    }
}
