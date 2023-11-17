package com.trading.bot.scheduler;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
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

    @Value("${trader.buylimit}")
    public float traderBuyLimit;


    // Only for test !!!!!!!!!!!!!
    private BigDecimal USDT = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork model) {
        this.exchange = exchange;
        this.model = model;
    }

    @Scheduled(cron = "55 * * * * *")
    public void trade() throws IOException {
        List<KucoinKline> kucoinKlines = getKlines();
        KucoinKline lastKline0 = kucoinKlines.get(0);
        KucoinKline lastKline1 = kucoinKlines.get(1);

        float[] predict = getOneMinutePredict(kucoinKlines.get(0));
        String rates = printRates(predict);

        boolean downNoLessThenPrev = lastKline0.getClose().compareTo(lastKline1.getOpen()) < 0;
        boolean downNoLessThenDelta = lastKline0.getOpen().subtract(lastKline0.getClose()).floatValue() > CURRENCY_DELTA;
        boolean notGreen = lastKline0.getClose().compareTo(lastKline0.getOpen()) < 0;
        boolean isGreen = lastKline0.getClose().compareTo(lastKline0.getOpen()) > 0;
        boolean isGoodTrend = isGoodTrend(exchange);

        if (active) {
            if ((downNoLessThenPrev || downNoLessThenDelta) && predict[OUTPUT_SIZE - 1] < 0.7 && notGreen) {
                USDT = USDT.multiply(lastKline0.getClose()).divide(firstPrice, 2, RoundingMode.HALF_UP);
                logger.info("{} SELL {} firstPrice {} newPrice {}", rates, USDT, firstPrice, lastKline0.getClose());
                active = false;
            } else {
                if (maxPrice.compareTo(lastKline0.getClose()) < 0) {
                    maxPrice = lastKline0.getClose();
                }
                logger.info("{} HOLD maxPrice {} curPrice {}", rates, maxPrice, lastKline0.getClose());
            }
        } else {
            if (predict[OUTPUT_SIZE - 1] > 0.7 && isGoodTrend) {
                active = true;
                firstPrice = lastKline0.getClose();
                maxPrice = lastKline0.getClose();
                logger.info("{} BUY {} Price {}", rates, USDT, lastKline0.getClose());
            } else {
                logger.info("{} {}", rates, isGoodTrend);
            }
        }
    }

    private List<KucoinKline> getKlines() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(2).toEpochSecond(ZoneOffset.UTC);

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
