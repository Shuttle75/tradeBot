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

import static com.trading.bot.configuration.BotConfig.CURRENCY_DELTA;
import static com.trading.bot.configuration.BotConfig.INPUT_SIZE;
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

        float[] predict = getPredict(kucoinKlines);
        String rates = printRates(predict);

        if (active) {
            if (lastKline0.getClose().compareTo(lastKline1.getOpen()) < 0 && predict[7] < 0.8                                                                         // Not predict
                    || lastKline0.getOpen().subtract(lastKline0.getClose()).floatValue() > CURRENCY_DELTA) {    // Minus CURRENCY_DELTA
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
            if (predict[7] > 0.8) {
                active = true;
                firstPrice = lastKline0.getClose();
                maxPrice = lastKline0.getClose();
                logger.info("{} BUY {} Price {}", rates, USDT, lastKline0.getClose());
            } else {
                logger.info("{}", rates);
            }
        }
    }

    private List<KucoinKline> getKlines() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
                .minusMinutes(10)
                .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
                .toEpochSecond(ZoneOffset.UTC);

        return getKucoinKlines(exchange, startDate, endDate);
    }

    private float[] getPredict(List<KucoinKline> kucoinKlines) {
        try (INDArray nextInput = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            nextInput.putScalar(new int[]{0, 0, 0},
                    kucoinKlines.get(0).getClose().subtract(kucoinKlines.get(0).getOpen()).floatValue());
            nextInput.putScalar(new int[]{0, 1, 0},
                    kucoinKlines.get(0).getVolume().floatValue());
            nextInput.putScalar(new int[]{0, 2, 0},
                    kucoinKlines.get(0).getClose().compareTo(kucoinKlines.get(0).getOpen()) > 0 ?
                            kucoinKlines.get(0).getOpen().subtract(kucoinKlines.get(0).getLow()).floatValue() :
                            kucoinKlines.get(0).getClose().subtract(kucoinKlines.get(0).getLow()).floatValue());

            return model.rnnTimeStep(nextInput).ravel().toFloatVector();
        }
    }
}
