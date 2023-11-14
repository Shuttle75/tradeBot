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
import java.util.concurrent.TimeUnit;

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
    public void trade() throws IOException, InterruptedException {

        if (active) {
            while (true) {
                List<KucoinKline> kucoinKlines = getKlines();
                KucoinKline lastKline0 = kucoinKlines.get(0);
                KucoinKline lastKline1 = kucoinKlines.get(1);
                BigDecimal curPrice = kucoinKlines.get(0).getClose();

                float[] floatResult = getPredict(kucoinKlines);
                String rates = printRates(floatResult);

                if (lastKline0.getClose().compareTo(lastKline1.getOpen()) < 0
                        && lastKline0.getClose().subtract(lastKline0.getOpen()).floatValue() < 0) {
                    USDT = USDT.multiply(curPrice).divide(firstPrice, 2, RoundingMode.HALF_UP);
                    logger.info("{} Sell crypto !!!!!!!! {} firstPrice {} newPrice {}", rates, USDT, firstPrice, curPrice);
                    active = false;
                    return;
                } else {
                    if (maxPrice.compareTo(curPrice) < 0) {
                        maxPrice = curPrice;
                    }
                    logger.info("{} HOLD the crypto maxPrice {} curPrice {}", rates, maxPrice, curPrice);
                }

                TimeUnit.SECONDS.sleep(5);    // Cooling CPU
            }
        } else {
            List<KucoinKline> kucoinKlines = getKlines();
            KucoinKline lastKline0 = kucoinKlines.get(0);

            float[] floatResult = getPredict(kucoinKlines);
            String rates = printRates(floatResult);

            if (floatResult[7] > 0.5) {
                active = true;
                firstPrice = lastKline0.getClose();
                maxPrice = lastKline0.getClose();
                logger.info("{} Buy crypto !!!!!!!! {} Price {}", rates, USDT, lastKline0.getClose());
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
        try (INDArray nextInput = Nd4j.zeros(1, 2, 1)) {

            nextInput.putScalar(new int[]{0, 0, 0},
                    kucoinKlines.get(0).getClose()
                            .subtract(kucoinKlines.get(0).getOpen())
                            .floatValue());
            nextInput.putScalar(new int[]{0, 1, 0},
                    kucoinKlines.get(0).getVolume()
                            .floatValue());

            return model.rnnTimeStep(nextInput).ravel().toFloatVector();
        }
    }
}
