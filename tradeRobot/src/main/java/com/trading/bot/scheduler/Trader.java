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
import java.time.temporal.ChronoUnit;
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

    @Scheduled(cron = "50 * * * * *")
    public void trade() throws IOException, InterruptedException {
        List<KucoinKline> kucoinKlines = getKlines();
        KucoinKline lastKline0 = kucoinKlines.get(0);
        KucoinKline lastKline1 = kucoinKlines.get(1);

        if (active) {
            for (int i = 0; i < 10; i++) {
                Ticker ticker = getTicker(exchange);
                BigDecimal curPrice = ticker.getLast();

                if (lastKline0.getOpen().subtract(lastKline0.getClose())
                        .compareTo(lastKline1.getClose().subtract(lastKline1.getOpen())) > 0) {
                    USDT = USDT.multiply(curPrice).divide(firstPrice, 2, RoundingMode.HALF_UP);
                    logger.info("Sell crypto !!!!!!!! {} firstPrice {} newPrice {}", USDT, firstPrice, curPrice);
                    active = false;
                } else {
                    if (maxPrice.compareTo(curPrice) < 0) {
                        maxPrice = curPrice;
                    }
                    logger.info("HOLD the crypto maxPrice {} curPrice {}", maxPrice, curPrice);
                }

                TimeUnit.SECONDS.sleep(5);    // Cooling CPU
            }
        } else {
            float[] floatResult = getPredict(kucoinKlines);
            String rates = printRates(floatResult);

            if (floatResult[7] > 0.7) {
                active = true;
                firstPrice = lastKline0.getClose();
                maxPrice = lastKline0.getClose();
                logger.info("{}  Buy crypto !!!!!!!! {} Price {}", rates, USDT, lastKline0.getClose());
            } else {
                logger.info("{}  NOT Buy crypto", rates);
            }
        }
    }

    private List<KucoinKline> getKlines() throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.MINUTES)
                .minusHours(2)
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
