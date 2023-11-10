package com.trading.bot.scheduler;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
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



    // Only for test !!!!!!!!!!!!!
    private BigDecimal USDT = BigDecimal.valueOf(10000.0F);

    public Trader(Exchange exchange, MultiLayerNetwork model) {
        this.exchange = exchange;
        this.model = model;
    }

    @Scheduled(cron = "*/5 * * * * *")
    public void trade() throws IOException {
        if (active) {
            Ticker ticker = getTicker(exchange);
            BigDecimal curPrice = ticker.getLast();
            BigDecimal curVolume = ticker.getVolume();

            if (maxPrice.subtract(curPrice).floatValue() > CURRENCY_DELTA / curVolume.floatValue()) {
                USDT = USDT.divide(firstPrice, 2, RoundingMode.HALF_UP).multiply(curPrice);
                logger.info("Sell crypto !!!!!!!! {} firstPrice {} newPrice {}", USDT, firstPrice, curPrice);
                active = false;
            } else {
                if (maxPrice.compareTo(curPrice) > 0) {
                    maxPrice = curPrice;
                }
                logger.info("HOLD the crypto maxPrice {}", maxPrice);
            }
        } else {
            List<KucoinKline> kucoinKlines = getKlines();
            KucoinKline lastKline = kucoinKlines.get(0);
            float[] floatResult = getPredict(kucoinKlines);
            String rates = printRates(floatResult);

            if (floatResult[7] > 0.8) {
                active = true;
                firstPrice = lastKline.getClose();
                maxPrice = lastKline.getClose();
                logger.info("{}  Buy crypto !!!!!!!! Price {}", rates, lastKline.getClose());
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
        float[][] floatData = new float[1][TRAIN_DEEP];
        int i = 0;
        for (int y = 0; y < TRAIN_DEEP; y++) {
            floatData[i][y] = calcData(kucoinKlines, i, y, 0);
        }

        return model.output(Nd4j.create(floatData)).toFloatVector();
    }
}
