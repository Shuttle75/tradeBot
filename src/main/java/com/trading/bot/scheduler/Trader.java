package com.trading.bot.scheduler;

import com.trading.bot.dto.Purchase;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.IOException;
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

    private final Purchase purchase = new Purchase();



    // Only for test !!!!!!!!!!!!!
    private float buyPrice;
    private float USDT = 10000.0F;


    public Trader(Exchange exchange, MultiLayerNetwork model) {
        this.exchange = exchange;
        this.model = model;
    }

    @Scheduled(cron = "30 * * * * *")
    public void trade() throws IOException {
        List<KucoinKline> kucoinKlines = getKlines();
        float[][] floatResult = getPredict(kucoinKlines);
        String rates = printRates(floatResult);
        float newPrice = kucoinKlines.get(0).getClose().floatValue();

        if (purchase.isActive()) {
            if (this.purchase.getMaxPrice() - newPrice > CURRENCY_DELTA * 2) {
                this.purchase.setActive(false);
                USDT = USDT / this.purchase.getCurPrice() * newPrice;
                logger.info("{} Sell crypto !!!!!!!! {} ", rates, USDT);
            } else {
                this.purchase.setCurPrice(buyPrice);
                this.purchase.setMaxPrice(buyPrice);
                logger.info("{}  HOLD the crypto", rates);
            }
        } else {
            if ((floatResult[0][7]) > 0.8) {
                this.purchase.setActive(true);
                this.purchase.setCurPrice(buyPrice);
                this.purchase.setMinPrice(buyPrice);
                this.purchase.setMaxPrice(buyPrice);
                logger.info("{}  Buy crypto !!!!!!!! Price {}", rates, buyPrice);
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

    private float[][] getPredict(List<KucoinKline> kucoinKlines) {
        buyPrice = kucoinKlines.get(0).getClose().floatValue();

        float[][] floatData = new float[1][TRAIN_DEEP];
        int i = 0;
        for (int y = 0; y < TRAIN_DEEP; y++) {
            floatData[i][y] = calcData(kucoinKlines, i, y, 0);
        }

        return model.output(Nd4j.create(floatData)).toFloatMatrix();
    }
}
