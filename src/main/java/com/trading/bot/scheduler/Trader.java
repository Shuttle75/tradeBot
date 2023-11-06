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

        if (purchase.isActive()) {
            if ((floatResult[0][0] + floatResult[0][1] + floatResult[0][2]) > 0.8 ||
                    this.purchase.getCurPrice() - kucoinKlines.get(0).getClose().floatValue() > 5.0) {
                this.purchase.setActive(false);
                USDT = USDT / this.purchase.getCurPrice() * kucoinKlines.get(0).getClose().floatValue();
                logger.info(printRates(floatResult) + " Sell crypto !!!!!!!! {} ", USDT);
            } else {
                logger.info(printRates(floatResult) + " HOLD the crypto");
            }
        } else {
            if ((floatResult[0][6] + floatResult[0][7]) > 0.8) {
                this.purchase.setActive(true);
                this.purchase.setCurPrice(buyPrice);
                logger.info(printRates(floatResult) + " Buy crypto !!!!!!!! Price {}", buyPrice);
            } else {
                logger.info(printRates(floatResult) + " NOT Buy crypto");
            }
        }
    }

    private static String printRates(float[][] floatResult) {
        return String.format("%.2f", floatResult[0][0]) + " " +
                String.format("%.2f", floatResult[0][1]) + " " +
                String.format("%.2f", floatResult[0][2]) + " " +
                String.format("%.2f", floatResult[0][3]) + " | " +
                String.format("%.2f", floatResult[0][4]) + " " +
                String.format("%.2f", floatResult[0][5]) + " " +
                String.format("%.2f", floatResult[0][6]) + " " +
                String.format("%.2f", floatResult[0][7]);
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
            floatData[i][y] = kucoinKlines.get(i + y).getClose()
                    .subtract(kucoinKlines.get(i + y).getOpen())
                    .multiply(kucoinKlines.get(i + y).getVolume()).floatValue();
        }

        return model.output(Nd4j.create(floatData)).toFloatMatrix();
    }
}