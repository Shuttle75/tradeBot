package com.trading.bot.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.BarSeries;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.*;

import static com.trading.bot.configuration.BotConfig.*;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min15;

public class TradeUtil {

    private TradeUtil() {
    }

    public static List<KucoinKline> getKucoinKlines(Exchange exchange, long startDate, long endDate) throws IOException {
        return ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, min15);
    }

    public static float[] getPredict(KucoinKline kline, MultiLayerNetwork net) {
        try (INDArray indData = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            calcData(indData, kline, 0, 0);

            return net.rnnTimeStep(indData).ravel().toFloatVector();
        }
    }

    public static int getDelta(List<KucoinKline> klines, int i) {
        BigDecimal delta = klines.get(i + PREDICT_DEEP).getClose()
            .divide(klines.get(i).getClose(), 6, RoundingMode.HALF_UP)
            .subtract(BigDecimal.ONE)
            .multiply(BigDecimal.valueOf(100));

        if (delta.floatValue() > PRICE_PERCENT) {
            return  2;
        } else if (delta.floatValue() < -PRICE_PERCENT / 2) {
            return  0;
        } else {
            return 1;
        }
    }

    public static void calcData(INDArray indData, KucoinKline kline, int i, int y) {
        indData.putScalar(new int[]{i, 0, y},
                          kline.getClose().subtract(kline.getOpen()).floatValue() / kline.getClose().floatValue() * 10F);
        indData.putScalar(new int[]{i, 1, y},
                          kline.getAmount().floatValue() / 10_000_000F);
        indData.putScalar(new int[]{i, 2, y},
                          kline.getClose().compareTo(kline.getOpen()) > 0 ?
                          kline.getHigh().subtract(kline.getClose()).floatValue() / kline.getClose().floatValue() * 10F :
                          kline.getHigh().subtract(kline.getOpen()).floatValue() / kline.getClose().floatValue() * 10F);
        indData.putScalar(new int[]{i, 3, y},
                          kline.getClose().compareTo(kline.getOpen()) > 0 ?
                          kline.getOpen().subtract(kline.getLow()).floatValue() / kline.getClose().floatValue() * 10F :
                          kline.getClose().subtract(kline.getLow()).floatValue() / kline.getClose().floatValue() * 10F);
        indData.putScalar(new int[]{i, 4, y},
                          Instant.ofEpochSecond(kline.getTime()).atOffset(ZoneOffset.UTC).getDayOfWeek().getValue() * 0.12F
                          + Instant.ofEpochSecond(kline.getTime()).atOffset(ZoneOffset.UTC).getHour() * 0.005F);
    }

    public static void loadBarSeries(BarSeries barSeries, KucoinKline kucoinKlines) {
        barSeries.addBar(Duration.ofMinutes(5L),
                         ZonedDateTime.ofInstant(Instant.ofEpochSecond(kucoinKlines.getTime()), ZoneOffset.UTC),
                         kucoinKlines.getOpen(),
                         kucoinKlines.getHigh(),
                         kucoinKlines.getLow(),
                         kucoinKlines.getClose(),
                         kucoinKlines.getVolume());
    }

    public static String printRates(float[] floatResult) {
        return String.format("%.2f", floatResult[0]) + " " +
                String.format("%.2f", floatResult[1]) + " " +
                String.format("%.2f", floatResult[2]);
    }
}
