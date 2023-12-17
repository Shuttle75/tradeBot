package com.trading.bot.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.KlineIntervalType;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.BarSeries;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
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

    public static float[] getPredict(KucoinKline kucoinKline, MultiLayerNetwork net) {
        try (INDArray indData = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            calcData(indData, kucoinKline, 0, 0);

            return net.rnnTimeStep(indData).ravel().toFloatVector();
        }
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int i) {
        BigDecimal delta = kucoinKlines.get(i + PREDICT_DEEP).getClose()
            .divide(kucoinKlines.get(i).getClose(), 6, RoundingMode.HALF_UP)
            .subtract(BigDecimal.ONE)
            .multiply(BigDecimal.valueOf(100));

        if (delta.floatValue() > UP_PERCENT) {
            return  2;
        } else if (delta.floatValue() < -DOWN_PERCENT) {
            return  0;
        } else {
            return 1;
        }
    }

    public static double getAvgCandle(Exchange exchange) throws IOException {
        return getKucoinKlines(exchange, LocalDateTime.now(ZoneOffset.UTC).minusDays(7).toEpochSecond(ZoneOffset.UTC), 0L)
            .stream()
            .map(kucoinKline -> kucoinKline.getClose().subtract(kucoinKline.getOpen()))
            .filter(candle -> candle.signum() > 0)
            .mapToDouble(candle -> candle.doubleValue())
            .average()
            .orElse(Double.MIN_VALUE);
    }

    public static void calcData(INDArray indData, KucoinKline kucoinKline, int i, int y) {
        indData.putScalar(new int[]{i, 0, y},
                          kucoinKline.getClose().subtract(kucoinKline.getOpen()).floatValue());
        indData.putScalar(new int[]{i, 1, y}, kucoinKline.getVolume().floatValue() * 0.0001);
        indData.putScalar(new int[]{i, 2, y},
                          kucoinKline.getClose().compareTo(kucoinKline.getOpen()) > 0 ?
                          kucoinKline.getHigh().subtract(kucoinKline.getClose()).floatValue() :
                          kucoinKline.getHigh().subtract(kucoinKline.getOpen()).floatValue());
        indData.putScalar(new int[]{i, 3, y},
                          kucoinKline.getClose().compareTo(kucoinKline.getOpen()) > 0 ?
                          kucoinKline.getOpen().subtract(kucoinKline.getLow()).floatValue() :
                          kucoinKline.getClose().subtract(kucoinKline.getLow()).floatValue());
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
