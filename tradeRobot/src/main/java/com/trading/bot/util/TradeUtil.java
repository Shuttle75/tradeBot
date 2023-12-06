package com.trading.bot.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.KlineIntervalType;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.BarSeries;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.num.DecimalNum;

import java.io.IOException;
import java.math.BigDecimal;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.*;
import java.util.function.IntToDoubleFunction;

import static com.trading.bot.configuration.BotConfig.*;

public class TradeUtil {

    private TradeUtil() {
    }

    public static List<KucoinKline> getKucoinKlines(Exchange exchange, long startDate, long endDate) throws IOException {
        return ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, KlineIntervalType.min5);
    }

    public static float[] getPredict(KucoinKline kucoinKline, MultiLayerNetwork net) {
        try (INDArray indData = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            calcData(indData, kucoinKline, 0, 0);

            return net.rnnTimeStep(indData).ravel().toFloatVector();
        }
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int i) {
        BigDecimal data = kucoinKlines.get(i + PREDICT_DEEP).getClose()
            .subtract(kucoinKlines.get(i).getClose());

        float currencyDelta = kucoinKlines.get(i).getClose().movePointLeft(3).floatValue() * DELTA_PRICE;

        if (data.floatValue() > currencyDelta) {
            return  2;
        } else if (data.floatValue() < -currencyDelta) {
            return  0;
        } else {
            return 1;
        }
    }

    private static IntToDoubleFunction getPrabolaMinusFunction(EMAIndicator emaIndicator, int pos) {
        return i -> emaIndicator.getValue(pos - PREDICT_DEEP + i)
                .minus(emaIndicator.getValue(pos))
                .plus(DecimalNum.valueOf(i - PREDICT_DEEP).pow(2).multipliedBy(DecimalNum.valueOf(2)))
                .abs()
                .doubleValue();
    }

    private static IntToDoubleFunction getPrabolaPlusFunction(EMAIndicator emaIndicator, int pos) {
        return i -> emaIndicator.getValue(pos - PREDICT_DEEP + i)
                .minus(emaIndicator.getValue(pos))
                .minus(DecimalNum.valueOf(i - PREDICT_DEEP).pow(2).multipliedBy(DecimalNum.valueOf(2)))
                .abs()
                .doubleValue();
    }

    public static void calcData(INDArray indData, KucoinKline kucoinKline, int i, int y) {
        indData.putScalar(new int[]{i, 0, y},
                          kucoinKline.getClose().subtract(kucoinKline.getOpen()).floatValue() * 0.01);
        indData.putScalar(new int[]{i, 1, y}, kucoinKline.getVolume().floatValue() * 0.01);
        indData.putScalar(new int[]{i, 2, y},
                          kucoinKline.getClose().compareTo(kucoinKline.getOpen()) > 0 ?
                          kucoinKline.getHigh().subtract(kucoinKline.getClose()).floatValue() * 0.01 :
                          kucoinKline.getHigh().subtract(kucoinKline.getOpen()).floatValue() * 0.01);
        indData.putScalar(new int[]{i, 3, y},
                          kucoinKline.getClose().compareTo(kucoinKline.getOpen()) > 0 ?
                          kucoinKline.getOpen().subtract(kucoinKline.getLow()).floatValue() * 0.01 :
                          kucoinKline.getClose().subtract(kucoinKline.getLow()).floatValue() * 0.01);
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
