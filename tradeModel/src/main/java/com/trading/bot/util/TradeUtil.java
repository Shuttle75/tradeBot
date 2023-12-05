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
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.*;
import java.util.function.IntToDoubleFunction;
import java.util.stream.IntStream;

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

    public static int getDelta(EMAIndicator emaIndicator, int pos) {
        double parabolaPlus = IntStream.range(0, 9)
                .mapToDouble(getPrabolaPlusFunction(emaIndicator, pos))
                .sum();
        double parabolaMinus = IntStream.range(0, 8)
                .mapToDouble(getPrabolaMinusFunction(emaIndicator, pos))
                .sum();

        float currencyDelta = emaIndicator.getValue(pos).floatValue() * DELTA_PRICE;

        if (parabolaPlus < currencyDelta) {
            return  2;
        } else if (parabolaMinus < currencyDelta) {
            return  0;
        } else {
            return 1;
        }
    }

    private static IntToDoubleFunction getPrabolaMinusFunction(EMAIndicator emaIndicator, int pos) {
        return i -> emaIndicator.getValue(pos - PREDICT_DEEP + i)
                .minus(emaIndicator.getValue(pos))
                .plus(DecimalNum.valueOf(i - PREDICT_DEEP).pow(2).multipliedBy(DecimalNum.valueOf(3)))
                .abs()
                .doubleValue();
    }

    private static IntToDoubleFunction getPrabolaPlusFunction(EMAIndicator emaIndicator, int pos) {
        return i -> emaIndicator.getValue(pos - PREDICT_DEEP + i)
                .minus(emaIndicator.getValue(pos))
                .minus(DecimalNum.valueOf(i - PREDICT_DEEP).pow(2).multipliedBy(DecimalNum.valueOf(3)))
                .abs()
                .doubleValue();
    }

    public static void calcData(INDArray indData, KucoinKline kucoinKline, int i, int y) {
        indData.putScalar(new int[]{i, 0, y},
                kucoinKline.getClose().subtract(kucoinKline.getOpen()).floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 1, y},
                kucoinKline.getClose().compareTo(kucoinKline.getOpen()) > 0 ?
                        kucoinKline.getHigh().subtract(kucoinKline.getClose()).floatValue() * NORMAL :
                        kucoinKline.getHigh().subtract(kucoinKline.getOpen()).floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 2, y},
                kucoinKline.getClose().compareTo(kucoinKline.getOpen()) > 0 ?
                        kucoinKline.getOpen().subtract(kucoinKline.getLow()).floatValue() * NORMAL :
                        kucoinKline.getClose().subtract(kucoinKline.getLow()).floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 3, y}, kucoinKline.getVolume().floatValue() * NORMAL);
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
