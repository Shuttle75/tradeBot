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
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.num.Num;

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

    public static float[] getPredict(KucoinKline kucoinKline, MultiLayerNetwork net,
                                     EMAIndicator emaF, EMAIndicator emaM, EMAIndicator emaS, RSIIndicator rsi) {
        try (INDArray indData = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            calcData(indData, kucoinKline, 0, 0, emaF, emaM, emaS, rsi);

            return net.rnnTimeStep(indData).ravel().toFloatVector();
        }
    }

    public static int getDelta(EMAIndicator emaIndicator, int pos) {
        Num data = emaIndicator.getValue(pos + PREDICT_DEEP)
                .minus(emaIndicator.getValue(pos));

        float currencyDelta = emaIndicator.getValue(pos).floatValue() * DELTA_PRICE;

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

    public static void calcData(INDArray indData, KucoinKline kucoinKline, int i, int y,
                                EMAIndicator emaF, EMAIndicator emaM, EMAIndicator emaS, RSIIndicator rsi) {
        indData.putScalar(new int[]{i, 0, y}, kucoinKline.getOpen().floatValue() * 0.000_01);
        indData.putScalar(new int[]{i, 1, y}, kucoinKline.getClose().floatValue() * 0.000_01);
        indData.putScalar(new int[]{i, 2, y}, kucoinKline.getHigh().floatValue() * 0.000_01);
        indData.putScalar(new int[]{i, 3, y}, kucoinKline.getLow().floatValue() * 0.000_01);
        indData.putScalar(new int[]{i, 4, y}, kucoinKline.getVolume().floatValue() * 0.01);
        indData.putScalar(new int[]{i, 5, y}, emaF.getValue(y).floatValue() * 0.000_01);
        indData.putScalar(new int[]{i, 6, y}, emaM.getValue(y).floatValue() * 0.000_01);
        indData.putScalar(new int[]{i, 7, y}, emaS.getValue(y).floatValue() * 0.000_01);
        indData.putScalar(new int[]{i, 8, y}, rsi.getValue(y).floatValue() * 0.01);
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
