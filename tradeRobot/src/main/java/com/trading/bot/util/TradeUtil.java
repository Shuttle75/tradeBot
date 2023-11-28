package com.trading.bot.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.KlineIntervalType;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.BarSeries;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.num.Num;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.*;

import static com.trading.bot.configuration.BotConfig.*;

public class TradeUtil {

    private TradeUtil() {
    }

    public static List<KucoinKline> getKucoinKlines(Exchange exchange, long startDate, long endDate, KlineIntervalType intervalType) throws IOException {
        return ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, intervalType);
    }

    public static float[] getPredict(KucoinKline kucoinKline, MultiLayerNetwork net, RSIIndicator rsi) {
        try (INDArray indData = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            calcData(indData, kucoinKline, 0, 0, rsi);

            return net.rnnTimeStep(indData).ravel().toFloatVector();
        }
    }

    public static int getDelta(RSIIndicator rsiIndicator, int i) {
        Num delta = rsiIndicator.getValue(i + FUTURE_PREDICT).minus(rsiIndicator.getValue(i));

        if (delta.floatValue() > DELTA_PERCENT) {
            return  2;
        } else if (delta.floatValue() < -DELTA_PERCENT) {
            return  0;
        } else {
            return 1;
        }
    }

    public static void calcData(INDArray indData, KucoinKline kucoinKline, int i, int y, RSIIndicator rsiIndicator) {
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
        indData.putScalar(new int[]{i, 4, y}, rsiIndicator.getValue(y).floatValue() * NORMAL);
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
