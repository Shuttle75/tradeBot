package com.trading.bot.util;

import com.bybit.api.client.domain.market.response.kline.MarketKlineEntry;
import com.bybit.api.client.domain.market.response.openInterests.OpenInterestEntry;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.*;

import static com.trading.bot.configuration.BotConfig.*;

public class TradeUtil {

    private TradeUtil() {
    }

    public static float[] getPredict(List<MarketKlineEntry> klines, List<OpenInterestEntry> openInterest, MultiLayerNetwork net) {
        try (INDArray indData = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            calcData(indData, 0, klines, openInterest, 0);

            return net.rnnTimeStep(indData).ravel().toFloatVector();
        }
    }

    public static int getDelta(List<MarketKlineEntry> klines, int i) {
        BigDecimal minValue = klines.subList(i, i + PREDICT_DEEP * 2).stream()
            .map(entry -> new BigDecimal(entry.getOpenPrice()))
            .min(BigDecimal::compareTo)
            .orElse(BigDecimal.ZERO);

        BigDecimal maxValue = klines.subList(i, i + PREDICT_DEEP * 2).stream()
            .map(entry -> new BigDecimal(entry.getOpenPrice()))
            .max(BigDecimal::compareTo)
            .orElse(BigDecimal.ZERO);

        if (minValue.equals(new BigDecimal(klines.get(i + PREDICT_DEEP).getOpenPrice()))) {
            return 2;
        } else if (maxValue.equals(new BigDecimal(klines.get(i + PREDICT_DEEP).getOpenPrice()))) {
            return 0;
        } else {
            return 1;
        }
    }

    public static void calcData(INDArray indData, int i, List<MarketKlineEntry> kline, List<OpenInterestEntry> openInterest, int y) {
        float open = Float.parseFloat(kline.get(y + PREDICT_DEEP).getOpenPrice());
        float high = Float.parseFloat(kline.get(y + PREDICT_DEEP).getHighPrice());
        float low = Float.parseFloat(kline.get(y + PREDICT_DEEP).getLowPrice());
        float close = Float.parseFloat(kline.get(y + PREDICT_DEEP).getClosePrice());
        float value = Float.parseFloat(kline.get(y + PREDICT_DEEP).getTurnover());
        float day = Instant.ofEpochMilli(kline.get(y + PREDICT_DEEP).getStartTime()).atOffset(ZoneOffset.UTC).getDayOfWeek().getValue() * 0.12F;
        float hour = Instant.ofEpochMilli(kline.get(y + PREDICT_DEEP).getStartTime()).atOffset(ZoneOffset.UTC).getHour() * 0.005F;
        float prevInterest = Float.parseFloat(openInterest.get(y + PREDICT_DEEP - 1).getOpenInterest());
        float interest = Float.parseFloat(openInterest.get(y + PREDICT_DEEP).getOpenInterest());

        indData.putScalar(new int[]{i, 0, y}, (close - open) / close * 10F);
        indData.putScalar(new int[]{i, 1, y}, value / 10_000_000F);
        indData.putScalar(new int[]{i, 2, y}, close > open ? (high - close) / close * 10F : (high - open) / close * 10F);
        indData.putScalar(new int[]{i, 3, y}, close > open ? (open - low) / close * 10F : (close - low) / close * 10F);
        indData.putScalar(new int[]{i, 4, y}, day + hour);
        indData.putScalar(new int[]{i, 5, y}, (prevInterest - interest) * 0.00001F);
    }

    public static String printRates(float[] floatResult) {
        return String.format("%.2f", floatResult[0]) + " " +
                String.format("%.2f", floatResult[1]) + " " +
                String.format("%.2f", floatResult[2]);
    }
}
