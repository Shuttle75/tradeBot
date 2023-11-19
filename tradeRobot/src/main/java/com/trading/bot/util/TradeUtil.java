package com.trading.bot.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;

import static com.trading.bot.configuration.BotConfig.*;
import static java.lang.Math.round;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min1;

public class TradeUtil {

    private TradeUtil() {
    }

    public static List<KucoinKline> getKucoinKlines(Exchange exchange, long startDate, long endDate) throws IOException {
        return ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, min1);
    }

    public static float[] getOneMinutePredict(KucoinKline kucoinKline, MultiLayerNetwork net) {
        try (INDArray nextInput = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            nextInput.putScalar(new int[]{0, 0, 0},
                    kucoinKline.getClose()
                            .subtract(kucoinKline.getOpen()).movePointLeft(NORMAL).floatValue());
            nextInput.putScalar(new int[]{0, 1, 0},
                    kucoinKline.getVolume().movePointLeft(NORMAL).floatValue());

            return net.rnnTimeStep(nextInput).ravel().toFloatVector();
        }
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int pos) {
        BigDecimal data = kucoinKlines.get(pos + PREDICT_DEEP).getClose()
                .subtract(kucoinKlines.get(pos).getClose());

        int delta = round((data.floatValue() + CURRENCY_DELTA * 2) / CURRENCY_DELTA);

        delta = Math.max(delta, 0);
        delta = Math.min(delta, OUTPUT_SIZE - 1);
        return delta;
    }

    public static void calcData(List<KucoinKline> kucoinKlines, int i, int y, INDArray indData, INDArray indLabels) {
        indData.putScalar(new int[]{i, 0, y},
                kucoinKlines.get(y).getClose()
                        .subtract(kucoinKlines.get(y).getOpen()).movePointLeft(NORMAL).floatValue());
        indData.putScalar(new int[]{i, 1, y},
                kucoinKlines.get(y).getClose()
                        .compareTo(kucoinKlines.get(y).getOpen()) > 0 ?
                        kucoinKlines.get(y).getHigh()
                                .subtract(kucoinKlines.get(y).getClose()).movePointLeft(NORMAL).floatValue() :
                        kucoinKlines.get(y).getHigh()
                                .subtract(kucoinKlines.get(y).getOpen()).movePointLeft(NORMAL).floatValue());
        indData.putScalar(new int[]{i, 2, y},
                kucoinKlines.get(y).getClose()
                        .compareTo(kucoinKlines.get(y).getOpen()) > 0 ?
                        kucoinKlines.get(y).getOpen()
                                .subtract(kucoinKlines.get(y).getLow()).movePointLeft(NORMAL).floatValue() :
                        kucoinKlines.get(y).getClose()
                                .subtract(kucoinKlines.get(y).getLow()).movePointLeft(NORMAL).floatValue());
        indData.putScalar(new int[]{i, 3, y},
                kucoinKlines.get(y).getVolume().movePointLeft(NORMAL).floatValue());
        indLabels.putScalar(new int[]{i, getDelta(kucoinKlines, y), y}, 1);
    }

    public static String printRates(float[] floatResult) {
        return String.format("%.2f", floatResult[0]) + " " +
                String.format("%.2f", floatResult[1]) + "  " +
                String.format("%.2f", floatResult[2]) + "  " +
                String.format("%.2f", floatResult[3]) + " " +
                String.format("%.2f", floatResult[4]);
    }
}
