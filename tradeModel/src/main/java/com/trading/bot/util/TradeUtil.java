package com.trading.bot.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.KlineIntervalType;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.BarSeries;
import org.ta4j.core.num.Num;

import java.io.IOException;
import java.math.BigDecimal;
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

    public static float[] getOneMinutePredict(KucoinKline kucoinKline, MultiLayerNetwork net) {
        try (INDArray indData = Nd4j.zeros(1, INPUT_SIZE, 1)) {

            calcData(indData, kucoinKline, 0, 0, null, null, null, null);

            return net.rnnTimeStep(indData).ravel().toFloatVector();
        }
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int i) {
        BigDecimal deltaUp = kucoinKlines.get(i + PREDICT_UP).getClose()
                .subtract(kucoinKlines.get(i).getClose());
        BigDecimal deltaDown = kucoinKlines.get(i + PREDICT_DOWN).getClose()
                .subtract(kucoinKlines.get(i).getClose());

        float priceForMilli = kucoinKlines.get(i).getClose().movePointLeft(3).floatValue();

        if (deltaUp.floatValue() > priceForMilli * DELTA_PRICE) {
            return  2;
        } else if (deltaDown.floatValue() < - priceForMilli) {
            return  0;
        } else {
            return 1;
        }
    }

    public static void calcData(INDArray indData, KucoinKline kucoinKline, int i, int y,
                                Num emaF, Num emaM, Num emaS, Num rsi) {
        indData.putScalar(new int[]{i, 0, y},
                kucoinKline.getClose().subtract(kucoinKline.getOpen()).floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 1, y},
                kucoinKline.getClose()
                        .compareTo(kucoinKline.getOpen()) > 0 ?
                        kucoinKline.getHigh().subtract(kucoinKline.getClose()).floatValue() * NORMAL :
                        kucoinKline.getHigh().subtract(kucoinKline.getOpen()).floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 2, y},
                kucoinKline.getClose()
                        .compareTo(kucoinKline.getOpen()) > 0 ?
                        kucoinKline.getOpen().subtract(kucoinKline.getLow()).floatValue() * NORMAL :
                        kucoinKline.getClose().subtract(kucoinKline.getLow()).floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 3, y}, kucoinKline.getVolume().floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 4, y}, emaF.floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 5, y}, emaM.floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 6, y}, emaS.floatValue() * NORMAL);
        indData.putScalar(new int[]{i, 7, y}, rsi.floatValue() * NORMAL);
    }

    public static void loadBarSeries(BarSeries barSeries, List<KucoinKline> kucoinKlines, int y) {
        barSeries.addBar(Duration.ofMinutes(5L),
                ZonedDateTime.ofInstant(Instant.ofEpochSecond(kucoinKlines.get(y).getTime()), ZoneOffset.UTC),
                kucoinKlines.get(y).getOpen(),
                kucoinKlines.get(y).getHigh(),
                kucoinKlines.get(y).getHigh(),
                kucoinKlines.get(y).getLow(),
                kucoinKlines.get(y).getVolume());
    }

    public static String printRates(float[] floatResult) {
        return String.format("%.2f", floatResult[0]) + " " +
                String.format("%.2f", floatResult[1]) + " " +
                String.format("%.2f", floatResult[2]);
    }
}
