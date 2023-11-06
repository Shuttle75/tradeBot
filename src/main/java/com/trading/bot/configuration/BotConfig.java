package com.trading.bot.configuration;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.ExchangeFactory;
import org.knowm.xchange.ExchangeSpecification;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.kucoin.KucoinExchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min1;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final long MINUS_DAYS = 1;
    public static final int TRAIN_CYCLES = 1200;
    public static final int OUTPUT_SIZE = 8;
    public static final int TRAIN_DEEP = 12;
    public static final int PREDICT_DEEP = 2;
    public static final int CURRENCY_DELTA = 10;

    public static final CurrencyPair CURRENCY_PAIR = new CurrencyPair("BTC", "USDT");

    @Bean
    public Exchange getXChangeExchange() {
        ExchangeSpecification exchangeSpecification = new ExchangeSpecification(KucoinExchange.class);

        exchangeSpecification.setUserName("alexey.osadchiy@gmail.com");
        exchangeSpecification.setApiKey("65380094ab6f110001a5a383");
        exchangeSpecification.setSecretKey("58da9817-38b5-4f89-85cd-718e54edd8fe");
        exchangeSpecification.setExchangeSpecificParametersItem("Use_Sandbox", false);
        exchangeSpecification.setExchangeSpecificParametersItem("passphrase", "cassandre");

        return ExchangeFactory.INSTANCE.createExchange(exchangeSpecification);
    }

    @Bean
    public MultiLayerConfiguration getConfig() {
        logger.info("Build model....");
        return new NeuralNetConfiguration.Builder()
                .seed(6)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(TRAIN_DEEP).nOut(4096)
                        .build())
                .layer(new DenseLayer.Builder().nIn(4096).nOut(512)
                        .build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(64)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                        .nIn(64).nOut(OUTPUT_SIZE).build())
                .build();
    }

    @Bean
    public MultiLayerNetwork getModel(Exchange exchange, MultiLayerConfiguration config) throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.DAYS)
                .minusDays(MINUS_DAYS + 1)
                .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.DAYS)
                .minusDays(MINUS_DAYS)
                .toEpochSecond(ZoneOffset.UTC);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate);

        float[][] floatData = new float[TRAIN_CYCLES][TRAIN_DEEP];
        int[][] intLabels = new int[TRAIN_CYCLES][OUTPUT_SIZE];
        for (int i = 0; i < TRAIN_CYCLES; i++) {
            for (int y = 0; y < TRAIN_DEEP; y++) {
                floatData[i][y] = kucoinKlines.get(i + y + PREDICT_DEEP).getClose()
                        .subtract(kucoinKlines.get(i + y + PREDICT_DEEP).getOpen())
                        .multiply(kucoinKlines.get(i + y + PREDICT_DEEP).getVolume()).floatValue();
            }

            intLabels[i][getDelta(kucoinKlines, i)] = 1;
        }

        INDArray indData = Nd4j.create(floatData);
        INDArray indLabels = Nd4j.create(intLabels);

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));



        for (int i = 0; i < 1000; i++) {
            model.fit(indData, indLabels);
        }
        return model;
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int i) {
        BigDecimal data0 = kucoinKlines.get(i).getClose().subtract(kucoinKlines.get(i).getOpen());
        BigDecimal data1 = kucoinKlines.get(i + 1).getClose().subtract(kucoinKlines.get(i + 1).getOpen());
        BigDecimal data2 = kucoinKlines.get(i + 2).getOpen().subtract(kucoinKlines.get(i + 2).getClose());
        BigDecimal data3 = kucoinKlines.get(i + 3).getOpen().subtract(kucoinKlines.get(i + 3).getClose());
        int delta = (data0.add(data1).add(data2).add(data3).intValue() + OUTPUT_SIZE * CURRENCY_DELTA / 2) / CURRENCY_DELTA;

        delta = Math.max(delta, 0);
        delta = Math.min(delta, 7);
        return delta;
    }

    public static List<KucoinKline> getKucoinKlines(Exchange exchange, long startDate, long endDate) throws IOException {
        List<KucoinKline> kucoinKlines = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, min1);

 //       reduceKucoinKlines(kucoinKlines);

        return kucoinKlines;
    }

    private static void reduceKucoinKlines(List<KucoinKline> kucoinKlines) {
        kucoinKlines.stream()
                .sorted(Comparator.comparing(KucoinKline::getVolume))
                .limit(400)
                .forEach(lowValueKline -> {
                    AtomicInteger index = new AtomicInteger();
                    kucoinKlines.stream()
                            .filter(kucoinKline -> kucoinKline.getTime().equals(lowValueKline.getTime()))
                            .findAny()
                            .ifPresent(kucoinKline -> index.set(kucoinKlines.indexOf(kucoinKline)));

                    kucoinKlines.indexOf(lowValueKline);
                    KucoinKline kucoinKlineBefore = index.get() > 0 ? kucoinKlines.get(index.get() - 1) : null;
                    KucoinKline kucoinKlineAfter = index.get() < kucoinKlines.size() - 1 ? kucoinKlines.get(index.get() + 1) : null;;

                    if (index.get() >= 1
                            && kucoinKlineBefore.getVolume().compareTo(kucoinKlineAfter.getVolume()) < 0) {
                        kucoinKlines.set(index.get() + 1, createBeforeKucoinKline(lowValueKline, kucoinKlines.get(index.get() + 1)));
                        kucoinKlines.remove(index.get());
                    } else {
                        kucoinKlines.set(index.get() + 1, createAfterKucoinKline(lowValueKline, kucoinKlines.get(index.get() + 1)));
                        kucoinKlines.remove(index.get());
                    }
                });
    }
    private static KucoinKline createBeforeKucoinKline(KucoinKline lowValueKline, KucoinKline kucoinKlineBefore) {
        return new KucoinKline(
                lowValueKline.getPair(),
                lowValueKline.getIntervalType(),
                new Object[]{
                        kucoinKlineBefore.getTime(),
                        kucoinKlineBefore.getOpen().floatValue(),
                        lowValueKline.getClose().floatValue(),
                        kucoinKlineBefore.getHigh().compareTo(lowValueKline.getHigh()) > 0 ?
                                kucoinKlineBefore.getHigh().floatValue() :
                                lowValueKline.getHigh().floatValue(),
                        kucoinKlineBefore.getLow().compareTo(lowValueKline.getHigh()) < 0 ?
                                kucoinKlineBefore.getLow().floatValue() :
                                lowValueKline.getLow().floatValue(),
                        kucoinKlineBefore.getVolume().floatValue() + lowValueKline.getVolume().floatValue(),
                        kucoinKlineBefore.getAmount().floatValue() + lowValueKline.getAmount().floatValue()}
        );
    }

    private static KucoinKline createAfterKucoinKline(KucoinKline lowValueKline, KucoinKline kucoinKlineAfter) {
        return new KucoinKline(
                lowValueKline.getPair(),
                lowValueKline.getIntervalType(),
                new Object[]{
                        kucoinKlineAfter.getTime(),
                        lowValueKline.getOpen().floatValue(),
                        kucoinKlineAfter.getClose().floatValue(),
                        kucoinKlineAfter.getHigh().compareTo(lowValueKline.getHigh()) > 0 ?
                                kucoinKlineAfter.getHigh().floatValue() :
                                lowValueKline.getHigh().floatValue(),
                        kucoinKlineAfter.getLow().compareTo(lowValueKline.getHigh()) < 0 ?
                                kucoinKlineAfter.getLow().floatValue() :
                                lowValueKline.getLow().floatValue(),
                        kucoinKlineAfter.getVolume().floatValue() + lowValueKline.getVolume().floatValue(),
                        kucoinKlineAfter.getAmount().floatValue() + lowValueKline.getAmount().floatValue()}
        );
    }
}
