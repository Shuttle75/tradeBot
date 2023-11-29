package com.trading.bot.configuration;

import com.amazonaws.AmazonServiceException;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.ExchangeFactory;
import org.knowm.xchange.ExchangeSpecification;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.kucoin.KucoinExchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeries;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;

import java.io.File;
import java.io.IOException;
import java.time.*;
import java.time.temporal.ChronoUnit;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.util.TradeUtil.*;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min5;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final int INPUT_SIZE = 5;
    public static final int LAYER_SIZE = 200;
    public static final int OUTPUT_SIZE = 3;
    public static final int TRAIN_EXAMPLES = 28;
    public static final int TRAIN_KLINES = 288;
    public static final float DELTA_PERCENT = 16;
    public static final int RSI_INDICATOR = 25;
    public static final int FUTURE_PREDICT = 8;
    public static final int HISTORY_INDICATOR = 100;
    public static final float NORMAL = 0.01F;

    @Value("${model.bucket}")
    public String bucketName;
    @Value("${exchange.username}")
    public String exchangeUserName;
    @Value("${exchange.apikey}")
    public String exchangeApiKey;
    @Value("${exchange.secretkey}")
    public String exchangeSecretKey;
    @Value("${exchange.passphrase}")
    public String exchangePassphrase;
    public static final CurrencyPair CURRENCY_PAIR = new CurrencyPair("BTC", "USDT");

    @Bean
    public Exchange getXChangeExchange() {
        ExchangeSpecification exchangeSpecification = new ExchangeSpecification(KucoinExchange.class);

        exchangeSpecification.setUserName(exchangeUserName);
        exchangeSpecification.setApiKey(exchangeApiKey);
        exchangeSpecification.setSecretKey(exchangeSecretKey);
        exchangeSpecification.setExchangeSpecificParametersItem("passphrase", exchangePassphrase);

        return ExchangeFactory.INSTANCE.createExchange(exchangeSpecification);
    }

    @Bean
    public MultiLayerConfiguration getConfig() {
        logger.info("Build model....");
        return new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(new LSTM.Builder().activation(Activation.TANH).nIn(INPUT_SIZE).nOut(LAYER_SIZE).build())
                .layer(new LSTM.Builder().activation(Activation.TANH).nOut(LAYER_SIZE).build())
                .layer(new LSTM.Builder().activation(Activation.TANH).nOut(LAYER_SIZE).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nOut(OUTPUT_SIZE).build())
                .build();
    }

    @Bean
    public MultiLayerNetwork getModel(Exchange exchange, MultiLayerConfiguration config) throws IOException {
        final String keyName = CURRENCY_PAIR.base + ".zip";
        final String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), keyName);
        final LocalDateTime now = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.HOURS).minusHours(2);
        final AmazonS3 s3 = AmazonS3ClientBuilder.standard().withRegion(Regions.EU_CENTRAL_1).build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        try (INDArray indData = Nd4j.zeros(TRAIN_EXAMPLES, INPUT_SIZE, TRAIN_KLINES);
             INDArray indLabels = Nd4j.zeros(TRAIN_EXAMPLES, OUTPUT_SIZE, TRAIN_KLINES)) {

            int i = TRAIN_EXAMPLES - 1;
            while (i >= 0) {
                BarSeries barSeries = new BaseBarSeries();
                ClosePriceIndicator closePrice = new ClosePriceIndicator(barSeries);
                RSIIndicator rsiIndicator = new RSIIndicator(closePrice, RSI_INDICATOR);

                LocalDateTime startDate = now.minusSeconds(
                        i * (long) TRAIN_KLINES * min5.getSeconds()
                                + TRAIN_KLINES * min5.getSeconds()
                                + HISTORY_INDICATOR * min5.getSeconds());
                LocalDateTime endDate = now.minusSeconds(
                        i * (long) TRAIN_KLINES * min5.getSeconds()
                                - FUTURE_PREDICT * min5.getSeconds());

                List<KucoinKline> kucoinKlines =
                        getKucoinKlines(
                                exchange,
                                startDate.toEpochSecond(ZoneOffset.UTC),
                                endDate.toEpochSecond(ZoneOffset.UTC),
                                min5);
                Collections.reverse(kucoinKlines);

                logger.info("startDate {} endDate {}", startDate, endDate);

                kucoinKlines.forEach(kucoinKline -> loadBarSeries(barSeries, kucoinKline));

                for (int y = 0; y < HISTORY_INDICATOR + TRAIN_KLINES; y++) {
                    if (y >= HISTORY_INDICATOR) {
                        calcData(indData, kucoinKlines.get(y), i, y - HISTORY_INDICATOR, rsiIndicator.getValue(y));
                        indLabels.putScalar(new int[]{i, getDelta(rsiIndicator, y), y - HISTORY_INDICATOR}, 1);
                    }
                }

                i--;
            }

            net.fit(indData, indLabels);
            while (net.score() > 1D) {
                net.fit(indData, indLabels);
            }
        }

        try {
            String fileName = CURRENCY_PAIR.base + ".zip";
            net.save(new File(path));

            s3.putObject(bucketName, fileName, new File(path));
        } catch (AmazonServiceException e) {
            logger.error(e.getErrorMessage());
        }

        return net;
    }


}
