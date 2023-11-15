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
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.util.TradeUtil.*;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final int INPUT_SIZE = 3;
    public static final int LAYER_SIZE = 128;
    public static final int OUTPUT_SIZE = 8;
    public static final int TRAIN_EXAMPLES = 120;
    public static final int TRAIN_MINUTES = 24;
    public static final int CURRENCY_DELTA = 10;
    public static final int PREDICT_DEEP = 4;
    public static final int NET_FIT_ITERATIONS = 1600;

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
                .updater(new Nadam())
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(new LSTM.Builder()
                        .activation(Activation.TANH).nIn(INPUT_SIZE).nOut(LAYER_SIZE).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(LAYER_SIZE).nOut(OUTPUT_SIZE).build())
                .build();
    }

    @Bean
    public MultiLayerNetwork getModel(Exchange exchange, MultiLayerConfiguration config) throws IOException {
        final LocalDateTime now = LocalDateTime.now(ZoneOffset.UTC);

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        try (INDArray indData = Nd4j.zeros(TRAIN_EXAMPLES, INPUT_SIZE, TRAIN_MINUTES);
             INDArray indLabels = Nd4j.zeros(TRAIN_EXAMPLES, OUTPUT_SIZE, TRAIN_MINUTES)) {

            for (int i = 0; i < TRAIN_EXAMPLES ; i++) {
                List<KucoinKline> kucoinKlines =
                        getKucoinKlines(
                                exchange,
                                now.minusMinutes(i * (long) TRAIN_MINUTES + TRAIN_MINUTES + PREDICT_DEEP).toEpochSecond(ZoneOffset.UTC),
                                now.minusMinutes(i * (long) TRAIN_MINUTES).toEpochSecond(ZoneOffset.UTC));
                Collections.reverse(kucoinKlines);

                for (int y = 0; y < TRAIN_MINUTES; y++) {
                    indData.putScalar(new int[]{i, 0, y},
                            kucoinKlines.get(y).getClose()
                                    .subtract(kucoinKlines.get(y).getOpen()).floatValue());
                    indData.putScalar(new int[]{i, 1, y},
                            kucoinKlines.get(y).getVolume().floatValue());
                    indData.putScalar(new int[]{i, 2, y},
                            kucoinKlines.get(y).getClose().compareTo(kucoinKlines.get(y).getOpen()) > 0 ?
                                    kucoinKlines.get(y).getOpen().subtract(kucoinKlines.get(y).getLow()).floatValue() :
                                    kucoinKlines.get(y).getClose().subtract(kucoinKlines.get(y).getLow()).floatValue());
                    indLabels.putScalar(new int[]{i, getDelta(kucoinKlines, y), y}, 1);
                }
            }

            for (int i = 0; i < NET_FIT_ITERATIONS; i++) {
                net.fit(indData, indLabels);
            }
        }

        final AmazonS3 s3 = AmazonS3ClientBuilder.standard().withRegion(Regions.EU_CENTRAL_1).build();

        try {
            String fileName = CURRENCY_PAIR.base + ".zip";
            String path = FilenameUtils.concat(
                    System.getProperty("java.io.tmpdir"), fileName);
            net.save(new File(path));

            s3.putObject(bucketName, fileName, new File(path));
        } catch (AmazonServiceException e) {
            logger.error(e.getErrorMessage());
        }

        return net;
    }
}
