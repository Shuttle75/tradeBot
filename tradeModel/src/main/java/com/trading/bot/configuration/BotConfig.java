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
    public static final int INPUT_SIZE = 2;
    public static final int LAYER_SIZE = 128;
    public static final int OUTPUT_SIZE = 5;
    public static final int TRAIN_EXAMPLES = 4;
    public static final int TRAIN_MINUTES = 120;
    public static final int PREDICT_DEEP = 2;
    public static final int CURRENCY_DELTA = 20;
    public static final int NET_FIT_ITERATIONS = 4000;
    public static final int NORMAL = 3;


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
                        .activation(Activation.TANH)
                        .nIn(INPUT_SIZE).nOut(LAYER_SIZE).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).dropOut(0.3)
                        .nIn(LAYER_SIZE).nOut(OUTPUT_SIZE).build())
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

            int iQuery = 0;
            int iTrain = 0;
            while (iTrain < TRAIN_EXAMPLES) {
                List<KucoinKline> kucoinKlines =
                        getKucoinKlines(
                                exchange,
                                now.minusMinutes(iQuery * (long) TRAIN_MINUTES + TRAIN_MINUTES + PREDICT_DEEP).toEpochSecond(ZoneOffset.UTC),
                                now.minusMinutes(iQuery * (long) TRAIN_MINUTES).toEpochSecond(ZoneOffset.UTC));
                Collections.reverse(kucoinKlines);
                iQuery++;

                if (kucoinKlines.get(0).getClose().compareTo(kucoinKlines.get(kucoinKlines.size() - 1).getClose()) < 0) {
                    continue;
                }

                for (int y = 0; y < TRAIN_MINUTES; y++) {
                    indData.putScalar(new int[]{iTrain, 0, y},
                            kucoinKlines.get(y).getClose()
                                    .subtract(kucoinKlines.get(y).getOpen()).movePointLeft(NORMAL).floatValue());
                    indData.putScalar(new int[]{iTrain, 1, y},
                            kucoinKlines.get(y).getVolume().movePointLeft(NORMAL).floatValue());
                    indLabels.putScalar(new int[]{iTrain, getDelta(kucoinKlines, y), y}, 1);
                }
                iTrain++;
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
