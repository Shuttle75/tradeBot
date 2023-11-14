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
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static com.trading.bot.util.TradeUtil.*;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final int OUTPUT_SIZE = 8;
    public static final int TRAIN_DEEP = 60;
    public static final int PREDICT_DEEP = 2;
    public static final int CURRENCY_DELTA = 20;

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
                        .activation(Activation.TANH).nIn(4).nOut(128).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(128).nOut(OUTPUT_SIZE).build())
                .build();
    }

    @Bean
    public MultiLayerNetwork getModel(Exchange exchange, MultiLayerConfiguration config) throws IOException {
        LocalDateTime startDate = LocalDateTime.now(ZoneOffset.UTC).truncatedTo(ChronoUnit.DAYS);
        LocalDateTime endDate = LocalDateTime.now(ZoneOffset.UTC).truncatedTo(ChronoUnit.DAYS).plusDays(1);
        final List<KucoinKline> kucoinKlines = new ArrayList<>();

        for (long i = 0; i < 11; i++) {
            kucoinKlines.addAll(
                    getKucoinKlines(
                            exchange,
                            startDate.minusDays(i).toEpochSecond(ZoneOffset.UTC),
                            endDate.minusDays(i).toEpochSecond(ZoneOffset.UTC)));
        }

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        try (INDArray indData = Nd4j.zeros(240, 4, TRAIN_DEEP);
             INDArray indLabels = Nd4j.zeros(240, OUTPUT_SIZE, TRAIN_DEEP)) {

            for (int i = 239; i >= 0 ; i--) {
                for (int y = TRAIN_DEEP - 1; y >= 0 ; y--) {
                    int pos = i + y + PREDICT_DEEP;
                    indData.putScalar(new int[]{i, 0, y},
                            kucoinKlines.get(pos).getClose().subtract(kucoinKlines.get(pos).getOpen()).floatValue());
                    indData.putScalar(new int[]{i, 1, y},
                            kucoinKlines.get(pos).getClose().compareTo(kucoinKlines.get(pos).getOpen()) > 0 ?
                                    kucoinKlines.get(pos).getHigh().subtract(kucoinKlines.get(pos).getClose()).floatValue() :
                                    kucoinKlines.get(pos).getHigh().subtract(kucoinKlines.get(pos).getOpen()).floatValue());
                    indData.putScalar(new int[]{i, 2, y},
                            kucoinKlines.get(pos).getClose().compareTo(kucoinKlines.get(pos).getOpen()) > 0 ?
                                    kucoinKlines.get(pos).getOpen().subtract(kucoinKlines.get(pos).getLow()).floatValue() :
                                    kucoinKlines.get(pos).getClose().subtract(kucoinKlines.get(pos).getLow()).floatValue());
                    indData.putScalar(new int[]{i, 3, y},
                            kucoinKlines.get(pos).getVolume().floatValue());
                    indLabels.putScalar(new int[]{i, getDelta(kucoinKlines, i * TRAIN_DEEP, y), y}, 1);
                }
            }

            for (int i = 0; i < 3200; i++) {
                model.fit(indData, indLabels);
                TimeUnit.MILLISECONDS.sleep(500);    // Cooling CPU
            }
        } catch (InterruptedException e) {
            logger.error(e.getMessage());
            Thread.currentThread().interrupt();
        }

        final AmazonS3 s3 = AmazonS3ClientBuilder.standard()
                .withRegion(Regions.EU_CENTRAL_1)
                .build();

        try {
            String fileName = CURRENCY_PAIR.base + ".zip";
            String path = FilenameUtils.concat(
                    System.getProperty("java.io.tmpdir"), fileName);
            model.save(new File(path));

            s3.putObject(bucketName, fileName, new File(path));
        } catch (AmazonServiceException e) {
            logger.error(e.getErrorMessage());
        }

        return model;
    }
}
