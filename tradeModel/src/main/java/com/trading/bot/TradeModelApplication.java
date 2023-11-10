package com.trading.bot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableScheduling
@SpringBootApplication
public class TradeModelApplication {

    public static void main(String[] args) {
        SpringApplication.run(TradeModelApplication.class, args);
    }

}
