package com.trading.bot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableScheduling
@SpringBootApplication
public class TradeRobotApplication {

    public static void main(String[] args) {
        SpringApplication.run(TradeRobotApplication.class, args);
    }

}
