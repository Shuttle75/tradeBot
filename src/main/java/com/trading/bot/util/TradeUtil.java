package com.trading.bot.util;

public class TradeUtil {

    public static String printRates(float[][] floatResult) {
        return String.format("%.2f", floatResult[0][0]) + " " +
                String.format("%.2f", floatResult[0][1]) + " " +
                String.format("%.2f", floatResult[0][2]) + " " +
                String.format("%.2f", floatResult[0][3]) + " | " +
                String.format("%.2f", floatResult[0][4]) + " " +
                String.format("%.2f", floatResult[0][5]) + " " +
                String.format("%.2f", floatResult[0][6]) + " " +
                String.format("%.2f", floatResult[0][7]);
    }
}
