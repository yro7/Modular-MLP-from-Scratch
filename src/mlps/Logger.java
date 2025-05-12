package mlps;


import static mlps.Logger.LogLevel.DEBUG;
import static mlps.Logger.LogLevel.WARN;

public class Logger {

    public enum LogLevel {
        NONE,
        WARN,
        DEBUG;

        public boolean test() {
            if(level == NONE) return false;

            if(level == DEBUG) return true;

            if(level == WARN && this == DEBUG) return false;

            return false;
        }

    }

    public static final LogLevel level = LogLevel.DEBUG;

    public static void log(String string, LogLevel logLevel) {
        if(logLevel.test()) {

            System.out.println(string);

        }
    }

    public static void warn(String string){
        log(string, WARN);
    }

    public static void debug(String string){
        log(string, DEBUG);
    }

    public static void printIfDebug() {
    }
}
