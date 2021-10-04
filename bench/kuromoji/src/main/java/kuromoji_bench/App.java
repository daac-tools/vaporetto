package kuromoji_bench;

import com.atilika.kuromoji.ipadic.Token;
import com.atilika.kuromoji.ipadic.Tokenizer;
import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;
import java.time.Instant;
import java.time.Duration;

public class App {
    public static void main(String[] args) {
        Tokenizer tokenizer = new Tokenizer();
        Scanner input = new Scanner(System.in);
        Instant start = Instant.now();
        while (input.hasNext()) {
            List<Token> tokens = tokenizer.tokenize(input.nextLine());
            List<String> words = new ArrayList<String>();
            for (Token token : tokens) {
                words.add(token.getSurface());
            }
            System.out.println(String.join(" ", words));
        }
        Instant finish = Instant.now();
        double timeElapsed = (double) Duration.between(start, finish).toMillis() / 1000;
        System.err.println("Elapsed-kuromoji: " + timeElapsed + " [sec]");
    }
}
