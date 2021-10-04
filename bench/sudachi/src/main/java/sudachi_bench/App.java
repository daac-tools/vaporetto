package sudachi_bench;

import java.io.IOException;
import com.worksap.nlp.sudachi.Tokenizer;
import com.worksap.nlp.sudachi.Dictionary;
import com.worksap.nlp.sudachi.DictionaryFactory;
import com.worksap.nlp.sudachi.Morpheme;
import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;
import java.time.Instant;
import java.time.Duration;
import java.nio.file.Paths;
import java.nio.file.Files;

public class App {
    public static void main(String[] args) throws IOException {
        String settings = Files.readString(Paths.get("sudachi.json"));
        Scanner input = new Scanner(System.in);
        try (Dictionary dict = new DictionaryFactory().create(settings)) {
            Tokenizer tokenizer = dict.create();
            Instant start = Instant.now();
            while (input.hasNext()) {
                List<Morpheme> tokens = tokenizer.tokenize(Tokenizer.SplitMode.C, input.nextLine());
                List<String> words = new ArrayList<String>();
                for (Morpheme token : tokens) {
                    words.add(token.surface());
                }
                System.out.println(String.join(" ", words));
            }
            Instant finish = Instant.now();
            double timeElapsed = (double) Duration.between(start, finish).toMillis() / 1000;
            System.err.println("Elapsed-sudachi: " + timeElapsed + " [sec]");
        }
    }
}
