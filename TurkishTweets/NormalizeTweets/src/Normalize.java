import java.io.*;
import java.nio.file.*;
import java.util.*;
import zemberek.morphology.TurkishMorphology;
import zemberek.normalization.TurkishSentenceNormalizer;
import zemberek.tokenization.TurkishSentenceExtractor;
import com.vdurmont.emoji.EmojiParser;

public class Normalize {
    public static void main(String[] args) throws IOException {
        if(args.length != 1) {
            System.err.println("Format: Normalize <input path>");
        }
        boolean isTest = true;
        ArrayList<Tweet> tweets = loadEntries(Paths.get(args[0]), isTest);

        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
        TurkishSentenceExtractor extractor = TurkishSentenceExtractor.DEFAULT;
        TurkishSentenceNormalizer normalizer = new TurkishSentenceNormalizer(morphology,
                Paths.get("lib/normalization_lookup"), Paths.get("lib/normalization_lookup/lm.2gram.slm"));

        for(Tweet tweet : tweets) {
            tweet.tweet = normalizer.normalize(detweetify(tweet.tweet));
//            System.out.println(tweet.tweet);
        }
        String output_location = isTest ? "lib/processed_test.tsv" : "lib/processed_training.tsv";

        Files.deleteIfExists(Paths.get(output_location));
        Files.createFile(Paths.get(output_location));
        String header = isTest ? "id\ttweet\n" : "id\ttweet\tsubtask_a\n";
        Files.write(Paths.get(output_location), header.getBytes(), StandardOpenOption.APPEND);
        tweets.forEach(tweet -> {
            try {
                Files.write(Paths.get(output_location), tweet.toString().getBytes(), StandardOpenOption.APPEND);
            } catch (IOException e) {
                e.printStackTrace();
//                System.err.println("Error encountered while saving. Aborting.");
//                System.exit(-1);
            }
        });
    }

    public static String detweetify(String tweet) {
        String[] tld = {".org", ".net", ".com", ".gov", ".edu"};
        tweet = EmojiParser.removeAllEmojis(tweet);
        String detweetified = "";

        for(String token : tweet.split(" ")) {
            if(token.length() == 0) {
                continue;
            }
            if(token.charAt(0) == '@') {
                continue;
            }
            if(token.charAt(0) == '#') { // TODO: hashtag separation
                continue;
            }
            if(Arrays.stream(tld).anyMatch(token::contains)) {
                continue;
            }

            detweetified += token + " ";
        }

        if(detweetified.length() == 0) {
            return "";
        } else {
            return detweetified.substring(0, detweetified.length() - 1);
        }
    }


    public static ArrayList<Tweet> loadEntries(Path path, boolean isTest) throws IOException {
        Scanner scan = new Scanner(path).useDelimiter("([\\t\\n])");
        ArrayList<Tweet> arrayList = new ArrayList<>();

        scan.nextLine();
        if (!scan.hasNextLine()) {
            return null;
        }

        while (scan.hasNextLine()) {
            try {
                if(!isTest)
                    arrayList.add(new Tweet(scan.next(),scan.next(),scan.next()));
                else
                    arrayList.add(new Tweet(scan.next(),scan.next()));
            } catch (NoSuchElementException e) {
                break;
            }
        }

        scan.close();
        return arrayList;
    }

    public static class Tweet {
        String id, tweet,  label;

        public Tweet(String id, String tweet) {
            this.id = id;
            this.tweet = tweet;
        }

        public Tweet(String id, String tweet, String label) {
            this.id = id;
            this.tweet = tweet;
            this.label = label;
        }

        @Override
        public String toString() {
            if(this.label != null) {
                return this.id + "\t" + this.tweet + "\t" + this.label + "\n";
            } else {
                return this.id + "\t" + this.tweet + "\n";
            }
        }
    }
}