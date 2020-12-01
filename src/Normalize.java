import java.io.*;
import java.nio.file.*;
import java.util.*;
import zemberek.morphology.TurkishMorphology;
import zemberek.normalization.TurkishSentenceNormalizer;

public class Normalize {
    public static void main(String[] args) {
        if(args.length != 1) {
            System.err.println("Format: Normalize <input path> <output path");
        }

        ArrayList<String> tweets = loadEntries(Paths.get(args[0]));
    }

    public Map<String, Entry> loadEntries(Path path) throws IOException {
        Scanner scan = new Scanner(path).useDelimiter("([\\t\\n])");
        ArrayList<String> arrayList = new ArrayList<>();

        scan.nextLine();
        if (!scan.hasNextLine()) {
            return null;
        }

        while (scan.hasNextLine()) {
            try {
                scan.next();
                arrayList.add(scan.next());
                scan.next();
            } catch (InputMismatchException e) {
                throw new IOException("Unexpected entry.")
            }
        }

        scan.close();
        return arrayList;
    }

}