package ai.pipestream.opennlp.analysis;

import static org.assertj.core.api.Assertions.assertThat;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import java.io.File;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * OPENNLP_NER_MODEL names a list, because the stock English finders cover
 * one entity type each and a deployment usually wants several.
 *
 * <p>The failure this guards against is quiet rather than loud: a service
 * given one model answers every ner request, reports ner_available, and
 * simply never returns the types it has no model for.</p>
 */
class NerModelListTest {

  private static ServiceConfig withNerSetting(String value) {
    final String previous = System.getProperty("opennlp.ner.model");
    try {
      if (value == null) {
        System.clearProperty("opennlp.ner.model");
      } else {
        System.setProperty("opennlp.ner.model", value);
      }
      return ServiceConfig.fromEnvironment(new String[] {"0"});
    } finally {
      if (previous == null) {
        System.clearProperty("opennlp.ner.model");
      } else {
        System.setProperty("opennlp.ner.model", previous);
      }
    }
  }

  @Test
  void severalModelsAreAcceptedOnThePathSeparatorAndOnCommas() {
    final String sep = File.pathSeparator;
    assertThat(withNerSetting("/m/person.bin" + sep + "/m/location.bin").nerModelPaths())
        .containsExactly(Path.of("/m/person.bin"), Path.of("/m/location.bin"));

    // A comma is accepted too: the platform separator is ':' on Linux and
    // ';' on Windows, and a configuration file that travels between them
    // should not have to change.
    assertThat(withNerSetting("/m/person.bin,/m/location.bin").nerModelPaths())
        .containsExactly(Path.of("/m/person.bin"), Path.of("/m/location.bin"));

    // Order is the caller's, since it is the order mentions come back in.
    assertThat(withNerSetting("/m/b.bin,/m/a.bin").nerModelPaths())
        .containsExactly(Path.of("/m/b.bin"), Path.of("/m/a.bin"));
  }

  @Test
  void aSingleModelStillWorksAndBlankMeansNone() {
    assertThat(withNerSetting("/m/person.bin").nerModelPaths())
        .containsExactly(Path.of("/m/person.bin"));

    // Unset means no NER at all rather than one empty path, which is what
    // every other model setting means by absent.
    assertThat(withNerSetting(null).nerModelPaths()).isEmpty();
    assertThat(withNerSetting("   ").nerModelPaths()).isEmpty();

    // Stray separators are ignored rather than turned into empty paths,
    // so a trailing colon from shell assembly is not a startup failure.
    assertThat(withNerSetting("/m/person.bin" + File.pathSeparator).nerModelPaths())
        .containsExactly(Path.of("/m/person.bin"));
  }

  @Test
  void aNullListIsNormalizedSoCallersReadACollection() {
    // Every other model setting says "absent" with null, so callers pass
    // null here too; the record turns it into an empty list at the
    // boundary rather than making every reader null-check a collection.
    final ServiceConfig config = new ServiceConfig(
        0, 1024, null, null, (java.util.List<Path>) null,
        null, null, null, null, null, null, null, null, null, 0);
    assertThat(config.nerModelPaths()).isEmpty();
  }
}
