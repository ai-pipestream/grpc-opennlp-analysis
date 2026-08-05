/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.pipestream.opennlp.analysis;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisServiceGrpc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamDoc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamResponse;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest;
import ai.pipestream.opennlp.analysis.v1.GetVocabDriftRequest;
import ai.pipestream.opennlp.analysis.v1.GetVocabDriftResponse;
import ai.pipestream.opennlp.analysis.v1.GetVocabStatsRequest;
import ai.pipestream.opennlp.analysis.v1.GetVocabStatsResponse;
import ai.pipestream.opennlp.analysis.v1.SnapshotVocabRequest;
import ai.pipestream.opennlp.analysis.v1.SnapshotVocabResponse;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import ai.pipestream.opennlp.analysis.v1.VocabChannel;
import ai.pipestream.opennlp.analysis.v1.VocabularyServiceGrpc;
import ai.pipestream.opennlp.analysis.vocab.VocabularyListener;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;

/**
 * The vocabulary listener wired into the service: AnalyzeStream feeds it,
 * unary Analyze does not, GetCapabilities reports the flag, and
 * VocabularyService serves stats, snapshots, and drift.
 */
class VocabularyListenerServiceTest {

  @TempDir
  Path vocabDir;

  private VocabularyListener listener;
  private Server server;
  private ManagedChannel channel;
  private AnalysisServiceGrpc.AnalysisServiceBlockingStub analysis;
  private AnalysisServiceGrpc.AnalysisServiceStub analysisAsync;
  private VocabularyServiceGrpc.VocabularyServiceBlockingStub vocab;

  private static ServiceConfig config(Path vocabDir) {
    return new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null,
        null, null, null, null, null, null, 0, vocabDir, 1_000, 16);
  }

  private static AnalysisOptions termVectorOptions() {
    return AnalysisOptions.newBuilder()
        .setTermVectors(TermVectorOptions.newBuilder()
            .setEnabled(true)
            .setMode(TermVectorOptions.Mode.MODE_SCORING_ONLY))
        .build();
  }

  @BeforeEach
  void setUp() throws Exception {
    final ServiceConfig config = config(vocabDir);
    listener = new VocabularyListener(vocabDir, config.vocabWindowDocs(),
        config.vocabTopK(), null);
    final String name = InProcessServerBuilder.generateName();
    server = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(PipelineEnvironment.empty(), config,
            listener, List.of()))
        .addService(new VocabServiceImpl(listener))
        .build().start();
    channel = InProcessChannelBuilder.forName(name).directExecutor().build();
    analysis = AnalysisServiceGrpc.newBlockingStub(channel);
    analysisAsync = AnalysisServiceGrpc.newStub(channel);
    vocab = VocabularyServiceGrpc.newBlockingStub(channel);
  }

  @AfterEach
  void tearDown() {
    channel.shutdownNow();
    server.shutdownNow();
    listener.close();
  }

  /** Collects the responses and terminal signal of one stream. */
  private static final class Collector implements StreamObserver<AnalyzeStreamResponse> {
    final List<AnalyzeStreamResponse> responses = new CopyOnWriteArrayList<>();
    final AtomicReference<Throwable> error = new AtomicReference<>();
    final CountDownLatch done = new CountDownLatch(1);

    @Override
    public void onNext(AnalyzeStreamResponse response) {
      responses.add(response);
    }

    @Override
    public void onError(Throwable t) {
      error.set(t);
      done.countDown();
    }

    @Override
    public void onCompleted() {
      done.countDown();
    }

    void await() throws InterruptedException {
      assertThat(done.await(30, TimeUnit.SECONDS))
          .as("stream terminated within 30 s").isTrue();
    }
  }

  private void streamOneDocument(String text) throws Exception {
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests =
        analysisAsync.analyzeStream(collector);
    requests.onNext(AnalyzeStreamRequest.newBuilder()
        .setOptions(termVectorOptions()).build());
    requests.onNext(AnalyzeStreamRequest.newBuilder()
        .setDoc(AnalyzeStreamDoc.newBuilder().setSequence(1).setText(text)).build());
    requests.onCompleted();
    collector.await();
    assertThat(collector.error.get()).isNull();
    assertThat(collector.responses).hasSize(1);
    assertThat(collector.responses.get(0).hasOk()).isTrue();
  }

  @Test
  void analyzeStreamFeedsBothChannels() throws Exception {
    streamOneDocument("Court court ruling");

    final GetVocabStatsResponse stats =
        vocab.getVocabStats(GetVocabStatsRequest.getDefaultInstance());
    assertThat(stats.getEnabled()).isTrue();
    assertThat(stats.getWindowDocuments()).isEqualTo(1);
    assertThat(stats.getLiveChannelsList()).hasSize(2);
    final var terms = stats.getLiveChannelsList().stream()
        .filter(c -> c.getChannel() == VocabChannel.VOCAB_CHANNEL_TERMS)
        .findFirst().orElseThrow();
    final var tokens = stats.getLiveChannelsList().stream()
        .filter(c -> c.getChannel() == VocabChannel.VOCAB_CHANNEL_TOKENS)
        .findFirst().orElseThrow();
    // TERMS: the folded identity — "court" x2, "ruling" x1 under the default
    // case-folding steps.
    assertThat(terms.getDocuments()).isEqualTo(1);
    assertThat(terms.getTermOccurrences()).isEqualTo(3);
    assertThat(terms.getHeavyHittersList().stream()
        .map(h -> h.getTerm())).contains("court", "ruling");
    // TOKENS: the raw surface forms, pre-normalization.
    assertThat(tokens.getDocuments()).isEqualTo(1);
    assertThat(tokens.getTermOccurrences()).isEqualTo(3);
    assertThat(tokens.getHeavyHittersList().stream()
        .map(h -> h.getTerm())).contains("Court", "court", "ruling");
  }

  @Test
  void unaryAnalyzeDoesNotFeedTheListener() {
    analysis.analyze(AnalyzeRequest.newBuilder()
        .setText("Court court ruling")
        .setOptions(termVectorOptions())
        .build());

    assertThat(vocab.getVocabStats(GetVocabStatsRequest.getDefaultInstance())
        .getWindowDocuments()).isZero();
  }

  @Test
  void failedDocumentsAreNotCounted() throws Exception {
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests =
        analysisAsync.analyzeStream(collector);
    requests.onNext(AnalyzeStreamRequest.newBuilder()
        .setOptions(termVectorOptions()).build());
    requests.onNext(AnalyzeStreamRequest.newBuilder()
        .setDoc(AnalyzeStreamDoc.newBuilder().setSequence(1).setText("")).build());
    requests.onNext(AnalyzeStreamRequest.newBuilder()
        .setDoc(AnalyzeStreamDoc.newBuilder().setSequence(2).setText("a fine document"))
        .build());
    requests.onCompleted();
    collector.await();

    assertThat(collector.responses).hasSize(2);
    assertThat(vocab.getVocabStats(GetVocabStatsRequest.getDefaultInstance())
        .getWindowDocuments()).isEqualTo(1);
  }

  @Test
  void capabilitiesReportsTheFlag() {
    assertThat(analysis.getCapabilities(GetCapabilitiesRequest.getDefaultInstance())
        .getVocabListenerAvailable()).isTrue();
  }

  @Test
  void capabilitiesReportsTheFlagOffWithoutAListener() throws Exception {
    final String name = InProcessServerBuilder.generateName();
    final Server plain = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(PipelineEnvironment.empty(),
            config(vocabDir)))
        .build().start();
    final ManagedChannel plainChannel =
        InProcessChannelBuilder.forName(name).directExecutor().build();
    try {
      assertThat(AnalysisServiceGrpc.newBlockingStub(plainChannel)
          .getCapabilities(GetCapabilitiesRequest.getDefaultInstance())
          .getVocabListenerAvailable()).isFalse();
    } finally {
      plainChannel.shutdownNow();
      plain.shutdownNow();
    }
  }

  @Test
  void snapshotThenDriftAgainstTheLiveWindow() throws Exception {
    streamOneDocument("alpha alpha beta");
    final SnapshotVocabResponse snapshot =
        vocab.snapshotVocab(SnapshotVocabRequest.getDefaultInstance());
    assertThat(snapshot.getPersisted()).isTrue();
    assertThat(snapshot.getSnapshot().getDocuments()).isEqualTo(1);
    assertThat(snapshot.getSnapshot().getSequence()).isZero();

    streamOneDocument("gamma gamma alpha");
    final GetVocabDriftResponse drift = vocab.getVocabDrift(
        GetVocabDriftRequest.newBuilder()
            .setFrom(snapshot.getSnapshot().getName()).setTo("live").build());

    assertThat(drift.getChannelsList()).hasSize(2);
    final var terms = drift.getChannelsList().stream()
        .filter(c -> c.getChannel() == VocabChannel.VOCAB_CHANNEL_TERMS)
        .findFirst().orElseThrow();
    // One of the two live-window terms (gamma) is new relative to the
    // snapshot, so roughly half the live vocabulary is novel.
    assertThat(terms.getNoveltyRate()).isBetween(0.3, 0.7);
    assertThat(terms.getFromCardinality()).isGreaterThan(1.0);
    assertThat(terms.getToCardinality()).isGreaterThan(1.0);
    assertThat(terms.getUnionCardinality())
        .isGreaterThanOrEqualTo(terms.getToCardinality());

    // The bare sequence number resolves too.
    assertThat(vocab.getVocabDrift(GetVocabDriftRequest.newBuilder()
        .setFrom("0").setTo("live").build()).getChannelsList()).hasSize(2);
  }

  @Test
  void unknownSnapshotReferenceIsInvalidArgument() {
    assertThatThrownBy(() -> vocab.getVocabDrift(GetVocabDriftRequest.newBuilder()
        .setFrom("snapshot-99-1.pb").setTo("live").build()))
        .isInstanceOfSatisfying(StatusRuntimeException.class,
            e -> assertThat(e.getStatus().getCode())
                .isEqualTo(Status.Code.INVALID_ARGUMENT));
  }

  @Test
  void snapshotOnAnEmptyWindowReportsNotPersisted() {
    final SnapshotVocabResponse response =
        vocab.snapshotVocab(SnapshotVocabRequest.getDefaultInstance());
    assertThat(response.getPersisted()).isFalse();
    assertThat(response.hasSnapshot()).isFalse();
  }
}
